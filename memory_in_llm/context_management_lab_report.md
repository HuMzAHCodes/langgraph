# Lab Report — Managing Conversation History: Trimming, Deletion, and Summarization

## Topic

Three distinct strategies for controlling an ever-growing conversation history
in LangGraph, each solving the same underlying problem — an unbounded
`messages` list — with a different tradeoff between simplicity, cost, and
information loss: **trimming** (filter what's sent, keep everything stored),
**deletion** (permanently remove old messages from state), and
**summarization** (compress old messages into a condensed summary before
deleting them).

## Why We Needed This

Every earlier chatbot lab in this series used `add_messages` to let
`state["messages"]` grow without limit — every turn appends, nothing is ever
removed. That's fine for a short demo, but it breaks down in two ways as a
real conversation gets longer: the message list can eventually exceed the
model's context window entirely, and even before that hard limit, sending the
*entire* history on every single call gets progressively more expensive
(more input tokens, more latency) for diminishing benefit, since very early
messages are often no longer relevant to the current question. All three
techniques in this lab exist to keep a long-running conversation usable and
affordable without simply discarding useful context.

## Theoretical Foundations

### 1. Trimming — filter what's sent, touch nothing stored

```python
def chat_node(state: ChatState):
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
    )
    response = llm.invoke(trimmed_messages)
    return {"messages": [response]}
```

`trim_messages()` operates entirely at the point of calling the LLM. It never
touches `state["messages"]` itself — the full history keeps accumulating in
state exactly as it always has, so `get_state_history`, resuming a thread, and
any auditing of the complete conversation all remain fully intact. Each
individual LLM call, however, only sees a **trimmed window**: `strategy="last"`
keeps the most recent messages, `token_counter=llm` uses the model's own
tokenizer to count accurately (rather than guessing by character count), and
`allow_partial=False` ensures a message is either fully included or fully
dropped, never cut in half.

**Tradeoff:** cheapest and safest — nothing is ever lost — but the *stored*
history keeps growing unbounded regardless. Trimming controls cost and
context-window pressure per call; it does not control storage growth over
time.

### 2. Deletion — permanently remove old messages from state

```python
def delete_old_messages(state: ChatState):
    messages = state["messages"]
    if len(messages) <= 2:
        return {}
    messages_to_delete = messages[:-2]
    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_delete]}
```

`RemoveMessage(id=...)` is a special message type the `add_messages` reducer
specifically recognizes: instead of appending like a normal message, the
reducer matches the given `id` against an existing stored message and
**deletes** it. This is the only mechanism among the three that actually
shrinks `state["messages"]` — after this node runs, those messages are
genuinely gone from the checkpointed state, not merely hidden from a given
LLM call.

Structurally, deletion is implemented as its **own dedicated graph node**
(run after the chat node, via `add_edge("chat_node", "delete_old_messages")`),
keeping "answer the question" and "manage history size" as separate concerns
that could be swapped or modified independently.

**Tradeoff:** genuinely reduces storage and keeps per-call context small, but
it is **irreversible** at the state level — once removed, that content cannot
be recovered from the thread's history. Suitable when old raw context is
truly disposable; unsuitable if a complete audit trail is ever required.

### 3. Summarization — compress old history before discarding it

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

def chat_node(state: ChatState):
    summary = state.get("summary", "")
    if summary:
        messages = [SystemMessage(content=f"Summary of earlier conversation: {summary}")] + state["messages"]
    else:
        messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def summarize_node(state: ChatState):
    summary = state.get("summary", "")
    summary_prompt = (
        f"This is the summary of the conversation so far: {summary}\n\nExtend it with the new messages above."
        if summary else
        "Create a summary of the conversation above."
    )
    response = llm.invoke(state["messages"] + [HumanMessage(content=summary_prompt)])
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_summarize(state: ChatState):
    return "summarize_node" if len(state["messages"]) > 6 else END
```

This is the most sophisticated of the three, combining ideas from both prior
approaches with a new one:

- **A new state field, `summary`**, carries condensed older context forward —
  something neither trimming nor plain deletion has any equivalent of.
- **A conditional routing function** (`should_summarize`, wired via
  `add_conditional_edges`) decides, based on message count, whether to
  compress history at all on a given turn — most turns skip straight to
  `END`; only once the threshold is crossed does `summarize_node` run.
- **`summarize_node` extends, not replaces, the existing summary** — if one
  already exists, the prompt explicitly asks the model to extend it with the
  newest messages, rather than starting over from scratch each time, so the
  summary itself accumulates understanding of the whole conversation as it
  goes.
- **The same `RemoveMessage` deletion mechanic from technique 2** is reused
  here to actually shrink `state["messages"]` after producing the summary —
  keeping only the last 2 raw messages, with everything older now represented
  only in condensed form.
- **`chat_node` re-injects the summary as a `SystemMessage`** on every
  subsequent call, which is what lets the model still correctly answer
  questions about early conversation content even after the original raw
  messages have been deleted — the model never sees the original text again,
  only the summary standing in for it.

**Tradeoff:** the most implementation complexity of the three, and some
fidelity is inevitably lost in the summarization step (a summary is not a
perfect substitute for the original text) — but it's the only approach that
keeps storage bounded *and* preserves the model's ability to recall
early-conversation facts indefinitely, rather than either discarding them
(deletion alone) or paying to resend them forever (trimming alone, at the
storage level).

## Comparison

```
TECHNIQUE       STORED STATE SHRINKS?   INFO LOST?           NEW STATE FIELDS
--------------  ----------------------  -------------------  ----------------
Trimming        No (grows forever)      No                   none
Deletion        Yes                     Yes, permanently      none
Summarization   Yes                     Partial (compressed)  summary
```

## Visualizing Each Graph

All three graphs were visualized with:
```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

print(graph.get_graph().draw_mermaid())   # text fallback, always works
```

The trimming graph is a straight line (`START → chat_node → END`) since
trimming happens inline inside the one node. The deletion graph adds a second
node in sequence (`START → chat_node → delete_old_messages → END`). The
summarization graph is the only one with a genuine **branch**: a conditional
edge out of `chat_node` that goes to either `summarize_node` or directly to
`END`, depending on `should_summarize`'s decision at runtime — visibly
distinguishing it from the other two, which have no branching at all.

## Testing Notes

Getting summarization to visibly trigger required either running enough
turns to naturally cross the `> 6` message threshold, or temporarily lowering
that threshold (e.g. to `> 2`) to see the effect fire on nearly every turn —
useful for observing the summary text actually extend itself call over call.
The definitive proof that the summary is genuinely being *used* (not merely
stored) is asking a question late in the conversation that only makes sense
if the model recalls an early fact whose original message has since been
deleted — a correct answer there confirms the `SystemMessage` injection in
`chat_node` is doing real work, since the original text is no longer
available to the model by any other means.

## Conclusion

Trimming, deletion, and summarization all address the same core problem —
letting a conversation grow indefinitely without letting either its cost or
its storage grow unbounded — but at three different points along a
tradeoff between simplicity and information preservation. Trimming is the
safest and simplest (never lose anything, control only what's sent per call).
Deletion is the most aggressive (genuinely bounded storage, but permanent
loss). Summarization sits between them: bounded storage like deletion, but
with a compressed record standing in for what would otherwise be lost
entirely — at the cost of being the most complex to implement, requiring a
dedicated state field, a conditional routing node, and coordination between
two node functions. Which to use in a real system depends on whether the
priority is minimizing cost per call, minimizing storage, or preserving the
model's ability to recall the full arc of a long conversation.
