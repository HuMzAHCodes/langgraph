# Lab Report — Long-Term Memory Implementation: Read, Write, Deduplicate, Merge

## Topic

Four progressively-built chatbot patterns, each demonstrating one piece of a
complete long-term memory system: reading existing memories to personalize
responses, writing new memories automatically from conversation, avoiding
duplicate memories when a fact is restated, and finally merging read and
write into a single working chatbot.

## Why We Needed This

The earlier LTM basics lab covered the store's raw API in isolation —
`put`, `get`, `search`, `delete` — with no LLM involved. That's necessary
groundwork, but a real chatbot needs the store wired into an actual
conversation: reading relevant memories before answering, and deciding what's
worth remembering from what the user just said, automatically, without a
human manually calling `store.put()`. This lab builds that connection one
capability at a time, rather than jumping straight to the full system, so
each moving part can be understood and tested on its own before combining
them.

## Theoretical Foundations

### Part 1 — Reading Existing Memories (Personalization, Read-Only)

```python
def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    user_details = ("user", user_id, "details")
    items = store.search(user_details)

    user_details_content = "\n".join(f"- {it.value.get('data', '')}" for it in items) if items else ""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details_content)
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}
```

This section seeds the store manually (`store.put(...)` called directly in a
setup cell, standing in for memories that a real system would have
accumulated over many past conversations), then builds a graph with a single
node that only **reads** — it never writes anything back to the store. The
system prompt template is the actual personalization mechanism: it explicitly
instructs the model to address the user by name, reference known projects,
and avoid generic phrasing whenever personal context is available, with the
retrieved memory content substituted into `{user_details_content}`. This
proves the read half of the system works before any writing logic is
introduced.

### Part 2 — Creating New Memories (Write-Only)

```python
class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[str] = Field(default_factory=list, description="Atomic user memories to store")

memory_extractor = extractor_llm.with_structured_output(MemoryDecision)

def remember_only_node(state, config, store):
    decision = memory_extractor.invoke([...])
    if decision.should_write:
        for mem in decision.memories:
            store.put(namespace, str(uuid.uuid4()), {"data": mem})
    return {"messages": [{"role": "assistant", "content": "Noted."}]}
```

The inverse of Part 1: this node never reads or uses memory to answer — its
only job is deciding, via a **separate LLM call with structured output**,
whether the user's latest message contains anything worth remembering
long-term (identity, stable preferences, ongoing projects), and if so, saving
each as an atomic fact under a fresh UUID key. The node's own reply is a
fixed, generic "Noted." — deliberately not using the LLM to converse at all,
isolating the write mechanic for inspection. `temperature=0` on the
extractor LLM keeps this structured decision consistent rather than
creatively varied, since it's a classification task, not a conversational one.

**Key idea:** deciding *what* to remember is itself an LLM task, using
`with_structured_output` (the same mechanism from the very first
structured-output labs in this series) to get a reliable `should_write` /
`memories` decision rather than free-form text.

### Part 3 — Creating New Memories Without Duplication

```python
class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW ... False if duplicate/already known.")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)
```

Part 2's writer has an obvious flaw: restate a fact you already told it
("I like Python for programming" after already saying so) and it would save
a near-duplicate memory every time. Part 3 fixes this by giving the extractor
LLM the **existing memories as context** (`MEMORY_PROMPT` includes
`{user_details_content}`, populated from `store.search(namespace)` before the
extraction call), and asking it to mark each candidate memory `is_new=True`
or `is_new=False` by comparing against what's already stored. Only items
marked `is_new` get written. This is a genuinely different structured-output
shape from Part 2 — a *list of typed objects* (`MemoryItem`, each carrying
its own `is_new` flag), not just a list of plain strings, letting the write
node filter selectively instead of writing everything unconditionally.

### Part 4 — Merged Workflow (Remember, Then Chat)

```python
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)
```

The complete system: **every turn now does both**, in sequence — first
`remember_node` (Part 3's dedup-aware writer, but returning `{}` instead of a
message, since chat's actual reply comes from the next node) extracts and
saves any new facts from what the user just said, *then* `chat_node` (Part
1's reader) retrieves the now-updated memory set and generates a
personalized reply using it. Because writing happens before reading within
the same turn, a fact stated in a message can influence the *same* turn's
response, not just future ones — though in the demo, the model's fixed
"Noted."-style acknowledgment was replaced with `chat_node`'s real
conversational reply, so each turn both learns and responds naturally in
one pass.

## Personalization Details Used in This Lab

For this run, the seeded and extracted memories reflected the user's own
stated details rather than the tutorial's original example: name **Hamza**,
role **final-year Software Engineering student at NUST**, and ongoing project
**the labgGraph LangChain/LangGraph chatbot project** — substituted
throughout the system prompt's personalization examples and the manually
seeded memory records in Part 1.

## Structural Comparison

```
PART   READS MEMORY?   WRITES MEMORY?   DEDUPLICATES?   REPLIES CONVERSATIONALLY?
-----  --------------  ---------------  --------------  -------------------------
1      Yes             No               n/a             Yes (personalized)
2      No               Yes             No              No (fixed "Noted.")
3      Yes (for compare) Yes             Yes             No (fixed "Noted.")
4      Yes              Yes             Yes             Yes (personalized)
```

## Conclusion

Building long-term memory as four separate, incrementally-composed patterns
— rather than the full merged system from the start — isolates exactly what
each piece contributes: reading memory is what makes responses *feel*
personalized; writing memory is what lets facts survive beyond a single turn;
deduplication is what keeps that memory from growing noisy and redundant over
a long-running relationship with a user; and the merged workflow is simply
those three capabilities composed into one graph, read after write, on every
turn. This progression mirrors the store's basic CRUD operations from the
earlier basics lab, now applied inside real chat nodes with an LLM making the
actual read and write decisions rather than hardcoded logic.
