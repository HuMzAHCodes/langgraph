# Lab Report — Persistence in LangGraph

## Topic

Giving a LangGraph workflow **memory that survives beyond a single `invoke()`
call** — using a checkpointer to save state after every node, organized into
independent conversation "threads," with the ability to inspect history, rewind
to any earlier point ("time travel"), edit past state, and recover cleanly from a
crash mid-execution.

## Why We Needed It

Every graph built before this lab shared the same limitation: state lived only
in Python memory for the duration of one `invoke()` call. The moment that call
returned, the state was gone. Call the same compiled graph again and it starts
completely fresh — no memory of anything that happened before.

That's fine for a single blog-outline generator or a one-shot tweet loop. It
breaks down the moment a workflow needs to:

- **Remember a conversation across many separate calls** — a chatbot that
  recalls what the user said five messages ago, without re-sending the entire
  history by hand each time.
- **Survive a crash** — a long-running workflow (agent tool calls, multi-step
  pipelines) that gets interrupted by a server restart or network failure
  shouldn't have to restart from step one.
- **Support multiple independent users at once** — one deployed graph serving
  many separate conversations simultaneously, each with its own isolated
  history.
- **Be inspectable and reversible** — the ability to look back at exactly what
  happened at each step, and even change a past decision and see how the
  outcome differs.

A **checkpointer** solves all of this in one mechanism: instead of state living
only in a Python variable, it's saved to durable storage after every single node
execution, tagged with an identifier called a `thread_id`.

## Theoretical Foundations

### The Checkpointer

A checkpointer is an object attached at graph-compile time:

```python
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
```

From that point on, LangGraph automatically saves a **checkpoint** — a full
snapshot of the graph's state — after every node finishes running, not just at
the end. This is fundamentally different from a chain or an unchecked graph,
where only the final return value is ever visible.

`InMemorySaver` keeps checkpoints in the Python process's memory, so they're
lost when the process ends — it exists to demonstrate the *mechanism* safely and
cheaply. In production, the same interface is backed by durable storage
(Postgres, Redis, SQLite, etc.), so checkpoints genuinely persist across
restarts, deployments, and days. The concepts learned here — threads, history,
time travel, resumability — apply identically regardless of which storage
backend sits underneath.

### Threads — the isolation unit

Every call to a checkpointed graph must supply a `config` containing a
`thread_id`:

```python
config1 = {"configurable": {"thread_id": "1"}}
workflow.invoke({'topic': 'pizza'}, config=config1)
```

A thread is the unit of memory isolation — think of it as a conversation ID or
a session ID. All checkpoints saved during calls that share a `thread_id`
belong to the same continuous history. A different `thread_id` (`config2`)
gets its own, entirely separate history; invoking the graph under `thread_id:
"2"` has zero effect on `thread_id: "1"`'s state. This is the mechanism a real
deployed chatbot uses to keep many users' conversations from bleeding into each
other while all being served by the same compiled graph object.

### Reading state — `get_state` and `get_state_history`

Two read operations expose what's been saved:

- **`workflow.get_state(config)`** returns the *latest* checkpoint for that
  thread — the most recent snapshot of state, plus metadata about which node
  produced it.
- **`workflow.get_state_history(config)`** returns *every* checkpoint ever
  saved for that thread, in order — one entry per node execution, not just the
  final result. This full history is what makes everything below possible: you
  can't time-travel to a moment that wasn't recorded.

### Time Travel

Because every intermediate checkpoint is retained, LangGraph lets you resume
execution from **any past checkpoint**, not just the latest one:

```python
workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": "<a past checkpoint's id>"}})
```

Passing `None` as the input (instead of a fresh state dict) combined with a
specific `checkpoint_id` tells LangGraph: "don't start over — resume exactly
from this saved point." The graph continues forward from there, using the
state as it existed at that moment. This is conceptually identical to
`git checkout <commit>` — you're moving the "current position" back to an
earlier point in the recorded history and continuing from there.

### Updating State — rewriting history and branching

`update_state` goes a step further than time travel — it lets you *change* the
data in a past checkpoint before resuming:

```python
workflow.update_state(
    {"configurable": {"thread_id": "1", "checkpoint_id": "<a past checkpoint's id>", "checkpoint_ns": ""}},
    {'topic': 'samosa'}
)
```

This doesn't destroy the original checkpoint — it creates a **new checkpoint**
that branches off the edited point, carrying the modified field forward. Continuing
execution from this new checkpoint produces a genuinely different downstream
result (a different joke and explanation, generated for the *new* topic),
proving the state was actually rewritten, not merely relabeled. This is the
same branching model version control uses: edit history at a point, and you get
a new branch, while the original line of history remains intact and inspectable.

### Fault Tolerance

The most practically important application of checkpointing: **automatic
recovery from a crash mid-workflow.**

Because every node's result is checkpointed the instant it completes, a
checkpointed graph that crashes partway through doesn't need to restart from
node one. Calling `invoke(None, config=...)` again — with the *same*
`thread_id` — resumes from the last successfully completed checkpoint,
re-running only the steps that hadn't finished yet.

The lab demonstrates this directly: a 3-step graph where step 2 deliberately
hangs (`time.sleep(1000)`), simulating a real-world crash when manually
interrupted. Re-invoking afterward with the same thread does **not** re-execute
step 1 — its result was already durably saved before the interruption. Execution
resumes precisely at step 2. This is the property that makes checkpointed
LangGraph workflows suitable for long-running, multi-step processes (agent tool
chains, multi-stage pipelines) where a mid-run failure would otherwise mean
losing all completed work and starting over from scratch.

## What Each Section of the Lab Demonstrates

**Basic checkpointing (Cells 1–12):** a two-node joke-generation graph compiled
with `InMemorySaver`. Running it under two different `thread_id`s proves thread
isolation — each thread's state is independent and unaffected by the other's
activity, even though both run through the exact same compiled graph.

**Time travel (Cells 13–16):** fetching state from a specific past
`checkpoint_id`, then resuming execution from that exact point rather than the
latest one.

**Updating state (Cells 17–20):** editing a past checkpoint's data
(`topic: 'samosa'`) and resuming from the newly created, edited checkpoint —
producing output for the changed topic, demonstrating that history can be
rewritten and branched, not just replayed.

**Fault tolerance (Cells 21–29):** a three-step graph with a deliberately
hanging middle step, manually interrupted to simulate a crash, then resumed —
proving the graph picks up exactly where it left off rather than restarting.

## Practical Notes From Running This Lab

- **Checkpoint IDs are unique per run.** Unlike every prior lab, cells that
  reference a specific `checkpoint_id` cannot be copy-pasted verbatim from a
  tutorial — each ID is a UUID generated fresh by *your own* execution. The
  correct workflow is: run `get_state_history()`, read the real ID from its
  output, then paste that ID into the next cell.
- **`InMemorySaver` is a teaching tool, not a production choice.** Its state
  vanishes when the Python process ends (e.g., the Colab runtime restarts).
  Real deployments swap in a persistent backend (Postgres, SQLite, Redis)
  behind the identical `checkpointer=` interface — none of the thread/history/
  time-travel/resume concepts change when that swap happens.
- **The fault-tolerance demo requires a manual action.** The hang in step 2 is
  intentional; you must click the notebook's interrupt/stop control while it's
  sleeping to simulate the crash, then re-run the invoke cell to see the
  recovery.

## Conclusion

Persistence is the concept that turns a LangGraph workflow from something that
"runs once and forgets" into something with real, inspectable, editable memory.
Threads provide isolation between separate conversations or sessions; full
checkpoint history enables both time travel (resume from any past point) and
state editing (rewrite and branch history); and because every node's result is
saved the moment it completes, a checkpointed workflow can survive a real crash
and resume exactly where it left off, rather than losing all completed work.
This is the foundation every production-grade conversational agent or
long-running multi-step LangGraph pipeline is built on.
