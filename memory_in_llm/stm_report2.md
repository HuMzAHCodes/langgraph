# Lab Report — Short-Term vs Long-Term Memory in LangGraph, and Why Docker

## Topic

Two complementary memory mechanisms in LangGraph, both built on the same
`store` interface (`store.search` / `store.put`), differing only in where the
data actually lives:

- **Short-term memory** — `InMemoryStore`, data lives in RAM, gone when the
  process restarts. Run on Google Colab.
- **Long-term memory** — `PostgresStore`, data lives in a real Postgres
  database, surviving restarts, reboots, and days passing — brought up
  locally via Docker, and run **locally** (not Colab — see the environment
  section below for why).

## Why We Needed a New Memory Mechanism (Beyond the Checkpointer)

Every earlier LangGraph lab in this series used a **checkpointer**
(`InMemorySaver` → `SqliteSaver` → `AsyncSqliteSaver`) to persist
**conversation history**, scoped to one `thread_id`. That solved "remember
what was said earlier in *this* chat" — but it does not solve a different,
common need: **remember a fact about a user across completely different
conversations.**

The lab proves this distinction directly. In both notebooks:

```python
config  = {"configurable": {"thread_id": "1", "user_id": "1"}}   # thread 1
config2 = {"configurable": {"thread_id": "2", "user_id": "1"}}   # thread 2, SAME user
```

Thread `"1"` is told "remember that I like blue." Thread `"2"` is a **brand
new conversation** — as far as the checkpointer is concerned, it shares zero
message history with thread `"1"`. Yet asking thread `"2"` "what color do I
like?" still correctly answers "blue" ("You mentioned that you like the color
blue!"). That's only possible because the fact was saved somewhere **not
scoped to a thread at all** — the `store`, keyed by `user_id` instead.

## Theoretical Foundations

### The `store` — a second, independent memory alongside the checkpointer

```python
def chat_node(state: ChatState, config, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])
    ...
    if "remember" in last_message.content.lower():
        memory = f"User said: {last_message.content}"
        store.put(namespace, str(len(memories)), {"data": memory})
```

- **`store: BaseStore`** in a node function's signature is how LangGraph
  *injects* the store automatically — it's never passed manually at
  `.invoke()` time, only wired in once at `.compile(store=...)`.
- **`namespace`** is a tuple acting like a folder path, grouping related
  keys together. Using `("memories", user_id)` scopes every fact to a specific
  user, so different users' memories never collide — deliberately chosen
  independent of `thread_id`.
- **`store.search(namespace)`** is the read side — retrieve everything saved
  under that namespace, regardless of which thread is currently active.
- **`store.put(namespace, key, value)`** is the write side — save a new fact.
- The **decision rule** here ("does the message contain the word 'remember'?")
  is deliberately simple, to keep the mechanic visible. A production system
  would use an LLM to judge what's actually worth remembering, rather than a
  keyword match.

### Short-term: `InMemoryStore`

```python
store = InMemoryStore()
graph = graph_builder.compile(checkpointer=checkpointer, store=store)
```

Plain, ready-to-use object — no setup, no external service. Fast and free to
experiment with, but everything it holds vanishes the moment the Python
process (or Colab runtime) ends. This is the "prove the mechanism cheaply"
version, exactly mirroring how `InMemorySaver` served that role for
checkpointing earlier in the series.

### Long-term: `PostgresStore` — same interface, durable backend

```python
with PostgresStore.from_conn_string(DB_URI) as store, \
     PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    store.setup()
    checkpointer.setup()

    graph = graph_builder.compile(checkpointer=checkpointer, store=store)
    ...
```

Nothing about *how the node uses the store* changes — `store.search()` and
`store.put()` are called identically. What changes is that this store now
writes to a real Postgres database, so a memory like "likes blue" survives a
full restart of the notebook, the machine, or days passing — the same
upgrade path as `InMemorySaver` → `SqliteSaver` in the earlier persistence
lab, applied here to the *store* rather than the checkpointer.

Two things specific to the Postgres backend, both different from every
in-memory version used before:

- **`from_conn_string(...)` returns a context manager**, not a plain object —
  because a Postgres connection is a real external resource that must be
  properly opened and closed. Every `.invoke()` call has to happen **inside**
  the `with` block; once it exits, the connection closes and the graph can no
  longer read or write.
- **`.setup()`** creates the underlying database tables the first time you
  connect to a fresh database (for memories, threads, checkpoints). Safe to
  call on every run — it's a no-op once the tables already exist.

## Why Docker, Specifically

`PostgresStore`/`PostgresSaver` need an actual running Postgres **server** to
connect to — not a Python library, a real database service listening on a
network port. Installing and configuring Postgres directly on a machine is
its own multi-step process (install the right version, initialize a data
directory, configure authentication, manage the service). Docker sidesteps
all of that: a `docker-compose.yml` file describes exactly the Postgres
instance needed, and a couple of commands bring up a fully configured,
disposable, isolated database server in seconds:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    ports:
      - "5442:5432"
```

- **`image: postgres:16`** — Docker pulls a ready-made, official Postgres 16
  image; no manual installation of Postgres itself on the host machine.
- **`environment`** — sets the database's username, password, and default
  database name at first startup, matching exactly what the notebook's
  connection string expects.
- **`ports: "5442:5432"`** — maps the container's internal Postgres port
  (`5432`, Postgres's standard port) to port `5442` on the host machine, which
  is why the notebook connects to `localhost:5442`, not `5432` — the mapping
  avoids clashing with any other Postgres instance that might already be
  running on the default port.

Commands used to bring it up and confirm it's running:
```
docker compose up -d     # start the Postgres container in the background
docker ps                 # confirm it's actually running
```

Docker Desktop itself must be launched and its engine fully started before
either command works — attempting `docker compose up -d` while Docker Desktop
is closed fails with a "failed to connect to the docker API" pipe error, since
there's no daemon listening yet.

## Why the Long-Term Memory Notebook Had to Run Locally, Not Colab

The Docker container runs on the **local machine** (wherever Docker Desktop
is installed). A notebook connecting to `localhost:5442` only reaches that
container if the notebook process is *also* running on that same machine.
Google Colab runs on a remote Google-hosted VM — `localhost` inside that VM
refers to the VM itself, which has nothing listening on port 5442. Attempting
the connection from Colab fails with `OperationalError: connection to server
at "127.0.0.1", port 5442 failed: Connection refused` — not a code bug, a
fundamental environment mismatch. This is why the long-term memory notebook,
unlike every other notebook in this series, was run **locally** in VS Code,
on the same machine as Docker Desktop.

## Environment Setup Gotchas (Local Notebook Execution)

Getting the notebook running locally in VS Code surfaced a real, recurring
class of problem worth documenting: **VS Code's Jupyter kernel picker
defaulting to the wrong Python environment**, silently using packages from an
unrelated older project instead of the intended one.

- **Symptom:** `ModuleNotFoundError` for a package that had already been
  `pip install`-ed, because the notebook's kernel was actually a *different*
  venv (`LangChain_Models\venv`) than the one packages were installed into
  (`labgGraph\myenv`). Confirmed via `import sys; print(sys.executable)`,
  which is the definitive way to check which interpreter a running kernel is
  actually using — the kernel picker's displayed name is not always trustworthy.
- **A near-miss:** pasting a full interpreter path directly into the kernel
  picker's environment-manager search box (rather than its dedicated
  "Enter Interpreter Path..." option) triggered "Quick Create venv" instead,
  creating a brand-new empty environment rather than selecting the existing one.
- **What did NOT reliably fix it:** registering the venv as a named Jupyter
  kernel via `python -m ipykernel install --user --name=... --display-name
  ...` — the kernel was confirmed registered (`jupyter kernelspec list`
  showed it), but VS Code's kernel picker still did not surface it, even after
  a window reload.
- **What did fix it:** setting `python.venvPath` and
  `python.defaultInterpreterPath` in VS Code's User Settings (JSON) to point
  at the project root and the exact venv's `python.exe`, followed by a full
  restart of VS Code (not just "Reload Window"). This made the correct venv
  appear as the "Recommended" option in `Python: Select Interpreter`, and
  from there it became selectable in the notebook's "Select Another Kernel..."
  → "Python Environments..." list.
- **A separate, empty-notebook pitfall:** attempting to batch-run the
  notebook via `jupyter nbconvert --execute --inplace` failed with
  `NotJSONError` / `Expecting value: line 1 column 1` because the `.ipynb`
  file itself was an empty shell (`"cells": []`) — the intended cells had
  never actually been pasted into that file and saved, despite believing they
  had been. A quick sanity check (`Get-Content <file> | Measure-Object -Line`,
  expecting well over a handful of lines for a real notebook) would have
  caught this immediately.

## Result

Both notebooks demonstrated the same proof: a memory saved in thread `"1"`
correctly surfaces when queried from an entirely separate thread `"2"`, for
the same `user_id`. The short-term version (Colab, `InMemoryStore`) confirmed
the mechanism works; the long-term version (local, `PostgresStore` backed by
Docker) confirmed the same mechanism with a durable backend, ending with the
model correctly replying: *"You mentioned that you like the color blue!"* —
generated from a thread with no shared message history, proving the memory
came from the store, not the checkpointer.

## Conclusion

The checkpointer and the store solve two different memory problems that are
easy to conflate: the checkpointer remembers **what happened in a
conversation**, scoped per thread; the store remembers **facts about a user**,
scoped however you design the namespace — independent of any one thread. Both
follow the identical durability upgrade pattern already seen elsewhere in this
series (swap the in-memory backend for a persistent one, keep the same
interface), and Docker's role is purely infrastructural: it's the fastest way
to stand up a real, disposable Postgres server locally, without installing and
configuring Postgres by hand. The environment friction encountered along the
way — a misconfigured VS Code kernel silently using the wrong virtual
environment — is itself a useful lesson: `sys.executable` is the one
unambiguous way to confirm which Python a running notebook is actually using,
and it's worth checking early whenever package-not-found errors persist
despite a seemingly successful `pip install`.
