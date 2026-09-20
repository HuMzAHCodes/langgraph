# Notes: LangGraph + Streamlit chatbot with local and remote MCP servers

Study notes for one task: a chatbot that uses tools from **two MCP servers**, one running on this computer and one hosted on the internet. The task covers two files:

| File | Job |
|---|---|
| `langgraph_mcp_backend.py` | The brain: LLM, tools, MCP connections, the LangGraph graph, saved chat history |
| `streamlit_mcp_frontend.py` | The face: chat window, sidebar of past chats, live streaming of answers |

Run it with `streamlit run streamlit_mcp_frontend.py` from this folder. The frontend imports the backend, so starting the app starts everything.

---

## 1. Big picture

```
 You (browser)
     |
 streamlit_mcp_frontend.py      <- UI, streaming, chat list
     |  imports chatbot
 langgraph_mcp_backend.py
     |-- LLM: Mistral (ministral-8b-latest)
     |-- LangGraph graph:  chat_node <-> tools
     |-- Tools
     |     |- duckduckgo_search       (ordinary LangChain tool)
     |     |- get_stock_price         (own @tool, Alpha Vantage API)
     |     |- MCP "arith"   -> LOCAL server, started as a subprocess (stdio)
     |     |- MCP "expense" -> REMOTE server on Prefect Horizon (HTTP + OAuth login)
     |-- Memory: chatbot.db (SQLite, one row set per conversation thread)
```

The important idea: the LLM never knows or cares where a tool lives. `MultiServerMCPClient` turns every MCP tool into a normal LangChain tool, and the graph treats them all the same.

---

## 2. MCP concepts used

**MCP (Model Context Protocol)** is a standard way for an app (the *client*) to discover and call tools that live in another program (the *server*). The client asks "what tools do you have?", gets names, descriptions and argument schemas, and later says "run this tool with these arguments".

Here the chatbot is the client. It talks to two servers, and they differ only in **transport** (how the messages travel).

### 2.1 Local server: `arith` (stdio transport)

```python
"arith": {
    "transport": "stdio",
    "command": LOCAL_MCP_SERVER_PYTHON,   # the math server's own venv python.exe
    "args": [LOCAL_MCP_SERVER_PATH],      # its main.py
}
```

- With **stdio**, the client *launches the server as a child process* and talks to it through its standard input/output. No network, no login.
- The server is the FastMCP "arith" math server (add, subtract, multiply, divide, power, modulo, sqrt, factorial, average: 9 tools).
- `command` points at that project's **own** `.venv\Scripts\python.exe`. A bare `python` would use whatever Python this project has, which may not have the server's packages.
- The path contains a space (`GEO COMPUTER S`). Writing it as a raw string `r"..."` avoids backslash problems, and passing `command` and `args` as separate items avoids quoting problems.
- The server only lives while the chatbot uses it. It is not something you start separately.

### 2.2 Remote server: `expense` (streamable HTTP transport)

```python
"expense": {
    "transport": "streamable_http",
    "url": EXPENSE_URL,          # https://.../mcp on Prefect Horizon
    "auth": expense_auth,        # OAuth login object
}
```

- With **streamable_http**, the server is already running somewhere else and the client connects to its URL. This is what "remote MCP server" means.
- It is the SQLite expense tracker deployed on Prefect Horizon (add/edit/delete expenses, list categories, summarize, and so on). Horizon redeploys it when the GitHub repo changes.
- Because it is on the public internet, Horizon **requires a login**. That leads to the next section.

### 2.3 Local vs remote at a glance

| | `arith` (local) | `expense` (remote) |
|---|---|---|
| Transport | `stdio` | `streamable_http` |
| Who starts it | The chatbot, as a subprocess | Horizon, always on |
| Where it runs | This PC | Prefect's cloud |
| Login | None | OAuth 2.1 through Horizon |
| Config keys | `command`, `args` | `url`, `auth` |
| Data | None (pure maths) | SQLite database on the server |

---

## 3. The authentication problem and its fix

**Symptom:** `httpx.HTTPStatusError: Client error '401 Unauthorized'` for the expense URL, right after starting the app.

**Cause:** the original config connected with no credentials at all. The server said "who are you?" (401) and the whole tool loading crashed.

**A wrong idea worth remembering:** "put a bearer token in `.env`". There is no permanent token to copy from the dashboard. The server uses **OAuth**, where the client gets a short-lived access token (about 1 hour) by sending the user through a browser login once.

**How OAuth works here (authorization code flow with PKCE):**

1. The client tries the server and gets 401.
2. It registers itself with the server (dynamic client registration) and receives a `client_id`.
3. It opens the login page in the browser. You sign in to Horizon and approve.
4. The browser is redirected to `http://localhost:53683/callback?code=...`. The client runs a tiny web server on that port just to catch this.
5. The client swaps the `code` for an access token (and a refresh token), proving it started the flow with a PKCE code verifier.
6. The token is sent with every later request, and saved so the next start does not need a browser.

**What the code does about it:**

| Piece | Why |
|---|---|
| `OAuth(mcp_url=..., ...)` from `fastmcp.client.auth` | Handles the whole flow above and plugs into the connection as `"auth"` |
| `additional_client_metadata={"token_endpoint_auth_method": "none"}` | Registers as a **public client** (PKCE, no secret). Without it the server issues a secret it cannot read back, and login ends in `401 Missing client_secret` |
| `FileTreeStore(...)` as `token_storage` | Saves the token on disk. The default is in memory, which would force a login on every restart |
| The two `...SanitizationStrategy` arguments | The storage key contains `https://...`, which is not a valid Windows file name. These convert it to a safe one |
| `EXPENSE_TOKEN_DIR` = `%USERPROFILE%\.mcp-oauth-cache\expense-tracker` | Tokens live **outside the git repo**, so they cannot be committed by accident. Override with `EXPENSE_TOKEN_DIR` in `.env` |
| `callback_port=53683` | A fixed port keeps the redirect address the same every run, so the saved registration stays valid. It differs from `client1.py` (53682) so the two never collide |
| `callback_timeout=EXPENSE_LOGIN_TIMEOUT` | How long to wait for you to approve. Default 300 s, set with `EXPENSE_LOGIN_TIMEOUT` |
| `_open_login_page` replacing `webbrowser.open` | The Windows default browser was Edge, but the Horizon session and the right Google account were in Chrome. This opens Chrome instead. `LOGIN_BROWSER` in `.env` picks another `.exe`, or `default` for the system browser |

**Practical rules learned:**

- Log in **once from a terminal** (`python langgraph_mcp_backend.py`), then start Streamlit.
- Only one login at a time can use port 53683. Old, still-waiting runs block new ones, so close them first.
- The access token lasts about an hour. If the expense tools disappear later, repeat the terminal login.
- If the wrong Google account appears, sign in to the right one in Chrome first, or paste the printed login link into a Chrome window that has the right account.

Related notes: `REMOTE_SERVER_AUTH_PROBLEMS.md` in the expense-tracker repo covers every login problem in more detail.

---

## 4. Loading tools: `load_mcp_tools()`

```python
for server_name in MCP_SERVERS:
    try:
        server_tools = run_async(client.get_tools(server_name=server_name))
    except BaseException as exc:
        ...unwrap, print [WARNING], remember the name...
        continue
    tools_loaded.extend(server_tools)

if not tools_loaded:
    raise RuntimeError(...)
```

Concepts:

- **Per-server loading.** Asking `get_tools()` for everything at once means one failing server takes all tools down. Loading each by `server_name` lets `arith` work even when the expense login fails.
- **Never fail silently.** Every failure prints the real reason. If *nothing* loads, it raises, so an empty tool list can never slip through unnoticed. The original code had a comment about this on purpose ("no try/except, let the real error surface").
- **Exception groups.** The MCP library runs work in task groups, so errors arrive wrapped in a `BaseExceptionGroup`. The loop unwraps it (`exc.exceptions[0]`) to show the real error, such as `TimeoutError`.
- `KeyboardInterrupt` and `SystemExit` are re-raised so Ctrl+C still stops the program.
- The result is a list of LangChain `BaseTool` objects, one per MCP tool.

---

## 5. LangGraph concepts used

LangGraph models an agent as a **graph** of steps that share a **state**.

### 5.1 State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The state is just the message list. `add_messages` is a *reducer*: when a node returns new messages, they are **appended** to the list instead of replacing it.

### 5.2 Nodes and edges

```
START -> chat_node --(tools_condition)--> tools -> chat_node -> ... -> END
                  \--(no tool calls)-----> END
```

- `chat_node` (async): sends all messages to the LLM and returns its reply. The LLM was set up with `llm.bind_tools(tools)` so it knows which tools exist and can answer with a tool call instead of text.
- `tools` is a prebuilt `ToolNode(tools)`. It runs whatever tool the LLM asked for and adds a `ToolMessage` with the result.
- `tools_condition` is a prebuilt router: if the last message has tool calls, go to `tools`; otherwise finish.
- `graph.add_edge("tools", "chat_node")` closes the loop, so the LLM sees the tool result and can answer or call another tool. This loop is why a request such as "add an expense" works even when the model first calls `list_categories`.

### 5.3 Checkpointer (memory)

```python
conn = await aiosqlite.connect(database="chatbot.db")
checkpointer = AsyncSqliteSaver(conn)
chatbot = graph.compile(checkpointer=checkpointer)
```

- The checkpointer saves the state after every step into `chatbot.db`.
- Each conversation is identified by a **`thread_id`** passed in `config={"configurable": {"thread_id": ...}}`. Same id means the conversation continues; new id means a fresh chat.
- `retrieve_all_threads()` walks `checkpointer.alist(None)` to collect every saved `thread_id`, which fills the sidebar after a restart.
- `chatbot.get_state(config)` returns the saved messages for one thread, used to reopen an old chat.

---

## 6. Async code concepts used (the tricky part)

MCP tools, the LLM call and the database are **async** (`async def`, `await`). Streamlit is **synchronous** and re-runs the whole script on each interaction. Mixing the two needs care.

### 6.1 A dedicated event loop on a background thread (backend)

```python
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):            # run and WAIT for the result
    return _submit_async(coro).result()

def submit_async_task(coro):    # run, do NOT wait
    return _submit_async(coro)
```

Why:

- Streamlit reruns the script again and again, and `asyncio.run()` creates and destroys a loop every time. A long-lived object such as the SQLite connection or an MCP session cannot survive that.
- One permanent loop on its own thread (`daemon=True` so it dies with the app) keeps everything async in one place.
- `run_coroutine_threadsafe` is the safe way to hand a coroutine from another thread to that loop. It returns a `concurrent.futures.Future`.
- `.result()` blocks the calling thread until done. That is how synchronous module-level code (`load_mcp_tools()`, `_init_checkpointer()`) uses async functions.

### 6.2 The queue bridge (frontend)

`chatbot.astream(...)` is async, but Streamlit's `st.write_stream()` wants a normal (sync) generator. The frontend bridges them:

```
backend loop thread                       Streamlit thread
run_stream():                             ai_only_stream():
  async for chunk in chatbot.astream:       while True:
      event_queue.put(chunk)   ---queue--->     item = event_queue.get()
  event_queue.put(None)        (sentinel)       if item is None: break
                                                yield chunk text
```

- `queue.Queue` is thread-safe, so it is a safe pipe between the two threads.
- `None` is a **sentinel**, meaning "no more items". It is put in a `finally:` so the consumer never hangs, even after an error.
- Errors are sent as `("error", exc)` and re-raised on the Streamlit side, so they show up in the UI instead of being lost in the background thread.
- `stream_mode="messages"` yields message chunks token by token, which gives the typing effect.

---

## 7. Streamlit concepts used

- **Reruns.** The whole script re-runs on every click or message, so anything that must survive lives in `st.session_state` (`message_history`, `thread_id`, `chat_threads`).
- **Initialisation guards.** `if "thread_id" not in st.session_state:` sets a value only the first time.
- **Sidebar** (`st.sidebar`): a New Chat button and one button per saved conversation. Clicking one loads its messages from the checkpointer and rebuilds `message_history`.
- **Chat widgets.** `st.chat_input` for typing, `st.chat_message("user" / "assistant")` for bubbles.
- **Streaming.** `st.write_stream(generator)` shows text as it arrives and returns the full text at the end, which is then saved to `message_history`.
- **Tool status box.** When a `ToolMessage` arrives, an `st.status("Using `tool_name` ...")` box appears. It is turned into "Tool finished" once the stream ends. `status_holder = {"box": None}` is a small dict so the inner generator can change a value defined outside it.
- **Filtering the stream.** Only `AIMessage` chunks are yielded as text. `ToolMessage`s are used only for the status box.
- **`thread_id` as `uuid.uuid4()`.** A random unique id per chat, so conversations never collide.

---

## 8. Other decisions worth remembering

- **Model:** `ministral-8b-latest`, hardcoded on purpose. On this Mistral plan `mistral-small`, `mistral-medium` and `magistral-*` returned `429` with an allowance of 0 requests per minute. A shared environment variable could have changed the model silently, so it is not read from `.env`. See `MISTRAL_MODEL_LIMITS.md`.
- **Secrets:** the Mistral key and other keys live in `.env` (loaded by `load_dotenv()`), never in code. OAuth tokens live outside the repo.
- **Debug prints:** the `[DEBUG]` and `[WARNING]` lines show exactly which servers and tools loaded. Read them first when something is missing.
- **Older files in this folder** (`langgraph_backend.py`, `..._database_backend.py`, `..._tool_backend.py` and their frontends) are earlier steps of the same course project: basic chat, then SQLite memory, then streaming, threading and tools. The MCP pair is the latest step.

---

## 9. Troubleshooting cheatsheet

| What you see | Meaning | Fix |
|---|---|---|
| `401 Unauthorized` on the expense URL | No valid login | Run the terminal login (section 3) |
| `[WARNING] Could NOT load MCP server 'expense': TimeoutError ...` | Login not approved in time | Run the login again and approve within 5 minutes |
| Expense tools missing after a while | Access token expired (about 1 hour) and refresh failed | Repeat the terminal login |
| Login page opens for the wrong Google account | Chrome's active account differs | Sign in to the right account first, or paste the printed link into the right window |
| Login never starts, port error | Port 53683 held by an old run | Stop old Streamlit or Python runs, then retry |
| `429` from Mistral | Model has no allowance on your plan | Use `ministral-8b-latest` |
| `No MCP tools could be loaded from any server` | Every server failed | Read the `[WARNING]` lines above it |
| Tool loading fails on `arith` | Wrong path or venv | Check the `[DEBUG] Does that path exist?` lines |

---

## 10. One-line summary of each concept

- **stdio MCP** = launch the server yourself as a subprocess. **Streamable HTTP MCP** = connect to a server that is already running at a URL.
- **OAuth + PKCE** = a browser login once, then a saved short-lived token, instead of a permanent password.
- **`MultiServerMCPClient`** = one client, many servers, all tools returned in the same LangChain format.
- **LangGraph** = state + nodes + edges; the `chat_node` / `tools` loop is what makes an agent.
- **Checkpointer + `thread_id`** = saved memory per conversation.
- **Background event loop + queue** = the bridge between async backend code and sync Streamlit.
