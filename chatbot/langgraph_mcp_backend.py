from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from fastmcp.client.auth import OAuth
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1CollectionSanitizationStrategy,
    FileTreeV1KeySanitizationStrategy,
)
from pathlib import Path
import aiosqlite
import requests
import asyncio
import subprocess
import threading
import webbrowser
import os

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
# DELIBERATELY hardcoded here, NOT read from a shared .env variable -- a
# generic env var like MISTRAL_MODEL would silently affect every file that
# happens to read it. See MISTRAL_MODEL_LIMITS.md: mistral-small-latest
# returns a hard 429 (zero allowance) on many accounts -- ministral-8b-latest
# is the confirmed-working default from that investigation.
MCP_BACKEND_MODEL = "ministral-8b-latest"
print(f"[DEBUG] Using Mistral model: {MCP_BACKEND_MODEL}")
llm = ChatMistralAI(model=MCP_BACKEND_MODEL)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.environ["ALPHA_VANTAGE_API_KEY"]
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


# Real path to your local math MCP server, found via a filesystem search --
# NOTE the space in "GEO COMPUTER S", the raw string (r"...") handles the
# backslashes correctly.
LOCAL_MCP_SERVER_PATH = r"C:\Users\GEO COMPUTER S\Desktop\mcp-math-server\main.py"

# That server has its OWN .venv with its own dependencies (fastmcp/mcp, etc.).
# Launching it with a bare "python" would use whichever python is on THIS
# process's PATH -- not necessarily one with those packages installed. Point
# "command" at that server's own venv interpreter directly so the subprocess
# actually has what it needs, regardless of what's installed in labgGraph's venv.
LOCAL_MCP_SERVER_PYTHON = r"C:\Users\GEO COMPUTER S\Desktop\mcp-math-server\.venv\Scripts\python.exe"

print(f"[DEBUG] Local MCP server script path: {LOCAL_MCP_SERVER_PATH}")
print(f"[DEBUG] Does that path exist? {os.path.exists(LOCAL_MCP_SERVER_PATH)}")
print(f"[DEBUG] Local MCP server venv python: {LOCAL_MCP_SERVER_PYTHON}")
print(f"[DEBUG] Does that venv python exist? {os.path.exists(LOCAL_MCP_SERVER_PYTHON)}")

# ---------------------------------------------------------------------------
# Remote "expense" server: it needs a LOGIN, not an API key.
# The 401 you get without one is the server asking for an OAuth login through
# Horizon. There is no static token to find in the dashboard or to put in .env
# (an access token only lasts 1 hour). Instead a browser window opens once, you
# approve, and the login is saved to disk and reused. Full story, including every
# problem hit on the way: REMOTE_SERVER_AUTH_PROBLEMS.md
# ---------------------------------------------------------------------------
EXPENSE_URL = "https://expense-tracker-mseepee.fastmcp.app/mcp"

# Saved login. Kept in your user folder, OUTSIDE this git repo, so tokens can never
# be committed by accident. Override with EXPENSE_TOKEN_DIR in .env if you like.
EXPENSE_TOKEN_DIR = Path(
    os.getenv("EXPENSE_TOKEN_DIR") or (Path.home() / ".mcp-oauth-cache" / "expense-tracker")
)
EXPENSE_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

# Fixed port for the login callback. Deliberately different from mcp-client/client1.py
# (53682) so the two can never block each other. Only ONE login can run at a time.
EXPENSE_CALLBACK_PORT = 53683

# How long to wait for you to approve the login before giving up on this server.
EXPENSE_LOGIN_TIMEOUT = float(os.getenv("EXPENSE_LOGIN_TIMEOUT", "300"))

# The login page has to open in the browser where you are signed in to Horizon.
# The Windows default here is Edge, but the Horizon session lives in Chrome.
# Set LOGIN_BROWSER in .env to another browser .exe, or to "default" for the system default.
_CHROME_PATHS = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]
_default_browser_open = webbrowser.open


def _open_login_page(url, *args, **kwargs):
    print("\n=== LOGIN NEEDED (expense server) ===")
    print("Approve the login in the browser window that opens. If nothing opens, paste this into Chrome:")
    print(url + "\n")
    browser = os.getenv("LOGIN_BROWSER") or next((p for p in _CHROME_PATHS if os.path.exists(p)), None)
    if browser and browser.lower() != "default":
        subprocess.Popen([browser, url])
        return True
    return _default_browser_open(url, *args, **kwargs)


webbrowser.open = _open_login_page

expense_auth = OAuth(
    mcp_url=EXPENSE_URL,
    client_name="LangGraph MCP Chatbot",
    # Register as a PUBLIC client (PKCE, no secret). At the default the server issues a
    # secret it then cannot read back, and the login ends in "401 Missing client_secret".
    additional_client_metadata={"token_endpoint_auth_method": "none"},
    token_storage=FileTreeStore(
        data_directory=EXPENSE_TOKEN_DIR,
        # Without these, the "https://..." in the storage key is an invalid Windows file name.
        key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(EXPENSE_TOKEN_DIR),
        collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(EXPENSE_TOKEN_DIR),
    ),
    callback_port=EXPENSE_CALLBACK_PORT,
    callback_timeout=EXPENSE_LOGIN_TIMEOUT,
)

MCP_SERVERS = {
    "arith": {
        "transport": "stdio",
        "command": LOCAL_MCP_SERVER_PYTHON,
        "args": [LOCAL_MCP_SERVER_PATH],
    },
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": EXPENSE_URL,
        "auth": expense_auth,
    },
}

client = MultiServerMCPClient(MCP_SERVERS)


def load_mcp_tools() -> list[BaseTool]:
    # Each server is loaded on its own. A failure is NEVER silent: it is printed loudly with
    # the real cause. But one server failing (for example, a login that was not approved in
    # time) no longer takes the others down. If NOTHING loads at all we raise, so an empty
    # tool list can never slip through unnoticed.
    print("[DEBUG] Attempting to load MCP tools, one server at a time...")
    print(f"[DEBUG] If the expense server needs a login, approve it in the browser "
          f"(waiting up to {EXPENSE_LOGIN_TIMEOUT:.0f}s)...")
    tools_loaded: list[BaseTool] = []
    failed: list[str] = []
    for server_name in MCP_SERVERS:
        try:
            server_tools = run_async(client.get_tools(server_name=server_name))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]  # unwrap to the real cause
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            if isinstance(exc, TimeoutError):
                reason += " (the login was not approved in time)"
            print(f"[WARNING] Could NOT load MCP server '{server_name}': {reason}")
            failed.append(server_name)
            continue
        print(f"[DEBUG] Loaded {len(server_tools)} tool(s) from '{server_name}'")
        tools_loaded.extend(server_tools)

    if not tools_loaded:
        raise RuntimeError(f"No MCP tools could be loaded from any server (failed: {failed})")
    if failed:
        print(f"[WARNING] Continuing WITHOUT: {failed}. The chatbot will not have those tools.")

    print(f"[DEBUG] Successfully loaded {len(tools_loaded)} MCP tool(s):")
    for t in tools_loaded:
        print(f"[DEBUG]   - {t.name}")
    return tools_loaded


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
print(f"[DEBUG] Total tools bound to LLM: {len(tools)} -> {[t.name for t in tools]}")
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())