from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # not langchain.text_splitter
from langchain_community.vectorstores import FAISS
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
import tempfile
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
# 1. LLM + embeddings
# -------------------
# DELIBERATELY hardcoded here, NOT read from a shared .env variable -- a
# generic env var like MISTRAL_MODEL would silently affect every file that
# happens to read it. See MISTRAL_MODEL_LIMITS.md: mistral-small-latest
# returns a hard 429 (zero allowance) on many accounts -- ministral-8b-latest
# is the confirmed-working default from that investigation.
MCP_BACKEND_MODEL = "ministral-8b-latest"
print(f"[DEBUG] Using Mistral model: {MCP_BACKEND_MODEL}")
llm = ChatMistralAI(model=MCP_BACKEND_MODEL)
embeddings = MistralAIEmbeddings(model="mistral-embed")

# -------------------
# 2. PDF retriever store (per thread) -- NEW for RAG
# -------------------
# Each conversation thread can have its OWN uploaded PDF, indexed separately.
# This dict lives in memory (not the checkpointer), so it resets if the
# backend process restarts -- re-upload the PDF for a thread after a restart
# if you need it again.
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Called from the Streamlit sidebar uploader -- NOT a LangGraph tool itself,
    just a plain helper function the frontend calls directly.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # FAISS keeps its own copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num, "second_num": second_num,
            "operation": operation, "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.environ["ALPHA_VANTAGE_API_KEY"]   # your own key, not hardcoded
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# Real path to your local math MCP server, found via a filesystem search --
# NOTE the space in "GEO COMPUTER S", the raw string (r"...") handles the
# backslashes correctly.
LOCAL_MCP_SERVER_PATH = r"C:\Users\GEO COMPUTER S\Desktop\mcp-math-server\main.py"
LOCAL_MCP_SERVER_PYTHON = r"C:\Users\GEO COMPUTER S\Desktop\mcp-math-server\.venv\Scripts\python.exe"

print(f"[DEBUG] Local MCP server script path: {LOCAL_MCP_SERVER_PATH}")
print(f"[DEBUG] Does that path exist? {os.path.exists(LOCAL_MCP_SERVER_PATH)}")
print(f"[DEBUG] Local MCP server venv python: {LOCAL_MCP_SERVER_PYTHON}")
print(f"[DEBUG] Does that venv python exist? {os.path.exists(LOCAL_MCP_SERVER_PYTHON)}")

# ---------------------------------------------------------------------------
# Remote "expense" server: needs a LOGIN, not an API key. See
# REMOTE_SERVER_AUTH_PROBLEMS.md and MCP_CHATBOT_401_FIX_CHANGES.md for the
# full story of why this OAuth setup looks the way it does.
# ---------------------------------------------------------------------------
EXPENSE_URL = "https://expense-tracker-mseepee.fastmcp.app/mcp"

EXPENSE_TOKEN_DIR = Path(
    os.getenv("EXPENSE_TOKEN_DIR") or (Path.home() / ".mcp-oauth-cache" / "expense-tracker")
)
EXPENSE_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

EXPENSE_CALLBACK_PORT = 53683
EXPENSE_LOGIN_TIMEOUT = float(os.getenv("EXPENSE_LOGIN_TIMEOUT", "300"))

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
    additional_client_metadata={"token_endpoint_auth_method": "none"},
    token_storage=FileTreeStore(
        data_directory=EXPENSE_TOKEN_DIR,
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
        "transport": "streamable_http",
        "url": EXPENSE_URL,
        "auth": expense_auth,
    },
}

client = MultiServerMCPClient(MCP_SERVERS)


def load_mcp_tools() -> list[BaseTool]:
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
                exc = exc.exceptions[0]
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

# rag_tool added alongside the original three + MCP tools
tools = [search_tool, get_stock_price, calculator, rag_tool, *mcp_tools]
print(f"[DEBUG] Total tools bound to LLM: {len(tools)} -> {[t.name for t in tools]}")
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    # thread_id is pulled from config so the SYSTEM MESSAGE can tell the model
    # exactly which thread_id to pass into rag_tool -- the model has no other
    # way to know which conversation it's in.
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about an uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use web search, stock price, calculator, "
            "and any connected MCP tools when helpful. If no document is available "
            "for this chat, ask the user to upload a PDF first."
        )
    )

    messages = [system_message, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 6. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 7. Graph
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
# 8. Helpers
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})