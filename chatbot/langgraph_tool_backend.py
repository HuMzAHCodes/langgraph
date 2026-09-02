from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatMistralAI(model="mistral-small-latest")

# -------------------
# 2. Tools
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

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
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

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)   # gives the model AWARENESS of the tools -- execution is still separate

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
# ToolNode is a LangGraph PREBUILT node -- it replaces the manual tool-call
# loop you wrote by hand in the earlier tool-calling lab (reading
# result.tool_calls, running each tool, appending a ToolMessage). Given ANY
# list of tools, ToolNode automatically executes whichever tool(s) the LLM's
# last message requested and returns the results as ToolMessages, ready to
# feed back into the next chat_node call.

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# tools_condition is a LangGraph PREBUILT routing function -- it inspects the
# last message chat_node produced: if it contains tool_calls, route to
# "tools"; otherwise route to END. This is the same conditional-edge pattern
# from your review-reply lab (a routing function returning a node name), but
# pre-written for you since "does this AI message request a tool?" is such a
# common check that LangGraph ships it as a built-in.
graph.add_conditional_edges("chat_node", tools_condition)

# THE LOOP: after tools run, flow goes BACK to chat_node (not to END) so the
# LLM can read the tool's result and either respond, or request ANOTHER tool
# call. This is what lets a question needing multiple sequential tool calls
# (e.g. "convert currency, then multiply the result") work automatically,
# without you writing the loop by hand.
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)