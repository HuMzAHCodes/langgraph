# chatbot_with_hitl.py

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------
# DELIBERATELY hardcoded, not read from a shared env var -- see
# MISTRAL_MODEL_LIMITS.md: mistral-small-latest has ZERO request allowance on
# many accounts (hard 429, not a speed issue). ministral-8b-latest is the
# confirmed-working default.
HITL_MODEL = "ministral-8b-latest"
llm = ChatMistralAI(model=HITL_MODEL)

# -------------------
# 2. Tools
# -------------------
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
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # interrupt() PAUSES the graph right here, mid-tool-execution, and
    # returns control to whoever called .invoke(). The string passed in
    # becomes the "question" a human needs to answer before this can continue.
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

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

# -------------------
# 5. Checkpointer (in-memory)
# -------------------
# REQUIRED for interrupt() to work at all -- LangGraph needs somewhere to
# persist the PAUSED state while waiting on a human. MemorySaver is fine for
# a demo; a real app would use SqliteSaver/AsyncSqliteSaver like your other
# chatbot backends, so a pause survives a process restart too.
memory = MemorySaver()

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)

# -------------------
# 7. Simple usage example (CLI with HITL)
# -------------------
if __name__ == "__main__":

    # Fixed thread_id so the whole CLI session is one continuous, resumable
    # conversation in the checkpointer.
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph -- this MAY pause partway through if purchase_stock
        # calls interrupt().
        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # __interrupt__ is a special key LangGraph adds to the result ONLY
        # when the graph paused mid-execution -- its presence is how you
        # detect "this run stopped early, waiting on a human."
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            # The interrupt's .value is exactly whatever was passed to
            # interrupt(...) inside the tool -- here, that's the question string.
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Command(resume=...) is what actually CONTINUES the paused graph --
            # it re-enters purchase_stock's interrupt() call, but now decision
            # is whatever you pass here, instead of blocking again.
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")