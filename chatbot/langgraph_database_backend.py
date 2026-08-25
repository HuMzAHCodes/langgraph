from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatMistralAI(model="mistral-small-latest")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_title: str   # NEW: a short, human-readable label for this conversation,
                         # auto-generated once and then checkpointed like any other
                         # state field -- persists across restarts via SqliteSaver.


def generate_title(first_message: str) -> str:
    # A SEPARATE, tiny LLM call -- deliberately simple/cheap, not routed
    # through the main chat_node's conversation context. We only need a
    # short label, not a full reasoning pass. Keeping this as its own
    # function (rather than folding it into chat_node's prompt) means it's
    # easy to test, swap models for, or upgrade independently later.
    prompt = f"""Summarize the following user message into a short chat title,
3 to 5 words maximum. No punctuation, no quotes, just the title itself.

Message: "{first_message}"

Title:"""
    title = llm.invoke(prompt).content.strip()
    return title


def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    # returning {'messages': [response]} does NOT overwrite the whole history --
    # the add_messages reducer APPENDS this response onto whatever was already
    # in state['messages']. Without the reducer, this return would wipe out
    # the user's original message entirely.
    update = {"messages": [response]}

    # Only generate a title ONCE per conversation -- the very first time this
    # thread has a message. len(messages) == 1 means this node is seeing a
    # state where ONLY the incoming HumanMessage exists (the AI's reply
    # hasn't been added to `messages` yet at this point -- `response` above
    # is still a local variable, not yet merged into state). On every
    # SUBSEQUENT call for this same thread, messages will already contain
    # prior turns, so len(messages) will be > 1 and this block is skipped --
    # the title, once set, stays fixed for the life of the conversation.
    if len(messages) == 1:
        update["thread_title"] = generate_title(messages[0].content)

    return update


# check_same_thread=False is REQUIRED: Streamlit (and SqliteSaver's internal
# access patterns) may use this connection from a different thread than the
# one that created it. SQLite forbids that by default for safety; this flag
# opts out so the checkpointer can actually function correctly here.
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# SqliteSaver: same checkpointer INTERFACE as InMemorySaver from the
# persistence lab (thread_id, get_state, get_state_history all behave
# identically) -- but every checkpoint, including thread_title now, is
# written to chatbot.db on disk. Conversations (and their titles) survive
# an app restart, a machine reboot, coming back tomorrow -- anything short
# of deleting the .db file itself.
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    # Asks SQLite for every checkpoint ever saved, across ALL thread_ids --
    # this is how the Streamlit sidebar knows which past conversations exist
    # at all when the app first starts, including ones from previous runs.
    # Only meaningful because storage is persistent; with InMemorySaver this
    # would always come back empty on a fresh launch.
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)   # set -> list: sidebar loop needs an indexable sequence


def get_thread_title(thread_id) -> str:
    # NEW: fetches ONE thread's saved title from its checkpointed state, for
    # display in the sidebar instead of the raw UUID. This is a SEPARATE
    # get_state() call per thread -- fine for a handful of conversations,
    # but worth knowing: with many threads, this means one DB read per
    # sidebar button on every rerun. A future optimization could batch this,
    # but for a learning-scale app it's simple and correct.
    #
    # Falls back to the raw thread_id string if no title exists yet -- this
    # covers a thread that was registered (e.g. via "New Chat") but never
    # actually had a message sent, so chat_node (and therefore
    # generate_title) never ran for it.
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('thread_title', str(thread_id))