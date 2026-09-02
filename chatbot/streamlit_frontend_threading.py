import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# **************************************** utility functions *************************

def generate_thread_id():
    # Every conversation gets its own unique ID. uuid4() generates a random,
    # effectively-unguessable ID -- this becomes the "thread_id" that LangGraph's
    # checkpointer uses to keep this conversation's history completely separate
    # from every other conversation.
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    # Called when the user wants to START A NEW CONVERSATION (the "New Chat" button).
    # Generates a fresh thread_id, registers it in the sidebar's thread list,
    # and clears the currently DISPLAYED message history -- note this does NOT
    # delete any previous conversation's data; the old thread_id's history still
    # lives inside the checkpointer, untouched, ready to be reloaded later.
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    # Registers a thread_id into the sidebar's list of known conversations,
    # but only if it isn't already there -- prevents duplicate entries from
    # piling up in the sidebar every time the app reruns (Streamlit reruns
    # this whole script top-to-bottom on every interaction).
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    # THIS is the "resume" feature's core mechanism. chatbot.get_state(...)
    # asks the CHECKPOINTER (not Streamlit) for the full saved state of a
    # given thread_id -- the same get_state() you used in the persistence lab.
    # Because the checkpointer already stores full message history per thread
    # (via the add_messages reducer), this single call retrieves an ENTIRE
    # past conversation, even one from a previous app run/session.
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    # (a brand-new thread that's never been invoked yet has no 'messages' key)
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************

# st.session_state persists across Streamlit reruns WITHIN one browser tab/session,
# but resets if the page is refreshed or a new session starts -- this is DIFFERENT
# from the checkpointer's persistence, which survives even across app restarts.
# These three initializations only run ONCE per session (the 'not in' guard).

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []          # what's currently SHOWN on screen

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()   # the ACTIVE conversation right now

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []              # list of ALL known conversation IDs (sidebar)

add_thread(st.session_state['thread_id'])   # make sure the very first thread is in the sidebar list


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    # Clicking this button triggers reset_chat(), which swaps in a brand-new
    # thread_id and clears the visible chat -- the user is now talking to a
    # FRESH, empty conversation, while every prior conversation remains intact
    # and resumable via the checkpointer.
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # [::-1] reverses the list so the MOST RECENT conversation appears at the
    # TOP of the sidebar instead of the bottom -- a small UX touch.
    if st.sidebar.button(str(thread_id)):
        # Clicking a past conversation's button in the sidebar:
        # 1. Switch the ACTIVE thread_id to the one that was clicked.
        st.session_state['thread_id'] = thread_id

        # 2. Pull that conversation's FULL history back out of the checkpointer
        #    (this is the actual "resume" moment -- old messages come back).
        messages = load_conversation(thread_id)

        # 3. Convert LangChain's message OBJECTS (HumanMessage/AIMessage) into
        #    the plain {'role': ..., 'content': ...} dict format Streamlit's
        #    chat display loop expects -- two different message "shapes" for
        #    two different systems (LangGraph's internal messages vs. the UI).
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        # 4. Replace what's currently displayed with the resumed conversation.
        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

# CONFIG is built HERE, inside the if-block, using whatever thread_id is
# CURRENTLY active -- this is what makes every new message get appended
# to the RIGHT conversation, whether that's a brand-new thread or one
# that was just resumed from the sidebar.
#
# UPGRADED for LangSmith observability: 'configurable.thread_id' is what
# LangGraph reads to route this call to the correct checkpointed thread --
# that part already worked. 'metadata' and 'run_name' are NEW additions
# that LangSmith reads to make each trace identifiable in the dashboard:
#   - run_name  -> labels this trace "chat_turn" instead of a generic
#                  default name, so a list of traces is actually skimmable.
#   - metadata  -> attaches the thread_id as a SEARCHABLE tag on the trace,
#                  so the dashboard can be filtered down to "show me every
#                  trace from THIS specific conversation" -- something that
#                  wasn't possible with configurable.thread_id alone, since
#                  that field only controls routing, LangSmith doesn't
#                  surface it as a filterable trace attribute on its own.
    CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}
    # first add the message to message_history
    with st.chat_message("assistant"):

        def ai_only_stream():
            # This wraps chatbot.stream(...) (same streaming mechanism as the
            # earlier lab) but adds a FILTER: this backend graph likely has
            # TOOL-CALLING nodes too (langgraph_tool_backend), which means the
            # stream can include internal tool-call/tool-result messages, not
            # just the final AI reply. isinstance(message_chunk, AIMessage)
            # ensures ONLY the assistant's actual reply text gets shown to the
            # user -- tool-call machinery stays hidden from the chat UI.
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})


# ─────────────────────────────────────────────────────────────────────────────
# DEEP DIVE — every function and button, what it does, and why it exists
# ─────────────────────────────────────────────────────────────────────────────
#
# THE BIG PICTURE FIRST
#   This app layers a MULTI-CONVERSATION, RESUMABLE chat UI on top of the
#   single-thread chatbot from the previous lab. The key insight: LangGraph's
#   checkpointer ALREADY stores full history per thread_id (that's the whole
#   persistence lab). This file's entire job is to let the USER see and switch
#   between multiple thread_ids through a sidebar, instead of being locked into
#   one hardcoded thread_id like before.
#
# ── generate_thread_id() ──
#   Produces a random UUID. Called in two places: once at session start (for
#   the very first conversation) and once every time "New Chat" is clicked.
#   Nothing about this function talks to the checkpointer -- it's PURE ID
#   GENERATION. The checkpointer only becomes aware of a thread_id the moment
#   chatbot.invoke()/.stream() is actually called with it.
#
# ── add_thread(thread_id) ──
#   Maintains the SIDEBAR'S list of conversations (st.session_state['chat_threads']).
#   This list is Streamlit-side bookkeeping ONLY -- it's how the UI knows what
#   buttons to draw in the sidebar. It has nothing to do with the checkpointer's
#   own storage; if this list were lost (e.g. browser refresh), the underlying
#   conversations would STILL exist in the checkpointer, just not be visibly
#   listed in the sidebar anymore (a real app would persist this list too, e.g.
#   in a database, to survive refreshes).
#
# ── load_conversation(thread_id) ──
#   THE mechanism that makes "resume" possible. chatbot.get_state(config=...)
#   is a direct call into LangGraph's checkpointer (same method from the
#   persistence lab), fetching the LATEST saved state for that thread --
#   which, thanks to the add_messages reducer, contains the ENTIRE accumulated
#   message history, not just the last exchange. state.values.get('messages', [])
#   defensively handles a thread that's been registered in the sidebar but never
#   actually had a message sent yet (no checkpoint exists, so no 'messages' key).
#
# ── reset_chat() ──
#   The handler behind the "New Chat" BUTTON. Three actions in sequence:
#     1. Mint a brand-new thread_id (a conversation the checkpointer has never seen).
#     2. Register it in the sidebar via add_thread().
#     3. Clear message_history so the chat window shows blank/empty.
#   Critically, this does NOT touch any OTHER thread's data -- every previous
#   conversation remains fully intact in the checkpointer, just no longer the
#   ACTIVE one being displayed or written to.
#
# ── THE "New Chat" BUTTON ──
#   if st.sidebar.button('New Chat'): reset_chat()
#   Streamlit buttons return True for exactly ONE script rerun -- the rerun
#   that happens immediately after the click. So reset_chat() runs once per
#   click, swapping the active thread and clearing the display.
#
# ── THE SIDEBAR CONVERSATION BUTTONS (the for-loop) ──
#   One button is rendered per known thread_id, MOST RECENT FIRST ([::-1]).
#   Clicking one does the actual "resume":
#     1. st.session_state['thread_id'] = thread_id  -- makes this the ACTIVE thread.
#     2. load_conversation(thread_id)                -- pulls its full history
#        back from the checkpointer (this can be a conversation from MINUTES
#        or, with a persistent checkpointer backend, DAYS ago).
#     3. Convert each LangChain message object to a plain dict Streamlit can
#        render, and overwrite message_history with it.
#   After this, the main UI loop (just below) redraws the chat window using
#   the RESUMED history -- visually, it looks exactly like you never left
#   that conversation, even though you were just looking at a totally
#   different thread a moment ago.
#
# ── WHY CONFIG IS BUILT INSIDE THE if user_input: BLOCK ──
#   CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
#   This line intentionally reads thread_id FRESH, at the moment a message is
#   sent -- not once at the top of the script. Since Streamlit reruns this
#   whole file on every interaction, and thread_id can change (New Chat, or
#   clicking a past conversation) BEFORE the next message is typed, building
#   CONFIG here guarantees every new message is appended to whichever thread
#   is CURRENTLY selected, not some stale value from an earlier rerun.
#
# ── ai_only_stream() — why a FILTER is needed on top of streaming ──
#   This backend is langgraph_tool_backend -- implying it's a graph WITH TOOLS
#   (like the currency-converter / weather agent labs), not a plain single-node
#   chatbot. A tool-using graph's message stream can include more than just the
#   final reply: tool-call requests and tool results also flow through as
#   messages. Blindly streaming everything to st.write_stream() would leak
#   that internal machinery into the visible chat. The isinstance(message_chunk,
#   AIMessage) check keeps ONLY the assistant's actual conversational text,
#   silently dropping anything else -- the tool activity happens, but stays
#   invisible to the end user, exactly like a real product's chat UI would.
#
# ── TWO KINDS OF "MEMORY" WORKING TOGETHER ──
#   st.session_state  -- Streamlit's own memory, scoped to ONE browser session.
#                         Holds what's currently DISPLAYED and which thread is
#                         ACTIVE. Lost on a full page refresh.
#   LangGraph's checkpointer -- the DURABLE memory. Holds the actual message
#                         history for every thread_id ever used. Outlives a
#                         page refresh (with a persistent backend; InMemorySaver
#                         specifically only outlives the Python process).
#   This app's entire "resume" feature exists at the SEAM between these two:
#   Streamlit remembers WHICH conversations exist and which is active; the
#   checkpointer remembers WHAT was actually said in each one.
# ─────────────────────────────────────────────────────────────────────────────