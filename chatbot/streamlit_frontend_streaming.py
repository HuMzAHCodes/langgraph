import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message('assistant'):
        # STREAMING RESPONSE -- see full explanation at the bottom of this file
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable': {'thread_id': 'thread-1'}},
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Streaming — chatbot.stream(...) + st.write_stream(...)
# ─────────────────────────────────────────────────────────────────────────────
#
# THE PROBLEM THIS SOLVES
#   Your earlier version used chatbot.invoke(...) -- it BLOCKS until the model
#   has generated the ENTIRE reply, then hands it back as one finished string.
#   For a long answer, the user stares at a blank chat bubble for the whole
#   generation time before anything appears. Every real chat UI (ChatGPT, etc.)
#   instead shows text appearing token-by-token as it's generated. That's
#   STREAMING, and it's what this block adds.
#
# chatbot.stream(...) -- streaming the GRAPH, not just the model
#   .stream() is the streaming counterpart to .invoke(). Instead of returning
#   once at the end, it returns a GENERATOR you iterate over, yielding pieces
#   of output as they become available -- as soon as the LLM inside chat_node
#   produces each new token, it's yielded immediately, before the node (or the
#   graph) has finished.
#
# stream_mode='messages' -- WHAT gets yielded on each iteration
#   LangGraph supports several stream_mode options (e.g. 'values', 'updates').
#   'messages' specifically yields a stream of (message_chunk, metadata) TUPLES
#   as the LLM generates -- each message_chunk is a small piece (often a few
#   tokens) of the AI's in-progress reply, and metadata carries info like which
#   node/step produced it. This is why the loop unpacks TWO values:
#       for message_chunk, metadata in chatbot.stream(...)
#
# THE GENERATOR EXPRESSION
#   message_chunk.content for message_chunk, metadata in chatbot.stream(...)
#   This pulls just the .content (the actual text piece) out of each chunk,
#   discarding metadata since the UI only needs the text itself. This whole
#   expression is a Python GENERATOR -- nothing runs yet, it's lazy; text only
#   gets pulled out chunk-by-chunk as something actually iterates over it.
#
# st.write_stream(...) -- Streamlit's job in this partnership
#   st.write_stream() is built specifically to consume a generator/iterator of
#   text pieces and render them into the chat bubble PROGRESSIVELY, appending
#   each new piece to the display as it arrives -- producing the classic
#   "typewriter" effect. It ALSO collects every piece internally and returns
#   the FULL CONCATENATED STRING once the generator is exhausted -- which is
#   why `ai_message = st.write_stream(...)` still gives you the complete final
#   reply afterward, ready to save into message_history exactly like the old
#   invoke() version did.
#
# WHY thread_id STILL MATTERS HERE
#   Nothing about streaming changes the checkpointing/threading concept from
#   before -- config={'configurable': {'thread_id': 'thread-1'}} is still what
#   ties this exchange into the SAME persisted conversation history. Streaming
#   only changes HOW the response is delivered (progressively vs. all-at-once);
#   it does NOT change what gets remembered or how state accumulates via
#   add_messages. The two concepts (persistence and streaming) are independent
#   and layer cleanly on top of each other.
#
# THE FULL DATA FLOW, END TO END
#   chatbot.stream(...)                    -> yields (chunk, metadata) tuples
#     -> generator expression               -> extracts chunk.content only
#       -> st.write_stream(...)             -> renders progressively AND
#                                               returns the joined full string
#         -> ai_message                     -> saved into session_state history
# ─────────────────────────────────────────────────────────────────────────────