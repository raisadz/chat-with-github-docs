import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from retrieval_chain import get_response

st.set_page_config(page_title="Chat with GitHub docs", page_icon="🤖")
st.title(f"Chat with {st.secrets['DOCS_NAME']} docs")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am an AI assistant. How can I help you?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))
