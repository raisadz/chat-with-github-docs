import streamlit as st
from retrieval_chain import create_chain

# App title
st.set_page_config(page_title="🤗💬 Chat with Narwhals docs")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "🤗💬 Chat with Narwhals documentation"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


chain = create_chain()


# User-provided prompt

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": prompt})["answer"]
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
