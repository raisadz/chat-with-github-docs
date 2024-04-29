import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import (
    StreamlitChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
from langchain_pinecone import PineconeVectorStore


def get_response(user_query):
    embeddings = OpenAIEmbeddings()
    db = PineconeVectorStore(
        index_name=st.secrets["PINECONE_INDEX_NAME"], embedding=embeddings
    )
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        temperature=0, max_tokens=1000, api_key=st.secrets["OPENAI_API_KEY"]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a useful AI assistant and you are knowledgeable about the context. "
                "You can provide the descriptions of the available functions "
                "and write code snippets using non-deprecated functions. "
                "Answer the following user questions in maximum six sentences "
                "considering the context and the history chat. "
                "Do not imagine anything and use only the context to get the correct answer. "
                "If you think that the context doesn't have the answer, say that you don't know. "
                "If the user asks you something not related to context "
                "politely remind the user that it is not related. "
                """
            Context: {context}

            History chat: {history}

            User question: {input}.
            """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return StreamlitChatMessageHistory()

    context = itemgetter("input") | retriever
    context_step = RunnablePassthrough.assign(context=context)
    chain = context_step | prompt | llm
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return with_message_history.stream(
        {"input": user_query},
        config={"configurable": {"session_id": "abc123"}},
    )
