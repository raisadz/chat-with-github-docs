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

    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a useful AI assistant that is knowledgeable about {context}. "
                "Answer the following user questions in maximum six sentences "
                "considering the context and the history chat: "
                """
            Context: {context}

            History chat: {history}

            User question: {input}.

            If the user asks you to something about the documentation use {context} to do it.

            If the user refers to the previous chat messages use both {context} 
            and {history} to answer he question.

            Do not imagine anything and try to find the answer using {context}. 
            If you think that {context} doesn't have the answer, say that you don't know.

            If the user asks you something not related to {context} 
            politely remind the user that it is no related.
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
