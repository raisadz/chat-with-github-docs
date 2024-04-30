import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import (
    StreamlitChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap


def get_response(user_query):
    embeddings = OpenAIEmbeddings()
    db = PineconeVectorStore(
        index_name=st.secrets["PINECONE_INDEX_NAME"], embedding=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 50})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000,
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a useful AI assistant.\n"
                "Respond in short but complete answers unless specifically "
                "asked by the User to elaborate on something.\n"
                "Use both Context and History to inform your answers.\n"
                "We have provided Context and History below. \n"
                "---------------------\n"
                "Context: {context} \n"
                "---------------------\n"
                "History: {history} \n"
                "Give an answer only if you checked that it is correct in Context. \n"
                "If you can't find an answer in Context, say that you don't know, don't imagine anything. \n"
                "Given this information, provide an answer to the following \n:"
                "---------------------\n"
                "User question: {input}\n",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return StreamlitChatMessageHistory()

    output_parser = StrOutputParser()
    chain = (
        RunnableMap(
            {
                "context": lambda x: "\n\n -----".join(
                    [
                        doc.page_content
                        for doc in retriever.get_relevant_documents(x["input"])
                    ]
                ),
                "input": lambda x: x["input"],
                "history": lambda x: x["history"],
            }
        )
        | prompt
        | llm
        | output_parser
    )

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
