__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import (
    StreamlitChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter


load_dotenv()


def get_response(user_query):
    embeddings = OpenAIEmbeddings()

    db = Chroma(persist_directory="./narwhalsdb", embedding_function=embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a useful AI assistant. Here is some {context} "
                "Answer the following questions considering the history chat and using context: "
                """    
            Context: {context}

            History chat: {history}

            User question: {input}""",
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
