from langchain_openai import OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    StreamlitChatMessageHistory,
)
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import langchain

langchain.debug = True

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

load_dotenv()


def create_chain():
    embeddings = OpenAIEmbeddings()

    db = Chroma(persist_directory="./narwhalsdb", embedding_function=embeddings)
    retriever = db.as_retriever()

    memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(),
        return_messages=True,
        memory_key="chat_history",
    )

    condense_prompt = PromptTemplate.from_template(
        "Combine the chat history and follow up question into "
        "a standalone question. Chat History: {chat_history} "
        "Follow up question: {question}."
    )

    combine_docs_custom_prompt = PromptTemplate.from_template(
        (
            "Answer the question {question} "
            "taking into account {context} and {chat_history}. "
            "Do not make any assumptions. If you don't know the answer, just say that you don't know."
        )
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt),
    )
    return chain

