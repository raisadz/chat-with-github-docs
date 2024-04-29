import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
import shutil
from git import Repo
import streamlit as st


def github_embeddings(
    git_url: str,
    dir_name: str,
    folders: str,
    pinecone_api_key: str,
    index_name: str,
    flag_new: bool = False,
):
    dir_copy = os.path.join(".", dir_name)

    dirpath = Path(dir_copy)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    Repo.clone_from(git_url, dir_copy)

    folders = folders.split(",")
    docs = []
    all_files = []
    for folder in folders:
        for dirpath, _, filenames in os.walk(os.path.join(dir_copy, folder)):
            for file in filenames:
                if not (
                    file.endswith(".py")
                    or file.endswith(".md")
                    or file.endswith(".qmd")
                ):
                    continue
                full_file_name = os.path.join(dirpath, file)
                all_files += [full_file_name]
                try:
                    loader = TextLoader(full_file_name, encoding="utf-8")
                    docs.extend(loader.load_and_split())
                except Exception:
                    print(f"failed for {full_file_name}")
                    pass

    TEXT_SPLITTER_CHUNK_PARAMS = {
        "chunk_size": 1000,
        "chunk_overlap": 500,
        "length_function": len,
    }

    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CHUNK_PARAMS)

    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key=pinecone_api_key)

    pc_indexes = [x["name"] for x in pc.list_indexes()]

    if flag_new:
        if index_name in pc_indexes:
            pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
    else:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        vectorstore.add_documents(documents)


if __name__ == "__main__":
    git_url = "https://github.com/kevinheavey/modern-polars.git"
    dir_name = "modern-polars"
    folders = "book"
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    index_name = "polars-docs"
    flag_new = False
    github_embeddings(
        git_url, dir_name, folders, pinecone_api_key, index_name, flag_new
    )
