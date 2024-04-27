__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

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
import streamlit as st
from git import Repo

from dotenv import load_dotenv

load_dotenv()

natwhals_git_url = "https://github.com/MarcoGorelli/narwhals.git"
narwhals_dir = "./narwhals_copy"

dirpath = Path(narwhals_dir)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

Repo.clone_from(natwhals_git_url, narwhals_dir)

# scrape all files from directory

docs = []
all_files = []
for dirpath, dirnames, filenames in os.walk(narwhals_dir):
    for file in filenames:
        full_file_name = dirpath + "/" + file
        if ("/." not in full_file_name) and ("venv" not in full_file_name):
            all_files += [full_file_name]
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception:
                pass


TEXT_SPLITTER_CHUNK_PARAMS = {
    "chunk_size": 1000,
    "chunk_overlap": 0,
    "length_function": len,
}

text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CHUNK_PARAMS)

documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.delete_index(st.secrets["PINECONE_INDEX_NAME"])
pc.create_index(
    name=st.secrets["PINECONE_INDEX_NAME"],
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

db = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=st.secrets["PINECONE_INDEX_NAME"]
)
