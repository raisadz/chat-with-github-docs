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


git_url = st.secrets["GIT_REPO"]
# the name of the local diroctory for cloning the Git repo 
# will be the same as the Pinecone index name
dir_copy = os.path.join('.', st.secrets["PINECONE_INDEX_NAME"])

dirpath = Path(dir_copy)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

Repo.clone_from(git_url, dir_copy)

folders = st.secrets['FOLDERS'].split(',')
docs = []
all_files = []
for folder in folders:
    for dirpath, dirnames, filenames in os.walk(os.path.join(dir_copy, folder)):
        for file in filenames:
            if not (file.endswith('.py') or file.endswith('.md')):
                continue
            full_file_name = dirpath + "/" + file
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

pc_indexes = [x['name'] for x in pc.list_indexes()]
if st.secrets["PINECONE_INDEX_NAME"] in pc_indexes:
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
