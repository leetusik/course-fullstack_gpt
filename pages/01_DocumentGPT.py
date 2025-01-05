import time

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# from langchain_core.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="DocumentGPT", page_icon="📄")

st.title("DocumentGPT")


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state.messages.append({"role": role, "message": message})


def paint_history():
    for message in st.session_state.messages:
        send_message(message=message["message"], role=message["role"], save=False)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=cached_embeddings,
    )
    retriever = vectorstore.as_retriever()
    return retriever


with st.sidebar:
    file = st.file_uploader(
        "Upload a document",
        type=["txt"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask a question about the document")
    if message:
        send_message(message=message, role="human")
        time.sleep(1)
        send_message(message="You said: " + message, role="ai")
else:
    st.session_state.messages = []
    st.markdown(
        """
Upload a document using sidebar's file uploader.\n
Ask about it!
"""
    )
