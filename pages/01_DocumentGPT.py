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


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore("./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    splitter = CharacterTextSplitter(
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


file = st.file_uploader(
    "Upload a document",
    type=["txt"],
)

if file:
    retriever = embed_file(file)
    response = retriever.invoke("What is the main topic of the document?")
    st.write(response)
