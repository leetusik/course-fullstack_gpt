import time

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.set_page_config(page_title="DocumentGPT", page_icon="📄")

st.title("DocumentGPT")


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can answer questions about the document. Use ONLY following context to answer the question: {context}",
        ),
        ("human", "{question}"),
    ]
)


def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state.messages:
        send_message(message=message["message"], role=message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | chat
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state.messages = []
    st.markdown(
        """
Upload a document using sidebar's file uploader.\n
Ask about it!
"""
    )
