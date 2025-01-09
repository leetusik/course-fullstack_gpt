import time

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

chat_for_memory = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)


# LCEL based memory
@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(
        llm=_llm,
        max_token_limit=120,
        return_messages=True,
    )


memory = init_memory(chat_for_memory)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.: {context}",
        ),
        MessagesPlaceholder(variable_name="history"),
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


def save_memory(input, output):
    memory.save_context({"input": input}, {"output": output})


def invoke_chain(chain, message):
    response = chain.invoke(message)
    save_memory(message, response.content)


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
                "history": RunnableLambda(load_memory),
            }
            | prompt
            | chat
        )
        with st.chat_message("ai"):
            invoke_chain(chain, message)


else:
    st.session_state.messages = []
    memory.clear()
    st.markdown(
        """
Upload a document using sidebar's file uploader.\n
Ask about it!
"""
    )
