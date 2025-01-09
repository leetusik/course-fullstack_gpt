import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="QuizGPT", page_icon="🎓")

st.title("QuizGPT")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)


@st.cache_resource(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = TextLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    option = st.selectbox(
        "Choose what you want to use",
        options=["File", "Wikipedia"],
    )

    if option == "File":
        file = st.file_uploader("Upload a file", type=["txt"])
        if file:
            docs = split_file(file)
            st.write(docs)

    elif option == "Wikipedia":
        subject = st.text_input("Search Wikipedia...")
        if subject:
            retriever = WikipediaRetriever(
                top_k_results=2,
            )
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(subject)
                st.write(docs)

if not docs:
    st.markdown("Please upload a file or search Wikipedia to get started.")
else:
    st.write(docs)
