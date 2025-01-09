import streamlit as st
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer

st.set_page_config(page_title="SiteGPT", page_icon="🌐")

st.title("SiteGPT")

with st.sidebar:
    url = st.text_input("Enter a URL", placeholder="https://www.example.com")

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    st.write(docs)
    st.write("---")
    docs = Html2TextTransformer().transform_documents(docs)
    st.write(docs)
