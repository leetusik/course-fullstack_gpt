import streamlit as st

# use splitter if docs too long.
# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader


def parse_page(soup):
    header = soup.find("nav", class_="p-navbar")
    if header:
        header.decompose()
    footer = soup.find("footer", class_="p-footer")
    if footer:
        footer.decompose()
    return str(soup.get_text())


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    # splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=400, chunk_overlap=50
    # )

    loader = SitemapLoader(
        url,
        filter_urls=[r"^(.*\/career\/).*"],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    # docs = loader.load_and_split(text_splitter=splitter)
    docs = loader.load()
    return docs


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)
