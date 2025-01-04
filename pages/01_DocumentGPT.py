import time

import streamlit as st

st.set_page_config(page_title="DocumentGPT", page_icon="📄")

st.title("DocumentGPT")


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state.messages.append({"role": role, "content": message})


if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    for message in st.session_state.messages:
        send_message(message["content"], message["role"], save=False)


input = st.chat_input("Ask me anything!")

if input:
    send_message(input, "human")
    time.sleep(1)
    send_message(f"you said: {input}", "ai")
