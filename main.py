from datetime import datetime

import streamlit as st

st.title("오늘 점심 뭐먹지")

st.write(f"현재 시간: {datetime.now().strftime('%H:%M:%S')}")

model = st.selectbox("choose your model", ["gpt-4o", "gpt-4o-mini"])

if model == "gpt-4o":
    st.write("gpt-4o is selected")
else:
    st.write("gpt-4o-mini is selected")
    st.slider("choose your temperature", 0.0, 1.0, 0.5)
