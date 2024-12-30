## 2. Introduction

### 2.5 virtual environment

* making virtual environment

```bash
python -m venv .venv
source venv/bin/activate
```

* making requirements.txt

```bash
pip freeze > requirements.txt
```

* installing requirements.txt

```bash
pip install -r requirements.txt
```


### 2.6 jupyter notebook

`jupyter commends`

esc + d + d : delete cell
commend + enter : add code cell
^ + enter : execute cell(one)

## 3. Welcome to Langchain

### 3.0 LLMs and ChatModels

* Basic usage

```python
from langchain.chat_models import ChatOpenAI
# use gpt-4o-mini because it's best for our use case
chat = ChatOpenAI(model_name="gpt-4o-mini")

chat.predict("how many planets are there in the solar system?")
```

### 3.1 Predict Messages

I can give a chat model not only a string but also a list of messages.

```python
from langchain.schema import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions about the distance between cities. you only reply with Korean."),
    AIMessage(content="안녕하세요? 제 이름은 이수강이에요. 무엇을 도와드릴까요?"),
    HumanMessage(content="how far is the Seoul Korea from the Beijing China? And what is your name?"),
]

answer = chat.predict_messages(messages)
answer
```