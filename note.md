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

### 3.2 Prompt Templates

`before magic`
```python
# Basic Setting
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(model_name="gpt-4o-mini")
```

```python
# when using PromptTemplate
template = PromptTemplate.from_template("how far is the {city_a} from the {city_b}? And what is your name?")

chat = ChatOpenAI(model_name="gpt-4o-mini")

prompt = template.format(city_a="Seoul", city_b="Beijing")
chat.predict(prompt)
```

```python
# when using ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about the distance between cities. you only reply with {language}."),
    ("ai", "안녕하세요? 제 이름은 {ai_name}이에요. 무엇을 도와릴까요?"),
    ("human", "how far is the {city_a} from the {city_b}? And what is your name?"),
])

prompt = template.format_messages(language="Japanese", ai_name="Nakamura", city_a="Seoul", city_b="Beijing")

chat.predict_messages(prompt)
```

PromptTemplate and ChatPromptTemplate are similar But ChatPromptTemplate is more flexible. And it makes template with list of messages(tuple).

### 3.3 Output Parser and LECL