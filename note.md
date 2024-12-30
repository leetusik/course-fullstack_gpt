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

### 3.3 Output Parser and LCEL(LangChain Expression Language)

* base output parser
```python
from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))
```

* Output Parser With no LECL
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You only reply with comma separated list of items. Do not include any other text. You answer maximum {max_items} items."),
    ("human", "What are the cities in the world?"),
])

prompt = template.format_messages(max_items=10)

result = chat.predict_messages(prompt)

p = CommaOutputParser()
p.parse(result.content)
```

* With LECL
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a list generating machine. Everthing you are asked You need to answer with a comma separated list of max {max_items} in list. Do NOT reply with anything else."),
    ("human", "{question}")
])
# LCEL do format_messages, chat.predict, outputparser.parse with below chain.
chain = template | chat | CommaOutputParser()
chain.invoke({
  "max_items": 5,
  "question": "What are the pokemons?",
})

# result
# ['Pikachu', 'Charizard', 'Bulbasaur', 'Squirtle', 'Jigglypuff']
```

### 3.4 Chaining Chains
![chain components](/img/chain_components.png)
Gives Prompt when invoke.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# callbacks are magical things for now but explained later so nomad said.
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()],)

tail_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a fairy tail expert. You tell about the fairy tail's story that human ask."),
    ("human", "{fairy_tail}")
])

tail_chain = tail_prompt | chat
```

```python
odd_story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very good story teller. You get a story, and You answer about it but twist the plot little oddly."),
    ("human", "{story}")
])

odd_chain = odd_story_prompt | chat
```

```python
all_chain = {"story": tail_chain} | odd_chain
all_chain.invoke({
    "fairy_tail": "Peter Pan"
})

# It says crazy about peter pan. Maybe I can use it for later like youtube short things.
```

## Problems
