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

## 4. Modle I/O(Input/Output)

### 4.1 FewShotPromptTemplate

`FewShotPromptTemplate takes four arguments in general. examples, example_template(which made by PromptTemplate.from_template), suffix, and input_variable from suffix.`

```python
# PromptTemplate and ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()],)

# The PromptTemplate can be detailed like input = somthing and template = something but just this is shortcut. just know there are smth like that.
t = PromptTemplate.from_template("what is {this}")
x = ChatPromptTemplate.from_messages([("system", "what is {this}"), ])
```

```
In general, a "suffix" is a term used to describe a letter or group of letters added to the end of a word to change its meaning or grammatical function. For example, adding "-ing" to "run" forms "running," which changes the verb to its present participle form.
In programming, a "suffix" can refer to a string or sequence of characters added to the end of another string. It is often used in file naming conventions, such as file extensions (e.g., ".txt", ".jpg"), or in algorithms and data structures, such as suffix trees or arrays, which are used for various string processing tasks.
```

```python

examples = [
{
"question": "What do you know about France?",
"answer": """
Here is what I know:
Capital: Paris
Language: French
Food: Wine and Cheese
Currency: Euro
""",
},
{
"question": "What do you know about Italy?",
"answer": """
Here is what I know:
Capital: Rome
Language: Italian
Food: Pizza and Pasta
Currency: Euro
""",
},
{
"question": "What do you know about Greece?",
"answer": """
Here is what I know:
Capital: Athens
Language: Greek
Food: Souvlaki and Feta Cheese
Currency: Euro
""",
},
]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

# prompt.format(country="Germany")

chain = prompt | chat

chain.invoke({"country": "Japan"})
```

### 4.2 FewShotChatMessagePromptTemplate

`Well unlike Non ChatMessage prompt template, this one call chatprompttemplate twice. one for to put it in to fewshotchat..template and other for actual prompt. the thing have no suffix unlike previous one but put the real question at the end of fewshotchat..template`

```python
examples = [
{
"country": "France",
"answer": """
I know this:
Capital: Paris
Language: French
Food: Wine and Cheese
Currency: Euro
""",
},
{
"country": "Italy",
"answer": """
I know this:
Capital: Rome
Language: Italian
Food: Pizza and Pasta
Currency: Euro
""",
},
{
"country": "Greece",
"answer": """
I know this:
Capital: Athens
Language: Greek
Food: Souvlaki and Feta Cheese
Currency: Euro
""",
},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What do you know about {country}?"),
    ("ai", "{answer}"),
])

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert. you give short answers"),
    example_prompt,
    ("human", "What do you know about {country}?"),
])

chain = final_prompt | chat

chain.invoke({"country": "Japan"})
```

### 4.3 LengthBasedExampleSelector

`I need to handle example selector. like very well.`

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import example_selector
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()],)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        from random import choice
        return [choice(self.examples)]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")
example_selector = RandomExampleSelector(examples = examples)
prompt = FewShotPromptTemplate(
    # examples = examples -> to below
    example_selector=example_selector,
    
    example_prompt=example_prompt,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

prompt.format(country="Brazil")
```

### 4.4 Serialization and Composition


use it when lot of prompt or.. there are maker for prompting.

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import load_prompt

prompt = load_prompt("prompt.yaml")
prompt = load_prompt("prompt.json")

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()],)

prompt.format(country="Kimchi")
```

And here is somthing called pipelinePromptTemplate.
```json
// prompt.json
{
    "_type": "prompt",
    "template": "What is your {country}",
    "input_variables": ["country"]
}
```

```yaml
# prompt.yaml
_type: "prompt"
template: "What is your {country}"
input_variables: ["country"]
```

```python

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

intro = PromptTemplate.from_template(
    """
    You are a role playing assistant.
    And you are impersonating a {character}
"""
)

example = PromptTemplate.from_template(
    """
    This is an example of how you talk:

    Human: {example_question}
    You: {example_answer}
"""
)

start = PromptTemplate.from_template(
    """
    Start now!

    Human: {question}
    You:
"""
)

final = PromptTemplate.from_template(
    """
    {intro}
                                     
    {example}
                              
    {start}
"""
)

prompts = [
    ("intro", intro),
    ("example", example),
    ("start", start),
]


full_prompt = PipelinePromptTemplate(
    final_prompt=final,
    pipeline_prompts=prompts,
)


chain = full_prompt | chat

chain.invoke(
    {
        "character": "Pirate",
        "example_question": "What is your location?",
        "example_answer": "Arrrrg! That is a secret!! Arg arg!!",
        "question": "What is your fav food?",
    }
)
```

### 4.5 Caching

`the prompt needed to be exact same to use cached answers.`

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache

set_llm_cache(SQLiteCache("cache.db"))


chat = ChatOpenAI(
    temperature=0.1,
    # streaming=True,
    # callbacks=[
    #     StreamingStdOutCallbackHandler(),
    # ],
)

chat.predict("How do you make italian pasta")
```

### 4.6 Serialization

```python
# can check usage with get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

chat = ChatOpenAI(model_name="gpt-4o-mini")


with get_openai_callback() as usage:
    chat.predict("How do you make italian pizza and pasta")
    print(usage)
```

Can save models and load models when using llms. maybe chat models are also possible.



## 5. Memory

### 5.1 ConversationBufferMemory
```Memory is for Memorization for the chatbot. for continuos chat.```

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input":"Kimchi"}, {"output": "FA"})
memory.load_memory_variables({})
```
### 5.2
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(return_messages=True, k=4)
def add_message(input, output):
    memory.save_context({"input":input}, {"output": output})

# save 1, 1 to 5, 5
for i in range(1, 6):
    add_message(i, i)

# the result contains only 2, 2 to 5, 5
memory.load_memory_variables({})
```
### 5.3


## Problems
매개변수 (Parameter) : 함수를 정의할 때 사용되는 변수 (variable)
인자 (Argument) : 실제로 함수를 호출할 때 넘기는 변수값 (value)

## Thoughts

4.1.1
    well maybe retriever takes relative context and fewshotPromptTemplate give those as example. that would be efficient.
    this handled by 4.3 LengthBasedExampleSelector. it's little bit diff I say, I don't know if the thing choose best example not just by length.

4.3
    serialize : saving, loading
    making customExampleselector.. need to do good in here.