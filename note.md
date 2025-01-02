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
### 5.2 ConversationBufferWindowMemory
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
### 5.3 ConservationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryMemory(llm=chat)

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})

def get_history():
    return memory.load_memory_variables({})

add_message("Hi my name is Lee and I love to run.", "Nice to meet you Mr. Lee. It's great to hear you like running!")
add_message("And My grandpa is awesome human being.", "Oh! I see. Your grandpa is great person.")

get_history()

# It might seems not like efficient but as conversation gets longer, the summary buffer gets better.
"""
The result:
{'history': "The human, named Lee, introduces himself and shares his love for running. The AI responds positively, expressing pleasure in hearing about Lee's interest in running. Lee also mentions that his grandpa is an awesome human being, to which the AI acknowledges and agrees that Lee's grandpa is a great person."}
"""
```

### 5.3(밀림) ConversationSummaryBufferMemory

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryBufferMemory(llm=chat, return_messages=True)
memory.max_token_limit = 100

def add_message(input, output):
    memory.save_context({"input":input}, {"output":output})

def get_history():
    return memory.load_memory_variables({})

add_message("Hi my name is Lee and I love to run.", "Nice to meet you Mr. Lee. It's great to hear you like running!")
add_message("And My grandpa is awesome human being.", "Oh! I see. Your grandpa is great person.")
add_message("He taught me how to fish when I was young.", "That's wonderful! Fishing with grandparents creates such special memories.")
add_message("We used to go camping by the river every summer.", "Those sound like beautiful summer traditions. Camping and fishing are great ways to bond.")
add_message("He also taught me to always be kind to others.", "Your grandpa sounds very wise. Teaching kindness is one of the most valuable lessons.")
add_message("I try to live by his example every day.", "That's really admirable. Carrying forward positive values from our elders is so important.")
add_message("I hope I can be that kind of role model someday too.", "I'm sure you will be! You clearly learned great values from your grandpa.")

get_history()

# the result
"""
{'history': [SystemMessage(content="The human, named Lee, introduces himself and shares his love for running. The AI responds positively, expressing pleasure in hearing about Lee's interest in running. Lee mentions that his grandpa is an awesome human being, to which the AI agrees. Lee adds that his grandpa taught him how to fish when he was young, and the AI acknowledges that fishing with grandparents creates special memories. Lee further shares that they used to go camping by the river every summer, and the AI notes that those sound like beautiful summer traditions. Lee concludes by stating that his grandpa taught him to always be kind to others."),
  AIMessage(content='Your grandpa sounds very wise. Teaching kindness is one of the most valuable lessons.'),
  HumanMessage(content='I try to live by his example every day.'),
  AIMessage(content="That's really admirable. Carrying forward positive values from our elders is so important."),
  HumanMessage(content='I hope I can be that kind of role model someday too.'),
  AIMessage(content="I'm sure you will be! You clearly learned great values from your grandpa.")]}
"""
```

### 5.4 ConversationKGMemory
`잘 이해못함. on Something 식으로 구분해서 저장하는듯`

```python
from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationKGMemory(llm=chat, return_messages=True)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})


add_message("Hi I'm Nicolas, I live in South Korea", "Wow that is so cool!")
memory.load_memory_variables({"input": "who is Nicolas"})
"""
{'history': [SystemMessage(content='On Nicolas: Nicolas lives in South Korea.')]}

"""

add_message("Nicolas likes kimchi", "Wow that is so cool!")
memory.load_memory_variables({"inputs": "what does nicolas like"})
"""
{'history': [SystemMessage(content='On Nicolas: Nicolas lives in South Korea. Nicolas likes kimchi.')]}

"""

```


### 5.5 Memory on LLMChain

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
)

template = """
    You are a helpful AI talking to a human.

    {chat_history}
    Human:{question}
    You:
"""

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template(template),
    verbose=True,
)

chain.predict(question="My name is Nico")
chain.predict(question="I live in Seoul")
chain.predict(question="What is my name?")
 
```

### 5.6 Chat Based Memory
`memory output could be in two ways. one is just string and second is messages?`
`link memory and predefined chain.`

Just String:
```python
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
)

memory.load_memory_variables({})

# return in string
```

Messages:
```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    # chat history is here.
    memory_key="chat_history",
    return_messages=True,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI taking to a human."),
    # chat history is here too.
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

chain.predict(question="My name is Nico")
chain.predict(question="I live in Seoul")
chain.predict(question="What is my name?")
```




### 5.7 LCEL Based Memory
`link memory with custom chain(LCEL)`

```python

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]

chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm


def invoke_chain(question):
    result = chain.invoke({"question": question})
    # save_context -> history로 넘어감
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result)

invoke_chain("My name is nico")
invoke_chain("What is my name?")
```

### 5.8 Memory Recap
`I also think the manual process is better then automatics. `

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1,)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    return_messages=True,
)
# from messages!
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

def load_memory(input):
    return memory.load_memory_variables({})["history"]

chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

def chain_invoke(question):
    result = chain.invoke({
        "question":question
    })
    memory.save_context({"input":question},{"output":result.content},)
    print(result)

chain_invoke("my name is Sugang.")
```


## 6. RAG Retriever Augmented Generation

### 6.0 Introduction

### 6.1 Data Loaders and Splitters
`newer versions.. don't know how to do it. I think do it with legacy first and replace the deprecated things with new one is fastest way i guess right now. just keep work it. especially, unstructuredFileLoader not work at newer version or needed api key."

```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
)

loader = UnstructuredFileLoader("./files/moby_dick.pdf")
len(loader.load_and_split(text_splitter=splitter))
```

### 6.2 Tiktoken

```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# by adding from_tiktoken_encoder, chunk_size spllited by tokens not len(python basic)
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
)

loader = UnstructuredFileLoader("./files/moby_dick.pdf")
len(loader.load_and_split(text_splitter=splitter))
```

### 6.3,4 Vectors
```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore

cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50,
)

# load docs
loader = UnstructuredFileLoader("./files/moby_dick.txt")

# transform(split) docs
docs = loader.load_and_split(text_splitter=splitter)


embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)
# embed docs
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, cache_dir
)

# store docs as vectorstoree(?)
vectorstore = Chroma.from_documents(docs, cached_embeddings)

# retrieve docs
vectorstore.similarity_search("What does Ishmael do?")
```



###
###
###
###
###
---

## Problems
매개변수 (Parameter) : 함수를 정의할 때 사용되는 변수 (variable)
인자 (Argument) : 실제로 함수를 호출할 때 넘기는 변수값 (value)

### 5.3 ConversationSummaryBufferMemory

the error:
```base
NotImplementedError: get_num_tokens_from_messages() is not presently implemented for model cl100k_base.See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.
```

## Thoughts

4.1.1
    well maybe retriever takes relative context and fewshotPromptTemplate give those as example. that would be efficient.
    this handled by 4.3 LengthBasedExampleSelector. it's little bit diff I say, I don't know if the thing choose best example not just by length.

4.3
    serialize : saving, loading
    making customExampleselector.. need to do good in here.