

#LLM

########## OPEN AI
from langchain_openai import OpenAI

llm = OpenAI()

summary = llm.invoke('tell me a joke about Nepal')
print(summary)

########## HuggingFace Hub

from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(
    models_kwargs = {"tempreature":0.5, "max_length":64},
    repo_id = 'google/flan-t5-xxl'
)

prompt = "In which country is everest? "

completion = llm.invoke(prompt)

print(completion)


############ FakeLLM
from langchain.llms.fake import FakeListLLM
fakellm = FakeListLLM(response=['LLM'])
fakellm.invoke('hi, FakelistLLM!')

########## Chat Models
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model='gpt-4-0613')
response = llm.invoke([HumanMessage(content='Say "Hello World" in python ')])
print(response)


from langchain_core.messages import SystemMessage

chat_output = llm.invoke([
    SystemMessage(content = 'you are a helpful assistant'),
    HumanMessage(content= 'What is Stochastic Parrot in AI concepts')
])

print(chat_output)


##Anthropic Model
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model='claude-3-opus-20240229')

response = llm.invoke([HumanMessage(content='What is the best language model in LLM in 2024?')])

print(response)


##############PROMPTS



