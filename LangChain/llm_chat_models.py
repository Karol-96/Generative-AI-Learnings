

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

prompt = """

Summarize this text in one sentence:
{text}
"""

llm = OpenAI()
summary = llm(prompt.format(text='Some story on Apple Inc.'))
print(summary)


###########
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(" Tell me a {adjective} joke about {content} ")

formatted_prompt = prompt_template.format(adjective='funny',content='chickens')

print(formatted_prompt)


prompt_val = prompt_template.invoke({'adjective':'funny','content':'chickens'})
prompt_val


######### PROMPT Templating Example 2
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

#Define a chatprompt template to translate text
template = ChatPromptTemplate.from_message([
    ('system','you are an english to french translator.'),
    ('user','Translate this to French: {text}')
])

llm = ChatOpenAI()
#Translate a joke about simpsons
response = llm.invoke(template.format_message(text = 'tell me a joke on simpsons intergalactic journey'))
print(response)



#VertexAI
from langchain_google_vertexai import ChatVertexAI
from langchain import PromptTemplate,LLMChain

llm = ChatVertexAI(model_name="gemini-pro")

template = """Question: {question}
Answer: Let's think step by step"""

prompt = PromptTemplate(template=template, input_variables=['question'])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
question = "What is the root square of infinity"
llm_chain.run(question)



########### Langchain Expression Language

from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

repo_id = "mistralai/Mistral-7B-Instruct-v0.2" 

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temprature= 0.5
)

template = """Question: {question} Answer: Let's think step by step"""
prompt = PromptTemplate.from_template(template)
runnable = prompt|llm|StrOutputParser()

question = 'Who won the last world cup'
summary = runnable.invoke({'question':question})
print(summary)

