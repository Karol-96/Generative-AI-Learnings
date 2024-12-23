

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



######### Text TO IMAGE
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import OpenAI
import requests

llm = OpenAI(temprature=0.8)
prompt = PromptTemplate(
    input_variables=['image_desc'],
    template = (
        "Generate a concise prompt to generate image based on following description.",
        "{image_desc}"
    )
)

chain = LLMChain(llm=llm, prompt=prompt)

image_url = DallEAPIWrapper().run(chain.run("Halloween night at haunted mueseum"))


response = requests.get(image_url)
image_path = "haunted_house.png"
with open(image_path, "wb") as f:
    f.write(response.content)


from langchain_community.llms import Replicate
text2img = Replicate(
       model=(
        "stability-ai/stable-diffusion:"
        "27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478"
    ),
    model_kwargs={'image_dimensions':'512x512'}
)

image_url = text2img('a book cover of book on creating genai using python.')



###################
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI

chat = ChatOpenAI(
    model = "gpt-4-turbo",
    max_tokens = 256
)

chat.invoke([
    HumanMessage(
        content=[
            {"type": "text", "text": "What is this image showing"}, 
            {
                "type": "image_url", 
                "image_url": { "url": image_url, "detail": "auto", },
            }, 
        ]
    )
])

langchain_image = """
The image appears to be a diagram representing the architecture or components of a software system or framework related to language processing, possibly named LangChain or associated with a project or product called LangChain, based on the prominent appearance of that term. The diagram is organized into several layers or aspects, each containing various elements or modules:\n\n1. **Protocol**: This may be the foundational layer, which includes "LCEL" and terms like parallelization, fallbacks, tracing, batching, streaming, async, and composition. These seem related to communication and execution protocols for the system.\n\n2. **Integrations Components**: This layer includes "Model I/O" with elements such as the model, output parser, prompt, and example selector. It also has a "Retrieval" section with a document loader, retriever, embedding model, vector store, and text splitter. Lastly, there\'s an "Agent Tooling" section. These components likely deal with the interaction with external data, models, and tools.\n\n3. **Application**: The application layer features "LangChain" with chains, agents, agent executors, and common application logic. This suggests that the system uses a modular approach with chains and agents to process language tasks.\n\n4. **Deployment**: This contains "Lang'"
"""

prompt = PromptTemplate(
    input_variables = ['image_desc'],
    template = (
                "Simplify this image description into a concise prompt to generate an image: "
        "{image_desc}"
    )
)

chain = LLMChain(llm=llm,prompt = prompt)

prompt = chain.run(langchain_image)
print(prompt)

image_url = DallEAPIWrapper().run(prompt)

from IPython.display import Image, display
display(Image(url=image_url))