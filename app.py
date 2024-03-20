import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint



repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.9, huggingfacehub_api_token=huggingfacehub_api_token
)
template = """
You are an helpful assistant. Give helpful, detailed, and correct answers to the user's questions. Do not make up answers.

{question}

"""



@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    print(template)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(prompt)
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    print(llm_chain.__dict__)
    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    question = message.content
    
    res = await llm_chain.arun(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
  

    # Send the response
    await cl.Message(content=res).send()

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import Runnable
# from langchain.schema.runnable.config import RunnableConfig

# import chainlit as cl
# import os
# huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']


# llm_chain = LLMChain(prompt=prompt, llm=llm)


# @cl.on_chat_start
# async def on_chat_start():
#     # model = ChatOpenAI(streaming=True)
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     model = HuggingFaceEndpoint(
#         repo_id=repo_id, max_length=128, temperature=0.5, token=huggingfacehub_api_token
#     )
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
#             ),
#             ("human", "{question}"),
#         ]
#     )

#     runnable = prompt | model | StrOutputParser()
#     cl.user_session.set("runnable", runnable)