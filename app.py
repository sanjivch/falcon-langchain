import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint



repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, huggingfacehub_api_token=huggingfacehub_api_token
)
template = """
You are an helpful assistant expert at machine learning. Give helpful, detailed, and correct answers to the user's questions. Do not make up answers.

{question}

"""



@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    print(template)
    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(prompt)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain)
    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    print(llm_chain)
    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    # res=await chain.acall(message.content, callbacks=[cb])
    # Do any post processing here
    content = res['text']
    content = content[content.find('Assistant:'):]

    # Send the response
    await cl.Message(content=content).send()