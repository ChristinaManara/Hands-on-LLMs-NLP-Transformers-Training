import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
import pymupdf

from pypdf import PdfReader
from io import BytesIO
import os
import asyncio

from dotenv import load_dotenv

load_dotenv() # Load the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

#model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    file_name = file.name
    
    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{file_name}` is uploaded."
    ).send()

    cl.user_session.set("file", file)

@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    file = cl.user_session.get("file")

    doc = pymupdf.open(file.path)
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text()

    docs = Document(page_content=pdf_text)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents([docs])

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(splits, hf)

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    history = []
    # Define the prompt template
    promptHist = PromptTemplate(
        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \
        Chat History: {chat_history} \
        Question: {question} \
        Context: {context} \
        Answer:",
        input_variables=["question", "context"]
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Also test "similarity", "mmr"
        search_kwargs={"k": 2},)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
        verbose = True,
        combine_docs_chain_kwargs={'prompt': promptHist},
        memory = memory,
        get_chat_history=lambda h : h,
    )

    # answer = qa({"question" : message.content, "chat_history" : history})
    # history.append((message.content, answer))
    # print(answer)
    
    # # res = await cl.make_async(llm_chain.run)(message.content)
    # await cl.Message(content=answer).send()
    # Get the answer from the ConversationalRetrievalChain
    response = await cl.make_async(qa.invoke)(
        {"question": message.content, "chat_history": history},
        callbacks=[cl.LangchainCallbackHandler()]
    )
    
    # Append the response to the history
    history.append({"question": message.content, "answer": response['answer']})

    # Send the response back to the user
    await cl.Message(content=f"Answer: {response['answer']}").send()