import chainlit as cl

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.chroma import Chroma
from langchain import hub
from langchain.schema import Document

from pypdf import PdfReader
from io import BytesIO
import os

from dotenv import load_dotenv

load_dotenv() # Load the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Whait for the user to upload a file
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

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    text_pdf = await pdf_reader(file=file)

    rag_chain = await chain(text_pdf=text_pdf)

    # Create user session to store data
    cl.user_session.set("rag_chain", rag_chain)

    # Send response back to user
    await cl.Message(
        content = f"Content parsed! Ask me anything related to the weblink!"
    ).send()


@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    rag_chain = cl.user_session.get("rag_chain")
    res = rag_chain.invoke(message.content)
    await cl.Message(content=res).send()

async def pdf_reader(file) -> str:

    # Create a pdf reader object
    reader = PdfReader(file.path)

    # Inform the user the number of pages in pdf file
    await cl.Message(
        content=f"`The uploaded pdf, {file.name}`, has {len(reader.pages)} pages."
    ).send()

    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    return pdf_text

async def chain(text_pdf):
    docs = Document(page_content=text_pdf)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents([docs])
    
    # Create the embedding
    # model_name = "mixedbread-ai/mxbai-embed-large-v1"
    # hf = HuggingFaceEmbeddings(
    #     model_name=model_name,
    # )
    # # Generate embeddings for each split
    # embeddings = [hf.embed_documents(text) for text in splits]

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def format_docs(splits):
        return "\n\n".join(split for split in splits)

    # Create chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain