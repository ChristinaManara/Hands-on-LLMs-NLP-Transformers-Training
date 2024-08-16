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


from pypdf import PdfReader
from io import BytesIO
import os

from dotenv import load_dotenv

load_dotenv() # Load the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

# Define the prompt template
# prompt = PromptTemplate(
#     template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
#     If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \
#     Question: {question} \
#     Context: {context} \
#     Answer:",
#     input_variables=["question", "context"]
# )

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

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

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Create a pdf reader object
    reader = PdfReader(file.path)

    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    docs = Document(page_content=pdf_text)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents([docs])

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(splits))]
    
    # Create the embedding
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
    )

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(splits, hf)
    retriever = vectorstore.as_retriever()

    # LLM
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Create chain
    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    rag_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=retriever,
    )


    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", splits)

    # Send response back to user
    await cl.Message(
        content = "Content parsed! Ask me anything related to the weblink!"
    ).send()

    # Save the metadata and texts in the user session
    # cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # Create user session to store data
    cl.user_session.set("rag_chain", rag_chain)



@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    print(chain)
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]

    await cl.Message(
        content = answer
    ).send()