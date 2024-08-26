import chainlit as cl
from chainlit.input_widget import Select, TextInput

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.summarize.chain import load_summarize_chain

from dotenv import load_dotenv
import os

load_dotenv() # Load the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    
    cl.user_session.set("url", settings["url"])
    cl.user_session.set("method", settings["method"])
    

@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            TextInput(id="url", label="YouTube URL: "),
            Select(
                id="method",
                label="Summarization Methods.",
                values=["stuff", "map_reduce", "refine"],
                initial_index=0,
            )
        ]
    ).send()

    cl.user_session.set("url", settings["url"])
    cl.user_session.set("method", settings["method"])
    
    

@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    url = cl.user_session.get("url")
    method = cl.user_session.get("method")

    await cl.Message(
        content=f"The URL you provided is: {url}",
    ).send()

    await cl.Message(
        content=f"The method you provided is: {method}",
    ).send()

    loader = YoutubeLoader.from_youtube_url(
            youtube_url=url,     
            add_video_info=True,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
        )

    splits = loader.load()

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

    # Define prompt
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )

    chain = load_summarize_chain(llm, chain_type=method)
    result = chain.run(splits)

    # Send the response back to the user
    await cl.Message(content=f"Answer: {result}").send()
