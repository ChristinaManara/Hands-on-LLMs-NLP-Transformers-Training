import chainlit as cl
from chainlit.input_widget import Select

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from dotenv import load_dotenv
import os

load_dotenv() # Load the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

@cl.on_chat_start
async def on_chat_start():

    settings = await cl.ChatSettings(
            [
                Select(
                    id="Method",
                    label="Summarization Methods.",
                    values=["stuffing", "map-reduce", "refine"],
                    initial_index=0,
                )
            ]
        ).send()
    value = settings["Method"]
    #cl.user_session.set("method", value)

    res = await cl.AskUserMessage(content="Please provide a URL:", timeout=30).send()
    if res:
        await cl.Message(
            content=f"The URL you provided is: {res['output']}",
        ).send()
        #cl.user_session.set("url", res['output'])
        url = res['output']

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

    # Instantiate chain
    chain = create_stuff_documents_chain(llm, prompt)

    # Invoke chain
    result = chain.invoke({"context": splits})
    print(result)

    # Send the response back to the user
    await cl.Message(content=f"Answer: {result}").send()
