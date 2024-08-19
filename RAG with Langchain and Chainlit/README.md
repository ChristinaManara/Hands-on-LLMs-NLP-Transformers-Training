# Chainlit PDF Question-Answering App

## Overview

This Chainlit application allows users to upload PDF files and interact with the content using a question-answering system. The application uses LangChain and FAISS for efficient retrieval of information from the uploaded PDF and answers user queries based on the extracted content.

## Features

- **PDF Upload**: Users can upload a PDF file.
- **Text Extraction**: Extracts text from the uploaded PDF.
- **Conversational Retrieval**: Uses FAISS and LangChain to provide answers based on the content of the PDF.
- **Contextual Responses**: Maintains conversation history to provide contextually relevant answers.

## Technologies

- **Chainlit**: For building the chat-based interface.
- **LangChain**: For managing document retrieval and question-answering.
- **FAISS**: For efficient similarity search and retrieval.
- **PyMuPDF**: For extracting text from PDFs.
- **Sentence Transformers**: For generating embeddings.

## Setup

### Prerequisites

1. **Python 3.8+**: Ensure you have Python 3.8 or higher installed.
2. **Virtual Environment** (optional but recommended): Create and activate a virtual environment.

### Install Dependencies

Create a `requirements.txt` file with the following content:

```text
chainlit
langchain-openai
langchain-huggingface
langchain-core
langchain-community
pymupdf
sentence-transformers
torch
python-dotenv
```

Install the dependencies using pip:

```pip install -r requirements.txt
```

### Environment Variables

Create a .env file in the root directory of your project and add the following:

```OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
```

### Usage

Running the Application

To start the Chainlit application, use the following command:

```chainlit run your_script_name.py
```

![Code Execution](https://github.com/ChristinaManara/Hands-on-LLMs-NLP-Transformers-Training/blob/main/RAG%20with%20Langchain%20and%20Chainlit/rag.gif)
