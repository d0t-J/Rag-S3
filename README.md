# RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using Pinecone, Groq, and HuggingFace embeddings. The system is designed to handle text embeddings, document processing, and query responses in the agriculture domain. <br> 
**This RAG script was coded for the project [OOP S3](https://github.com/OOP_S3.git)**

## Table of Contents

- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
    - [Running the Main Script](#running-the-main-script)
    - [Running the Flask API](#running-the-flask-api)
- [Endpoints](#endpoints)
    - [Upload Translated Text](#upload-translated-text)
    - [Handle RAG Request](#handle-rag-request)

## Installation

1. Clone the repository:
        ```sh
        git clone <repository-url>
        cd <repository-directory>
        ```

2. Install the required dependencies:
        ```sh
        pip install -r requirements.txt
        ```

## Environment Variables

Create a `.env` file in the root directory and add the following environment variables:
```
PINECONE_API_KEY=<your_pinecone_api_key>
GROQ_API_KEY=<your_groq_api_key>
OPENROUTER_API_KEY=<your_openrouter_api_key>
HF_TOKEN=<your_huggingface_token>
```

## Usage

### Running the Main Script

To run the main script, execute the following command:
```sh
python RAG_main.py
```

### Running the Flask API

To start the Flask API, execute the following command:
```sh
python api/RAG_Backend.py
```

The API will be available at `http://0.0.0.0:8000`.

## Endpoints

### Upload Translated Text

- **URL:** `/upload`
- **Method:** `POST`
- **Description:** Accepts translated text and indexes it into Pinecone.
- **Request Body:**
        ```json
        {
                "text": "Translated text here",
                "index_name": "index_name",
                "file_name": "file_name"
        }
        ```
- **Response:**
        ```json
        {
                "message": "Text successfully indexed into Pinecone.",
                "document_id": "file_name",
                "index_name": "index_name"
        }
        ```

### Handle RAG Request

- **URL:** `/rag`
- **Method:** `POST`
- **Description:** Handles RAG queries and returns responses based on the provided context.
- **Request Body:**
        ```json
        {
                "query": "Your query here",
                "index_name": "index_name",
                "namespace": "namespace"
        }
        ```
- **Response:**
        ```json
        {
                "response": "Generated response based on the query and context."
        }
        ```

## Example Usage

1. **Initialize Pinecone Index:**
        ```python
        pinecone_index = initialize_pinecone_index("index_name")
        ```

2. **Process and Embed Text:**
        ```python
        plain_texts = ["Your text data here"]
        chunked_texts = split_text_into_documents(plain_texts)
        documents = process_text_embedding(chunked_texts)
        ```

3. **Insert Documents into Pinecone:**
        ```python
        vectorstore = PineconeVectorStore.from_documents(
                documents, embeddings, index_name="index_name", namespace="namespace"
        )
        ```

4. **Perform RAG Query:**
        ```python
        query = "Your query here"
        response = perform_rag(query, pinecone_index, "namespace")
        print(response)
        ```
