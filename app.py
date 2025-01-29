import os
import numpy as np

from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone
from groq import Groq

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


def get_huggingface_embeddings(text, model_name=embedding_model):
    model = SentenceTransformer(model_name)
    return model.encode(text)


@app.route("/")
def home():
    print(
        "Routes: \n 1. /upload => to upload text from the pdf into pinecone \n 2. /rag to apply rag to the given text"
    )
    return "'app.py' successfully running"


@app.route("/upload", methods=["POST"])
def upload_translated_text():
    try:
        # Parse the incoming JSON payload
        data = request.json
        translated_text = data.get("text")
        index_name = "sem3"
        file_name = data.get("file_name")
        namespace = file_name

        # Validate required fields
        if not translated_text or not file_name:
            return jsonify({"error": "Translated text and file_name is required."}), 400

        # Split text into chunks
        chunked_text = split_text_into_documents([translated_text], file_name)

        # Process chunks into Pinecone document format
        documents = process_text_embedding(chunked_text)

        # Initialize Pinecone index
        pinecone_index = initialize_pinecone_index(index_name)  # noqa: F841

        # Insert documents into Pinecone
        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=index_name,
            namespace=namespace,
        )

        return jsonify(
            {
                "message": "Text successfully indexed into Pinecone.",
                "document_id": namespace,
                "index_name": index_name,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/rag", methods=["POST"])
def handle_rag_request():
    data = request.json

    query = data.get("query")
    index_name = "sem3"
    namespace = data.get("namespace")

    pinecone_index = initialize_pinecone_index(index_name)
    if not is_query_relevant(query, pinecone_index, namespace):
        return jsonify(
            {"error": "Your query does not seem related to the uploaded document."}
        ), 400
    response = perform_rag(query, pinecone_index, namespace)

    return jsonify({"response": response})


def cosine_similarity_between_sentences(sentence1, sentence2):
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


def process_text_embedding(texts, file_name):
    document_data = []
    for i, text in enumerate(texts):
        doc = Document(
            page_content=text,
            metadata={"file_name": file_name, "chunk_id": i},
        )
        document_data.append(doc)
    return document_data


#! Use this snippet if you want the each string in text[] to be treated as a separate document
def split_text_into_documents(texts, file_name, chunk_size=1024, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"file_name": file_name, "chunk_id": i},
                )
            )
    return documents


def initialize_pinecone_index(index_name):
    return pinecone_client.Index(index_name)


def is_query_relevant(query, pinecone_index, namespace, threshold=0.5):
    raw_query_embeddings = get_huggingface_embeddings(query)
    query_embeddings = np.array(raw_query_embeddings).reshape(1, -1)

    top_matches = pinecone_index.query(
        vector=query_embeddings.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace,
    )

    if not top_matches["matches"]:
        return False

    top_texts = [item["metadata"]["text"] for item in top_matches["matches"]]

    similarities = [
        cosine_similarity(
            query_embeddings, get_huggingface_embeddings(text).reshape(1, -1)
        )[0][0]
        for text in top_texts
    ]

    return max(similarities) > threshold


def perform_rag(query, pinecone_index, namespace):
    raw_query_embeddings = get_huggingface_embeddings(query)
    query_embeddings = np.array(raw_query_embeddings)
    top_matches = pinecone_index.query(
        vector=query_embeddings.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace,
    )
    contexts = [item["metadata"]["text"] for item in top_matches["matches"]]
    augmented_query = (
        "<CONTEXT>\n"
        + "\n\n-------\n\n".join(contexts[:10])
        + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n"
        + query
    )

    # Define system prompt
    system_prompt = (
        "You are an AI assistant with expertise in the agriculture domain. Your role is to generate detailed, contextually relevant, and accurate responses based strictly on the provided context and query."
        "\n\nStrictly adhere to the context and only respond based on the information provided."
        "\nDon't answer if the query is irrelevant to the context or out of scope."
    )
    # Query Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Bind to '0.0.0.0' to ensure the server is accessible externally
    app.run(host="0.0.0.0", port=port)
