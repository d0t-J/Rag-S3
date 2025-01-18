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

@app.route('/', methods=['POST', 'GET'])
def home():
    print("Routes: \n 1. /upload => to upload text from the pdf into pinecone \n 2. /rag to apply rag to the given text")

@app.route("/upload", methods=["POST"])
def upload_translated_text():
    """
    Endpoint to accept translated text and index it into Pinecone.
    """
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
        chunked_text = split_text_into_documents([translated_text])

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

        return jsonify({
            "message": "Text successfully indexed into Pinecone.",
            "document_id": namespace,
            "index_name": index_name,
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/rag", methods=["POST"])
def handle_rag_request():
    data = request.json

    query = data.get("query")
    index_name = "sem3"
    namespace = data.get("namespace")

    pinecone_index = initialize_pinecone_index(index_name)
    response = perform_rag(query, pinecone_index, namespace)

    return jsonify({"response": response})


def cosine_similarity_between_sentences(sentence1, sentence2):
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


def process_text_embedding(texts):
    document_data = []
    for i, text in enumerate(texts):
        document_source = f"text {i + 1}"

        # Create a Document object with structured content and metadata
        doc = Document(
            page_content=f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{text}\n</Content>",
            metadata={
                "file_name": document_source,
            },
        )
        document_data.append(doc)
    return document_data


#! Use this snippet if you want the each string in text[] to be treated as a separate document
def split_text_into_documents(texts, chunk_size=2000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        documents.extend([Document(page_content=chunk) for chunk in chunks])
    return documents


def initialize_pinecone_index(index_name):
    return pinecone_client.Index(index_name)


def perform_rag(query, pinecone_index, namespace):
    raw_query_embeddings = get_huggingface_embeddings(query)
    query_embeddings = np.array(raw_query_embeddings)
    top_matches = pinecone_index.query(
        vector=query_embeddings.tolist(),
        top_k=10,
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
    "You are an AI assistant with expertise in the agriculture domain. Your role is to generate detailed, contextually relevant, and accurate responses based strictly on the provided context and query. Adhere to the following guidelines:"
    "\n\n1. **Strict Relevance to Agriculture:**"
    "\n   - Only respond if the query is directly related to the agriculture domain and aligns with the provided context. If the question is unclear, unrelated, or not in English, respond with an appropriate clarification request or indicate that the question is out of scope."
    "\n\n2. **Structured and Contextual Responses:**"
    "\n   - Analyze the provided context thoroughly before answering."
    "\n   - Organize information in clear formats such as lists, tables, or plain text summaries, tailored to the user's needs."
    "\n   - Avoid introducing information not grounded in the provided context or the agricultural domain."
    "\n\n3. **Coverage of Diverse Agricultural Topics:**"
    "\n   - Address various aspects of agriculture, including but not limited to:"
    "\n     - Historical practices and modern advancements."
    "\n     - Emerging technologies like agri-tech and AI applications."
    "\n     - Sustainable farming, agroforestry, irrigation techniques, and climate-resilient practices."
    "\n     - Challenges like pest and disease management or soil degradation, and solutions for these."
    "\n\n4. **Clarity and Accessibility:**"
    "\n   - Use concise, clear, and easy-to-understand language suitable for a broad audience, including farmers, researchers, and policymakers."
    "\n   - Avoid unnecessary technical jargon unless explicitly requested by the user."
    "\n\n5. **Customization and Precision:**"
    "\n   - Tailor responses to the query, ensuring they align with specific user needs, whether educational, practical, or advisory."
    "\n   - If the query requires analysis or recommendations, ensure all guidance is actionable and practical for agricultural applications."
    "\n\n6. **Boundary Enforcement:**"
    "\n   - Do not respond to queries outside the agriculture domain."
    "\n   - If a query lacks clarity, request additional information or clarification before proceeding."
    "\n   - Avoid speculative or generic responses; strictly adhere to the context provided in the query and accompanying data."
    "Don't answer if the query is irrelevant to the context or out of scope of the defined context."
    "\n\n7. **Ethical and Respectful Interaction:**"
    "\n   - Provide accurate, respectful, and ethical responses."
    "Don't provide response if the question is not related to agriculture domain or if the question is not clear or if the question is not in English language."
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

if __name__ == '__main__':
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Bind to '0.0.0.0' to ensure the server is accessible externally
    app.run(host='0.0.0.0', port=port)
