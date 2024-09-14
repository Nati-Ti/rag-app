import os
import json
import re
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')


index_name = 'aws-index'
pinecone = Pinecone(api_key=PINECONE_API_KEY)


if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_ENVIRONMENT)
    )
pinecone_index = pinecone.Index(index_name)


def load_and_process_data():
    data_file_path = os.path.join(os.path.dirname(__file__), "data", "rag_context.json")
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_descriptions = []
    for item in data.get("faq", []):
        text_descriptions.append(item["answer"])
    for item in data.get("troubleshooting", []):
        text_descriptions.append(" ".join(item["steps"]))
    for item in data.get("knowledge_base", []):
        text_descriptions.append(item["content"])
    
    return text_descriptions


def sentence_window_chunking(text, window_size=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    for i in range(0, len(sentences), window_size):
        chunk = ' '.join(sentences[i:i + window_size])
        chunks.append(chunk)
    return chunks


def upsert_embeddings():
    
    text_descriptions = load_and_process_data()

    text_chunks = []
    for description in text_descriptions:
        chunks = sentence_window_chunking(description, window_size=3)
        text_chunks.extend(chunks)


    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)


    docs = [(str(idx), embedding.tolist(), {'text': chunk}) for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))]
    
    
    pinecone_index.upsert(vectors=docs, show_progress=True)

    print("Data successfully embedded and upserted to Pinecone.")

if __name__ == "__main__":
    upsert_embeddings()
