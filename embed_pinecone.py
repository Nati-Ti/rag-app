import os, json
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')


pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'aws-index-norm'


if index_name not in pinecone.list_indexes():

    pinecone.create_index(
        index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_ENVIRONMENT)
    )
pinecone_index = pinecone.Index(index_name)


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


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

text_chunks = []
for description in text_descriptions:
    chunks = text_splitter.split_text(description)
    text_chunks.extend(chunks)


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks)

docs = [(str(idx), embedding, {'text': chunk}) for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))]
pinecone_index.upsert(vectors=docs, show_progress=True)

print("Data successfully embedded and upserted to Pinecone.")
