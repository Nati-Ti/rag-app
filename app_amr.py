from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
from llama_index.core import ServiceContext, set_global_service_context
from langchain_huggingface import HuggingFaceEndpoint
import os, json
from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import AutoMergingRetriever
import re, os
from flask import Flask, request, render_template
from langchain.prompts import ChatPromptTemplate


load_dotenv()
app = Flask(__name__)

HG_KEY = os.getenv('HG_KEY')
HG_URL = os.getenv('HG_URL')

# Initialize HuggingFaceEndpoint and embedding model
llm = HuggingFaceEndpoint(
    endpoint_url=HG_URL,
    task="text-generation",
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HG_KEY
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", max_length=512)

from llama_index.core import Document
from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()

docs0 = loader.load(file_path=Path("./data/aws_faq.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

# Load documents
# with open('./data/context.html', 'r', encoding='utf-8') as file:
#     html_content = file.read()

# Load the content using SimpleWebPageReader
# documents = SimpleWebPageReader(html_to_text=True).load_data([html_content])


# Set chunk sizes and parse nodes
chunk_sizes = [512, 256, 128]
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
nodes = node_parser.get_nodes_from_documents(docs)
leaf_nodes = get_leaf_nodes(nodes)


Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", max_length=512)
Settings.node_parser = node_parser
Settings.num_output = 512
Settings.context_window = 3900

# Storage context
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, service_context=Settings)


re_rank = True
similarity_top_k = 12
rerank_top_n = 4

base_retriever = automerging_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)


# from llama_index.core.response.notebook_utils import display_source_node

# for node in nodes:
#     display_source_node(node, source_length=10000)


# for node in base_nodes:
#     display_source_node(node, source_length=10000)

def get_context(query):
    
    nodes = retriever.retrieve(query)
    base_nodes = base_retriever.retrieve(query)


    query_engine = RetrieverQueryEngine.from_args(retriever)
    base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

    response1 = query_engine.query(query)
    response2 = base_query_engine.query(query)

    return(str(response1), str(response2))


# get_context("AWS Marketplace?")


def chat_with_rag_swr(message):
    
    context = get_context(message)
    
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        prompt
        | llm
    )

    formatted_context = "\n\n".join(context)
    
    response = chain.invoke({"context": formatted_context, "question": message})
    
    return response




@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']
    try:
        bot_message = chat_with_rag_swr(user_message)
        pattern = r"Answer:\s*(.*)"
        match = re.search(pattern, bot_message, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return {'response': answer}
        else:
            return {'response': "Answer not found as per context"}
    except Exception as e:
        print(f"Error in chat: {e}")
        return {'response': "An error occurred while processing your request."}

if __name__ == '__main__':
    app.run()
