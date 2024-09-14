from flask import Flask, request, render_template
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import re, os
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)


HG_KEY = os.getenv('HG_KEY')
HG_URL = os.getenv('HG_URL')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_CLOUD = os.getenv('PINECONE_CLOUD')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')


llm = HuggingFaceEndpoint(
    endpoint_url=HG_URL,
    task="text-generation",
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HG_KEY
)

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'aws-index'
pinecone_index = pinecone.Index(index_name)


def get_data_from_pinecone(query, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()
    search_results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    context = [result['metadata']['text'] for result in search_results['matches']]
    return context


def chat_with_rag_swr(message):
    
    context = get_data_from_pinecone(message)
    
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
