# from flask import Flask, request, render_template
# import openai
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import FAISS, Chroma
# import re, os, json


# app = Flask(__name__)

# # HuggingFace API key
# hg_key = "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"

# # Initialize the HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     top_k=30,
#     temperature=0.1,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=hg_key
# )

# # Define your RAG chatbot function
# def chat_with_rag(message):
#     # Construct the path to doc_rag.txt
#     # data_file_path = os.path.join(os.path.dirname(__file__), "data", "doc_rag.txt")
    
#     # # Read the document text
#     # try:
#     #     with open(data_file_path, "r", encoding='utf-8') as f:
#     #         full_text = f.read()
#     # except FileNotFoundError:
#     #     return "Source document not found. Please upload the necessary file."


#     data_file_path = os.path.join(os.path.dirname(__file__), "data", "rag_context.json")
#     # Read the document JSON
#     try:
#         with open(data_file_path, "r", encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         return "Source document not found. Please upload the necessary file."

#     # Extract relevant context based on the message (e.g., FAQ, troubleshooting)
#     context = ""
    
#     # Search FAQs for matching questions
#     for faq in data.get("faq", []):
#         if message.lower() in faq["question"].lower():
#             context += faq["answer"] + "\n\n"
    
#     # If no relevant FAQs found, provide a default response
#     if not context:
#         context = "No relevant information found. Please refine your question."

#     # Split the text into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.split_text(context)

#     # Initialize embeddings with an explicit model name
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Initialize vector stores
#     # db_chroma = Chroma.from_texts(texts, embeddings)
#     db_faiss = FAISS.from_texts(texts, embeddings)
#     retriever = db_faiss.as_retriever()

#     # Define the prompt template
#     template = """Answer the question based only on the following context:
#     {context}
#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#     model = llm

#     # Function to format documents
#     def format_docs(docs):
#         return "\n\n".join([d.page_content for d in docs])

#     # Create the chain
#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | model
#     )

#     # Invoke the chain with the user message
#     return chain.invoke(message)

# # Define Flask routes
# @app.route('/')
# def home():
#     return render_template('bot_1.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.form['user_input']
#     bot_message = chat_with_rag(user_message)
    
#     # Define the regex pattern to extract the answer
#     pattern = r"Answer:\s*(.*)"
#     match = re.search(pattern, bot_message, re.DOTALL)

#     if match:
#         answer = match.group(1).strip()
#         print("Extracted Answer:", answer)
#         return {'response': answer}
#     else:
#         print("Answer not found")
#         return {'response': "Answer not found as per context"}

# if __name__ == '__main__':
#     app.run()
