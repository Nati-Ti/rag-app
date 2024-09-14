# import json
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sentence_transformers import SentenceTransformer, util
# from app_rag import chat_with_rag
# from app_swr import chat_with_rag_swr

# # Load the test dataset
# def load_test_dataset(file_path="data/test_data.json"):
#     """
#     Load the test dataset from the specified file path.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # Simulate chatbot response generation
# def generate_chatbot_response_rag(question):
#     """
#     Generate chatbot response using chat_with_rag.
#     """
#     return chat_with_rag(question)

# def generate_chatbot_response_swr(question):
#     """
#     Generate chatbot response using chat_with_rag_swr.
#     """
#     return chat_with_rag_swr(question)

# # Compare responses using sentence similarity (semantic similarity)
# def calculate_similarity(generated, expected, model):
#     """
#     Calculate cosine similarity between the generated response and the expected response.
#     """
#     embedding_1 = model.encode(generated, convert_to_tensor=True)
#     embedding_2 = model.encode(expected, convert_to_tensor=True)
    
#     # Calculate cosine similarity
#     similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
#     return similarity.item()

# # Evaluate chatbot responses
# def evaluate_chatbot(test_file="data/test_data.json"):
#     """
#     Evaluate chatbot responses by comparing them to expected responses and computing precision, recall, and F1 scores.
#     """
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
#     # Load the test dataset
#     test_data = load_test_dataset(test_file)
    
#     similarities_rag = []
#     similarities_swr = []

#     for data in test_data:
#         question = data["question"]
#         expected_answer = data["expected_answer"]
        
#         # Generate chatbot responses
#         generated_answer_rag = generate_chatbot_response_rag(question)
#         generated_answer_swr = generate_chatbot_response_swr(question)
        
#         # Calculate similarities
#         similarity_rag = calculate_similarity(generated_answer_rag, expected_answer, model)
#         similarity_swr = calculate_similarity(generated_answer_swr, expected_answer, model)
        
#         similarities_rag.append(similarity_rag)
#         similarities_swr.append(similarity_swr)
    
#     # Set a threshold to classify if a response is correct (e.g., similarity > 0.7)
#     threshold = 0.5
#     y_pred_rag = [1 if sim >= threshold else 0 for sim in similarities_rag]
#     y_pred_swr = [1 if sim >= threshold else 0 for sim in similarities_swr]
#     y_true = [1] * len(similarities_rag)  # Assume all expected answers are correct

#     # Calculate precision, recall, and F1 score for each chatbot
#     precision_rag = precision_score(y_true, y_pred_rag)
#     recall_rag = recall_score(y_true, y_pred_rag)
#     f1_rag = f1_score(y_true, y_pred_rag)

#     precision_swr = precision_score(y_true, y_pred_swr)
#     recall_swr = recall_score(y_true, y_pred_swr)
#     f1_swr = f1_score(y_true, y_pred_swr)

#     # Print results
#     print("Results for chat_with_rag:")
#     print(f"Precision: {precision_rag}")
#     print(f"Recall: {recall_rag}")
#     print(f"F1 Score: {f1_rag}")
    
#     print("\nResults for chat_with_rag_swr:")
#     print(f"Precision: {precision_swr}")
#     print(f"Recall: {recall_swr}")
#     print(f"F1 Score: {f1_swr}")

# # Call the evaluation
# if __name__ == "__main__":
#     evaluate_chatbot()
