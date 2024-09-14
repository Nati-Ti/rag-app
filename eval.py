from app_rag import chat_with_rag
from app_swr import chat_with_rag_swr 
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def calculate_f1(reference, candidate):
    ref_tokens = set(reference.split())
    cand_tokens = set(candidate.split())
    mlb = MultiLabelBinarizer()
    ref_binary = mlb.fit_transform([ref_tokens])
    cand_binary = mlb.transform([cand_tokens])
    return f1_score(ref_binary, cand_binary, average='macro')

def evaluate_rag_system(queries_with_answers):
    

    
    results = []

    for query, ground_truth in queries_with_answers:
        
        generated_answer_rag = chat_with_rag(query)
        generated_answer_swr = chat_with_rag_swr(query)
        
        # Calculate BLEU, ROUGE, and F1 scores for both answers
        bleu_rag = calculate_bleu(ground_truth, generated_answer_rag)
        rouge_rag = calculate_rouge(ground_truth, generated_answer_rag)
        f1_rag = calculate_f1(ground_truth, generated_answer_rag)

        bleu_swr = calculate_bleu(ground_truth, generated_answer_swr)
        rouge_swr = calculate_rouge(ground_truth, generated_answer_swr)
        f1_swr = calculate_f1(ground_truth, generated_answer_swr)

        results.append({
            "query": query,
            "generated_answer_rag": generated_answer_rag,
            "generated_answer_swr": generated_answer_swr,
            "ground_truth": ground_truth,
            "bleu_rag": bleu_rag,
            "rouge_rag": rouge_rag,
            "f1_rag": f1_rag,
            "bleu_swr": bleu_swr,
            "rouge_swr": rouge_swr,
            "f1_swr": f1_swr
        })

    return results

queries_with_answers = [
    ("What are the benefits of using AWS?", "AWS provides scalability, reliability, and a wide range of services that can be used for different types of applications."),
    ("How can I improve the performance of my AWS Lambda function?",
    "You can improve performance by optimizing code, adjusting memory allocation, and using provisioned concurrency.")
]

results = evaluate_rag_system(queries_with_answers)

for result in results:
    print(f"Query: {result['query']}")
    print(f"Generated Answer (RAG): {result['generated_answer_rag']}")
    print(f"Generated Answer (SWR): {result['generated_answer_swr']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"BLEU Score (RAG): {result['bleu_rag']}")
    print(f"ROUGE Score (RAG): {result['rouge_rag']}")
    print(f"F1 Score (RAG): {result['f1_rag']}")
    print(f"BLEU Score (SWR): {result['bleu_swr']}")
    print(f"ROUGE Score (SWR): {result['rouge_swr']}")
    print(f"F1 Score (SWR): {result['f1_swr']}")
    print()

with open("evaluation_result.txt", "w") as f:
    for result in results:
        f.write(f"Query: {result['query']}\n")
        f.write(f"Generated Answer (RAG): {result['generated_answer_rag']}\n")
        f.write(f"Generated Answer (SWR): {result['generated_answer_swr']}\n")
        f.write(f"Ground Truth: {result['ground_truth']}\n")
        f.write(f"BLEU Score (RAG): {result['bleu_rag']}\n")
        f.write(f"ROUGE Score (RAG): {result['rouge_rag']}\n")
        f.write(f"F1 Score (RAG): {result['f1_rag']}\n")
        f.write(f"BLEU Score (SWR): {result['bleu_swr']}\n")
        f.write(f"ROUGE Score (SWR): {result['rouge_swr']}\n")
        f.write(f"F1 Score (SWR): {result['f1_swr']}\n")
        f.write("\n")
