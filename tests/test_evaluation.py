# test_evaluation.py
from app.orchestration.graph import build_graph

def test_evaluation():
    graph = build_graph()
    initial_state = {
        "document_id": "knn-demo",
        "query": "What is naive Bayes classification?",
        "top_k": 3,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "eval_scores": None,
        "error": None
    }
    result = graph.invoke(initial_state)
    
    print("--- Answer ---")
    print(result["answer"])
    print("\n--- Evaluation Scores ---")
    print(result.get("eval_scores", "No scores computed"))
    
    if result.get("error"):
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_evaluation()