# test_langgraph.py
from app.orchestration.graph import build_graph

def test_graph():
    graph = build_graph()
    initial_state = {
        "document_id": "knn-demo",
        "query": "What is naive Bayes classification?",
        "top_k": 3,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "error": None
    }
    result = graph.invoke(initial_state)
    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Citations ---")
    print(result["citations"])
    if result.get("error"):
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_graph()