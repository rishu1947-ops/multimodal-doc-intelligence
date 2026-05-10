# test_generation.py
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import generate_answer

def test_generation():
    retriever = HybridRetriever()
    chunks = retriever.retrieve("knn-demo", "What is naive Bayes classification?", top_k=3)
    print(f"Retrieved {len(chunks)} chunks")
    
    result = generate_answer("What is naive Bayes classification?", chunks)
    
    print("\n--- Full result dict ---")
    print(result.keys())   # see what keys exist
    print("\n--- Answer ---")
    print(result.get("answer", "No answer key"))
    
    if "citations" in result:
        print("\n--- Citations ---")
        print(result["citations"])
    else:
        print("\n⚠️ 'citations' key not found in result. Trying 'citation' or other...")
        # fallback: try to extract citations from answer text
        import re
        citations = re.findall(r'\(Page (\d+)\)', result.get("answer", ""))
        print(f"Manually extracted citations from answer: {citations}")

if __name__ == "__main__":
    test_generation()