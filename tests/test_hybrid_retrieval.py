# test_hybrid_retriever.py
from app.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_retriever():
    retriever = HybridRetriever()
    document_id = "knn-demo"  # must match the collection name used in indexing
    query = "What is naive Bayes classification?"
    results = retriever.retrieve(document_id, query, top_k=3)
    
    print(f"Query: {query}\n")
    for i, res in enumerate(results):
        print(f"--- Result {i+1} (rrf_score={res.get('rrf_score', 0):.4f}) ---")
        print(f"Page: {res.get('page_number', -1)}")
        if 'distance' in res:
            print(f"Chroma distance: {res['distance']:.4f}")
        if 'bm25_score' in res:
            print(f"BM25 score: {res['bm25_score']:.2f}")
        print(f"Text: {res['text'][:200]}...\n")

if __name__ == "__main__":
    test_hybrid_retriever()