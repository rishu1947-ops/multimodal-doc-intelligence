# test_indexing.py
from app.ingestion.extractor import extract_text_native_pdf
from app.indexing.chunker import chunk_document
from app.indexing.chroma_store import ChromaStore
from app.indexing.bm25_store import BM25Store

PDF_PATH = r"C:\Users\rishi\OneDrive\Documents\projects\Multimodal doc intelligence\9 Reference Materials KNN and Deep Learning.pdf"

def test_indexing():
    # 1. Extract and chunk
    doc = extract_text_native_pdf(PDF_PATH, document_id="knn-demo")
    chunks = chunk_document(doc, similarity_threshold=0.3, min_chunk_sentences=2, min_chunk_chars=100)
    print(f"Chunks to index: {len(chunks)}")
    
    # 2. ChromaDB (dense)
    chroma = ChromaStore(persist_directory="./chroma_db")
    collection_name = "knn-demo"
    chroma.create_collection(collection_name)
    chroma.add_chunks(collection_name, chunks)
    print("✅ ChromaDB indexing complete")
    
    # 3. BM25 (sparse)
    bm25 = BM25Store(persist_dir="./bm25_indexes")
    bm25.build_index("knn-demo", chunks)
    print("✅ BM25 indexing complete")
    
    # 4. Quick query test
    query = "What is naive Bayes classification?"
    print(f"\n--- Query: {query} ---")
    
    # Dense
    dense_results = chroma.query(collection_name, query, n_results=2)
    print("Top dense results:")
    for i, (doc_text, dist) in enumerate(zip(dense_results['documents'][0], dense_results['distances'][0])):
        print(f"  {i+1}: {doc_text[:100]}... (dist={dist:.4f})")
    
    # Sparse
    sparse_results = bm25.query("knn-demo", query, top_k=2)
    print("Top BM25 results:")
    for i, res in enumerate(sparse_results):
        print(f"  {i+1}: {res['text'][:100]}... (score={res['bm25_score']:.2f})")
    
    print("\n✅ Indexing test complete. Both indexes ready for hybrid retrieval.")

if __name__ == "__main__":
    test_indexing()