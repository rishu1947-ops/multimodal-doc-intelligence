# app/retrieval/hybrid_retriever.py
from typing import List, Dict
from app.indexing.chroma_store import ChromaStore
from app.indexing.bm25_store import BM25Store
from app.retrieval.rrf import reciprocal_rank_fusion
import re

def extract_page_from_chunk_id(chunk_id: str) -> int:
    """Parse page number from chunk_id like 'knn-demo_p3_c12'."""
    match = re.search(r'_p(\d+)_', chunk_id)
    return int(match.group(1)) if match else -1

class HybridRetriever:
    def __init__(self, chroma_persist_dir: str = "./chroma_db", bm25_persist_dir: str = "./bm25_indexes"):
        self.chroma = ChromaStore(persist_directory=chroma_persist_dir)
        self.bm25 = BM25Store(persist_dir=bm25_persist_dir)
    
    def retrieve(self, document_id: str, query: str, top_k: int = 5) -> List[Dict]:
        # Dense retrieval
        collection_name = document_id
        dense_result = self.chroma.query(collection_name, query, n_results=top_k)
        dense_chunks = []
        if dense_result and 'ids' in dense_result and len(dense_result['ids'][0]) > 0:
            for i, chunk_id in enumerate(dense_result['ids'][0]):
                # Try metadata first, fallback to parsing chunk_id
                page = -1
                if dense_result['metadatas'] and dense_result['metadatas'][0]:
                    page = dense_result['metadatas'][0][i].get("page_number", -1)
                if page == -1:
                    page = extract_page_from_chunk_id(chunk_id)
                
                chunk = {
                    "chunk_id": chunk_id,
                    "text": dense_result['documents'][0][i],
                    "distance": dense_result['distances'][0][i],
                    "page_number": page,
                    "metadata": dense_result['metadatas'][0][i] if dense_result['metadatas'] else {}
                }
                dense_chunks.append(chunk)
        
        # Sparse retrieval
        sparse_chunks = self.bm25.query(document_id, query, top_k=top_k)
        for chunk in sparse_chunks:
            # Ensure page number exists
            if "page_number" not in chunk or chunk["page_number"] == -1:
                chunk["page_number"] = extract_page_from_chunk_id(chunk["chunk_id"])
        
        # Fuse
        fused = reciprocal_rank_fusion([dense_chunks, sparse_chunks], k=60)
        return fused[:top_k]