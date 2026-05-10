# app/indexing/bm25_store.py
import pickle
import os
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return word_tokenize(text.lower())

class BM25Store:
    def __init__(self, persist_dir: str = "./bm25_indexes"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
    
    def build_index(self, document_id: str, chunks: List[Dict]) -> None:
        """Build BM25 index from list of chunks (each has 'chunk_id' and 'text')."""
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        # Store both the bm25 object and the chunk metadata (ids, texts)
        data = {
            "bm25": bm25,
            "chunks": chunks  # store original chunk dicts to retrieve text later
        }
        save_path = os.path.join(self.persist_dir, f"{document_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def load_index(self, document_id: str) -> Optional[Dict]:
        save_path = os.path.join(self.persist_dir, f"{document_id}.pkl")
        if not os.path.exists(save_path):
            return None
        with open(save_path, "rb") as f:
            return pickle.load(f)
    
    def query(self, document_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Return top_k chunks as dicts with 'chunk_id', 'text', and 'score'."""
        data = self.load_index(document_id)
        if not data:
            return []
        bm25 = data["bm25"]
        chunks = data["chunks"]
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        # Get top_k indices
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in indexed[:top_k]]
        results = []
        for idx in top_indices:
            chunk = chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
        return results