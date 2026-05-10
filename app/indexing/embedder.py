# app/indexing/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    """Wrapper for sentence-transformers to generate embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 embedding size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (documents/chunks)."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.model.encode([text])[0].tolist()

# Singleton for reuse
_embedder = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder