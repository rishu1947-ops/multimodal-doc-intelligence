# app/indexing/chroma_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from app.indexing.embedder import get_embedder

class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedder = get_embedder()
    
    def create_collection(self, collection_name: str) -> None:
        """Create a new collection (overwrites if exists)."""
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.client.create_collection(
            name=collection_name,
            embedding_function=None  # we'll provide embeddings manually
        )
    
    def add_chunks(self, collection_name: str, chunks: List[Dict]) -> None:
        """Add chunks to the collection. Each chunk dict must have 'chunk_id', 'text', and metadata."""
        collection = self.client.get_collection(collection_name)
        
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(texts)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def query(self, collection_name: str, query_text: str, n_results: int = 5) -> Dict:
        """Query the collection with a text string."""
        collection = self.client.get_collection(collection_name)
        query_embedding = self.embedder.embed_query(query_text)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]