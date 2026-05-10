# app/indexing/chunker.py
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from app.ingestion.schemas import ExtractedDocument

class ManualSemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.3):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold  # lower = larger chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Better sentence splitting using regex."""
        # Handle periods, question marks, exclamation marks, newlines
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        # Remove empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _merge_small_chunks(self, chunks: List[str], min_chars: int = 100) -> List[str]:
        """Merge chunks that are too small with the previous chunk."""
        if not chunks:
            return chunks
        
        merged = []
        buffer = chunks[0]
        
        for chunk in chunks[1:]:
            if len(buffer) < min_chars and len(chunk) < min_chars:
                # Merge two small chunks
                buffer = buffer + " " + chunk
            elif len(buffer) < min_chars:
                # Merge small buffer with next chunk
                buffer = buffer + " " + chunk
            else:
                merged.append(buffer)
                buffer = chunk
        
        if buffer:
            merged.append(buffer)
        return merged
    
    def chunk_document(
        self,
        extracted_doc: ExtractedDocument,
        min_chunk_sentences: int = 2,
        min_chunk_chars: int = 100
    ) -> List[Dict]:
        all_chunks = []
        global_chunk_idx = 0
        
        for page in extracted_doc.pages:
            sentences = self._split_sentences(page.page_text)
            if len(sentences) < min_chunk_sentences:
                # If page has very few sentences, treat whole page as one chunk
                chunks = [" ".join(sentences)]
            else:
                # Get embeddings for sentences
                embeddings = self.model.encode(sentences)
                
                # Group sentences into chunks based on similarity
                raw_chunks = []
                current_chunk = [sentences[0]]
                current_emb = embeddings[0]
                
                for i in range(1, len(sentences)):
                    sim = self._get_similarity(current_emb, embeddings[i])
                    
                    # If similarity low OR current chunk has enough sentences, break
                    if sim < self.similarity_threshold and len(current_chunk) >= min_chunk_sentences:
                        raw_chunks.append(' '.join(current_chunk))
                        current_chunk = [sentences[i]]
                        current_emb = embeddings[i]
                    else:
                        current_chunk.append(sentences[i])
                        # Update embedding as moving average
                        current_emb = np.mean(embeddings[max(0, i-len(current_chunk)+1):i+1], axis=0)
                
                if current_chunk:
                    raw_chunks.append(' '.join(current_chunk))
                
                # Merge any chunks that are too small
                chunks = self._merge_small_chunks(raw_chunks, min_chars=min_chunk_chars)
            
            # Convert to final chunk dicts
            for chunk_text in chunks:
                if len(chunk_text.strip()) < 20:  # skip extremely short leftovers
                    continue
                chunk = {
                    "chunk_id": f"{extracted_doc.document_id}_p{page.page_number}_c{global_chunk_idx}",
                    "text": chunk_text,
                    "page_number": page.page_number,
                    "metadata": {
                        "document_id": extracted_doc.document_id,
                        "source_type": extracted_doc.source_type,
                    }
                }
                all_chunks.append(chunk)
                global_chunk_idx += 1
        
        return all_chunks

def chunk_document(
    extracted_doc: ExtractedDocument,
    similarity_threshold: float = 0.3,  # lower = larger chunks
    min_chunk_sentences: int = 2,
    min_chunk_chars: int = 100
) -> List[Dict]:
    chunker = ManualSemanticChunker(similarity_threshold=similarity_threshold)
    return chunker.chunk_document(
        extracted_doc,
        min_chunk_sentences=min_chunk_sentences,
        min_chunk_chars=min_chunk_chars
    )