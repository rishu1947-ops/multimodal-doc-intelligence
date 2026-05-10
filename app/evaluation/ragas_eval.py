# app/evaluation/ragas_eval.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

# Use the same embedder as before
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def compute_ragas_scores(question: str, answer: str, contexts: List[str]) -> Dict:
    """
    Local evaluation using cosine similarity.
    Faithfulness: avg similarity between answer and each context chunk.
    Answer relevance: similarity between answer and question.
    Returns dict with faithfulness and answer_relevancy scores (0-1).
    """
    embedder = get_embedder()
    
    # Embed question, answer, and contexts
    q_emb = embedder.encode(question)
    a_emb = embedder.encode(answer)
    ctx_embs = embedder.encode(contexts)
    
    # Faithfulness: average cosine similarity between answer and each context
    if len(ctx_embs) > 0:
        sims = [np.dot(a_emb, ctx) / (np.linalg.norm(a_emb) * np.linalg.norm(ctx)) for ctx in ctx_embs]
        faithfulness = float(np.mean(sims))
    else:
        faithfulness = 0.0
    
    # Answer relevance: cosine similarity between answer and question
    answer_relevancy = float(np.dot(a_emb, q_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(q_emb)))
    
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy
    }