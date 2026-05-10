# app/retrieval/rrf.py
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    rankings: List[List[Dict]], 
    k: int = 60
) -> List[Dict]:
    """
    rankings: list of ranked lists, each is a list of chunk dicts (with at least 'chunk_id')
    Returns fused list sorted by RRF score.
    """
    scores = {}
    chunk_map = {}  # chunk_id -> chunk dict
    
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    
    sorted_chunk_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for chunk_id in sorted_chunk_ids:
        chunk = chunk_map[chunk_id].copy()
        chunk["rrf_score"] = scores[chunk_id]
        fused.append(chunk)
    return fused