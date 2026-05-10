# app/generation/generator.py
import ollama
import re
from typing import List, Dict

def generate_answer(query: str, chunks: List[Dict], model: str = "qwen3:8b") -> Dict:
    context_parts = []
    for chunk in chunks:
        page = chunk.get("page_number", "?")
        context_parts.append(f"[Page {page}] {chunk['text']}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant that answers questions based only on the provided context.
Context:
{context}

Question: {query}

Instructions:
- Answer concisely using ONLY the context above.
- For each fact, cite the page number like (Page X).
- If the context does not contain the answer, say "The context does not provide enough information to answer this question."

Answer:"""
    
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    
    # Extract citations like (Page 1) or (page 1)
    citations = list(set(re.findall(r'\([Pp]age (\d+)\)', answer)))
    return {
        "answer": answer,
        "citations": [{"page_number": int(p)} for p in citations],
        "used_chunks": chunks
    }