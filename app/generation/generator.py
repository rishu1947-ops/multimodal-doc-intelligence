# app/generation/generator.py
import ollama
import re
from typing import List, Dict

def generate_answer(query: str, chunks: List[Dict], model: str = "llama3.2:latest", image_context: str = None) -> Dict:
    context_parts = []
    for chunk in chunks:
        page = chunk.get("page_number", "?")
        context_parts.append(f"[Page {page}] {chunk['text']}")
    context = "\n\n".join(context_parts)

    # Add image description as extra context if provided
    image_section = ""
    if image_context:
        image_section = f"\nImage provided with query:\n{image_context}\n"

    prompt = f"""Answer using ONLY the context below. Cite pages as (Page X).

        Context:
        {context}
        {image_section}
        Question: {query}
        Answer:"""

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    citations = list(set(re.findall(r'\([Pp]age (\d+)\)', answer)))
    return {
        "answer": answer,
        "citations": [{"page_number": int(p)} for p in citations],
        "used_chunks": chunks
    }