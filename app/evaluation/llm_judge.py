# app/evaluation/llm_judge.py
import ollama
import re
from typing import List, Dict

def llm_as_judge(question: str, answer: str, contexts: List[str], model: str = "qwen3:8b") -> float:
    """
    Asks an LLM to rate factual grounding from 1 to 5.
    Returns a float score (1 to 5).
    """
    context_str = "\n".join(contexts)
    prompt = f"""You are an evaluator. Rate the factual grounding of the answer using the provided context.
Rate from 1 to 5, where:
1 = Answer completely contradicts context
3 = Answer partially matches but has unsupported claims
5 = Answer fully grounded in context, no hallucinations

Context:
{context_str}

Question: {question}

Answer: {answer}

Respond with ONLY a number between 1 and 5, nothing else.

Score:"""
    
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip()
    # Extract first number
    match = re.search(r'\d+(?:\.\d+)?', raw)
    if match:
        score = float(match.group())
        return max(1.0, min(5.0, score))
    return 3.0  # default middle score