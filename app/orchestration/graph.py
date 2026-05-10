# app/orchestration/graph.py
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import generate_answer
from app.evaluation.ragas_eval import compute_ragas_scores
from app.evaluation.llm_judge import llm_as_judge

class GraphState(TypedDict):
    document_id: str
    query: str
    top_k: int
    retrieved_chunks: List[Dict]
    answer: str
    citations: List[Dict]
    eval_scores: Optional[Dict]
    error: Optional[str]

def retrieve_node(state: GraphState) -> GraphState:
    try:
        retriever = HybridRetriever()
        chunks = retriever.retrieve(state["document_id"], state["query"], top_k=state["top_k"])
        return {**state, "retrieved_chunks": chunks}
    except Exception as e:
        return {**state, "error": str(e)}

def generate_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    try:
        result = generate_answer(state["query"], state["retrieved_chunks"])
        return {**state, "answer": result["answer"], "citations": result["citations"]}
    except Exception as e:
        return {**state, "error": str(e)}

def evaluate_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    try:
        # Prepare contexts (list of chunk texts)
        contexts = [chunk["text"] for chunk in state["retrieved_chunks"]]
        
        # RAGAS scores
        ragas = compute_ragas_scores(state["query"], state["answer"], contexts)
        
        # LLM-as-judge score
        judge_score = llm_as_judge(state["query"], state["answer"], contexts)
        
        eval_scores = {
            "faithfulness": ragas["faithfulness"],
            "answer_relevancy": ragas["answer_relevancy"],
            "llm_judge_score": judge_score
        }
        return {**state, "eval_scores": eval_scores}
    except Exception as e:
        # If evaluation fails, still return state without scores
        return {**state, "eval_scores": {"error": str(e)}}

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)
    
    return workflow.compile()