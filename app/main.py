# app/main.py
import os
import shutil
import uuid
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import traceback
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi import Form
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import your pipeline modules
from app.ingestion.classifier import classify_pdf
from app.ingestion.extractor import extract_text_native_pdf, extract_scanned_pdf, extract_scanned_image
from app.indexing.chunker import chunk_document
from app.indexing.chroma_store import ChromaStore
from app.indexing.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import generate_answer
from app.evaluation.ragas_eval import compute_ragas_scores
from app.evaluation.llm_judge import llm_as_judge
from app.ingestion.extractor import describe_query_image
from app.observability.langfuse_client import langfuse, flush

app = FastAPI(title="Multimodal Document Intelligence API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store for document metadata and query logs (for demo; use persistent DB in production)
documents_store = {}       # doc_id -> metadata
query_logs = {}            # doc_id -> list of query records

# Shared pool for RAGAS / LLM-judge so blocking eval code does not stall the event loop
executor = ThreadPoolExecutor(max_workers=4)

# ------------------------------------------------------------------
# Helper: process a document (ingestion)
# ------------------------------------------------------------------
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
PDF_EXTENSIONS = {'.pdf'}

def process_document(file_path: str, doc_id: str) -> dict:
    start_time = time.time()
    file_ext = os.path.splitext(file_path)[1].lower()

    # Route based on file type
    if file_ext in IMAGE_EXTENSIONS:
        # Single image → treat as one-page scanned document
        extracted_doc = extract_scanned_image(file_path, doc_id)
    elif file_ext in PDF_EXTENSIONS:
        # PDF: classify as text-native or scanned
        doc_type, _ = classify_pdf(file_path)
        if doc_type == "text_native":
            extracted_doc = extract_text_native_pdf(file_path, doc_id)
        else:
            extracted_doc = extract_scanned_pdf(file_path, doc_id)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF")
    
    # 3. Chunk
    chunks = chunk_document(extracted_doc, similarity_threshold=0.3, min_chunk_sentences=2, min_chunk_chars=100)
    
    # 4. Index with ChromaDB
    chroma = ChromaStore(persist_directory="./chroma_db")
    chroma.create_collection(doc_id)
    chroma.add_chunks(doc_id, chunks)
    
    # 5. Index with BM25
    bm25 = BM25Store(persist_dir="./bm25_indexes")
    bm25.build_index(doc_id, chunks)
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    metadata = {
        "document_id": doc_id,
        "pages_processed": len(extracted_doc.pages),
        "chunks_indexed": len(chunks),
        "extraction_summary": {
            "tables_found": sum(len(p.tables) for p in extracted_doc.pages),
            "entities_found": 0  # placeholder, could be extended
        },
        "processing_time_ms": processing_time_ms
    }
    documents_store[doc_id] = metadata
    query_logs[doc_id] = []  # initialize query list
    return metadata

# ------------------------------------------------------------------
# API Models
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    document_id: str
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict]
    eval_scores: Dict
    latency_ms: int

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type up front
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    file_ext = os.path.splitext(file.filename)[1]
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{file_ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        metadata = process_document(save_path, doc_id)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))
    
    # Cleanup uploaded file (optional)
    os.remove(save_path)
    
    return metadata

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(
        f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        status_code=500
    )

@app.post("/query")
async def query_document(
    document_id: str = Form(...),
    query: str = Form(...),
    top_k: int = Form(3),
    image: UploadFile = File(None),
    run_llm_judge: bool = Form(False)
):
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")

    start_time = time.time()

    # If image provided, extract description and augment query
    image_context = None
    augmented_query = query

    if image is not None:
        img_ext = os.path.splitext(image.filename)[1]
        img_path = os.path.join(UPLOAD_DIR, f"query_{uuid.uuid4()}{img_ext}")
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        try:
            image_context = describe_query_image(img_path)
            augmented_query = f"{query}\n\nImage context: {image_context}"
        finally:
            os.remove(img_path)

    # --- Langfuse tracing (v2) ---
    trace = langfuse.trace(
        name="query_document",
        metadata={"document_id": document_id, "query": query}  
    )

    # Retrieve
    retrieve_span = trace.span(name="retrieve")
    retriever = HybridRetriever()
    chunks = retriever.retrieve(document_id, augmented_query, top_k=top_k)
    retrieve_span.update(output={"num_chunks": len(chunks)})
    retrieve_span.end()

    # Generate
    generate_span = trace.span(name="generate")
    gen_result = generate_answer(query, chunks, model="qwen3:8b", image_context=image_context)  
    answer = gen_result["answer"]
    raw_citations = gen_result["citations"]
    generate_span.update(output={"answer_length": len(answer), "num_citations": len(raw_citations)})
    generate_span.end()

   # Evaluate
    evaluate_span = trace.span(name="evaluate")
    contexts = [ch["text"] for ch in chunks]

    loop = asyncio.get_running_loop()
    ragas_future = loop.run_in_executor(executor, compute_ragas_scores, query, answer, contexts)
    judge_future = loop.run_in_executor(executor, llm_as_judge, query, answer, contexts) if run_llm_judge else None

    if judge_future:
        ragas, judge_score = await asyncio.gather(ragas_future, judge_future)
    else:
        ragas = await ragas_future
        judge_score = None

    eval_scores = {
        "faithfulness": ragas["faithfulness"],
        "answer_relevance": ragas["answer_relevancy"],
        "llm_judge_score": judge_score
    }
    evaluate_span.update(output=eval_scores)
    evaluate_span.end()

    # Attach scores to trace
    trace.score(name="faithfulness", value=eval_scores["faithfulness"])
    trace.score(name="answer_relevance", value=eval_scores["answer_relevance"])
    trace.score(name="llm_judge_score", value=eval_scores["llm_judge_score"])

    # Flush to Langfuse
    flush()

    # --- Compute latency ---
    latency_ms = int((time.time() - start_time) * 1000)

    # --- Store logs ---
    query_logs[document_id].append({       
        "query": query,                    
        "answer": answer,
        "eval_scores": eval_scores,
        "latency_ms": latency_ms
    })

    clean_citations = []
    for item in raw_citations:
        if isinstance(item, dict) and "page_number" in item:
            clean_citations.append({"page_number": item["page_number"]})
        elif isinstance(item, int):
            clean_citations.append({"page_number": item})
    if not clean_citations and raw_citations:
        import re
        found = re.findall(r'\([Pp]age (\d+)\)', answer)
        clean_citations = [{"page_number": int(p)} for p in set(found)]

    return {
        "answer": answer,
        "citations": clean_citations,
        "eval_scores": eval_scores,
        "latency_ms": latency_ms
    }

@app.get("/eval/summary/{document_id}")
def evaluation_summary(document_id: str):
    if document_id not in query_logs:
        raise HTTPException(status_code=404, detail="No queries found for this document")
    
    logs = query_logs[document_id]
    if not logs:
        return {"message": "No queries yet"}
    
    total_queries = len(logs)
    avg_faithfulness = sum(q["eval_scores"]["faithfulness"] for q in logs) / total_queries
    avg_relevance = sum(q["eval_scores"]["answer_relevance"] for q in logs) / total_queries
    avg_latency = sum(q["latency_ms"] for q in logs) / total_queries
    
    return {
        "document_id": document_id,
        "total_queries": total_queries,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_relevance": avg_relevance,
        "avg_latency_ms": avg_latency
    }