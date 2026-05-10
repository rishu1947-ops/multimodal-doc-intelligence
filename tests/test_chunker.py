# test_chunker.py
from app.ingestion.extractor import extract_text_native_pdf
from app.indexing.chunker import chunk_document

PDF_PATH = r"C:\Users\rishi\OneDrive\Documents\projects\Multimodal doc intelligence\9 Reference Materials KNN and Deep Learning.pdf"

def test_chunking():
    doc = extract_text_native_pdf(PDF_PATH, document_id="knn-demo")
    print(f"Extracted {len(doc.pages)} pages.")
    
    # Use new parameters: lower similarity threshold = larger chunks
    chunks = chunk_document(
        doc,
        similarity_threshold=0.3,      # lower = fewer, larger chunks
        min_chunk_sentences=2,         # at least 2 sentences per chunk
        min_chunk_chars=100            # merge chunks smaller than 100 chars
    )
    print(f"Generated {len(chunks)} semantic chunks.")
    
    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Text preview: {chunk['text'][:200]}...")
    
    # Check chunk size distribution
    if chunks:
        lengths = [len(c['text']) for c in chunks]
        print(f"\nChunk length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f} chars")
    
    print("\n✅ Chunking complete. Ready for embedding and indexing.")

if __name__ == "__main__":
    test_chunking()