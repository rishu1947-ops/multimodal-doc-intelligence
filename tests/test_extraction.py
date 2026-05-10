# test_extraction.py
from app.ingestion.extractor import extract_text_native_pdf

PDF_PATH = r"C:\Users\rishi\OneDrive\Documents\projects\Multimodal doc intelligence\9 Reference Materials KNN and Deep Learning.pdf"

def test_extraction():
    doc = extract_text_native_pdf(PDF_PATH, document_id="test-knn-doc")
    
    print(f"Document ID: {doc.document_id}")
    print(f"Source type: {doc.source_type}")
    print(f"Number of pages: {len(doc.pages)}")
    
    first_page = doc.pages[0]
    print(f"\nPage 1 text (first 300 chars):\n{first_page.page_text[:300]}...")
    print(f"\n✅ Extraction successful. ExtractedDocument object is valid.")

if __name__ == "__main__":
    test_extraction()