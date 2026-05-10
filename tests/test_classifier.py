# test_classifier.py
from app.ingestion.classifier import classify_pdf

# Replace with your actual PDF path
PDF_PATH = r"C:\Users\rishi\OneDrive\Documents\projects\Multimodal doc intelligence\9 Reference Materials KNN and Deep Learning.pdf"   

def test_classifier():
    doc_type, page_texts = classify_pdf(PDF_PATH)
    print(f"Document type: {doc_type}")
    print(f"Number of pages: {len(page_texts)}")
    
    # Show first 200 chars of first page (if exists)
    if page_texts:
        print("\nFirst 200 characters of page 1:")
        print(repr(page_texts[0][:200]))
    
    # Based on classification, we'll know which extraction path to use later
    if doc_type == "text_native":
        print("\n✅ This PDF is text-native. PyMuPDF will extract text directly.")
    else:
        print("\n⚠️ This PDF appears scanned. Vision LLM (Qwen2-VL) will be needed.")

if __name__ == "__main__":
    test_classifier()