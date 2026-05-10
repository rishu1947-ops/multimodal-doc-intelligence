# app/ingestion/classifier.py
import fitz  # PyMuPDF
from typing import Tuple, List

def classify_pdf(pdf_path: str, min_text_length_per_page: int = 50) -> Tuple[str, List[str]]:
    """
    Determines if a PDF is text-native or scanned.
    
    Args:
        pdf_path: path to PDF file
        min_text_length_per_page: if average characters per page < this, classify as scanned
    
    Returns:
        (classification, list_of_text_per_page)
        classification: "text_native" or "scanned"
    """
    doc = fitz.open(pdf_path)
    page_texts = []
    total_chars = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        page_texts.append(text)
        total_chars += len(text)
    
    doc.close()
    
    if len(page_texts) == 0:
        return "scanned", page_texts
    
    avg_chars = total_chars / len(page_texts)
    if avg_chars < min_text_length_per_page:
        return "scanned", page_texts
    else:
        return "text_native", page_texts