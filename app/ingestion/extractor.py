# app/ingestion/extractor.py
import re
import fitz
from typing import List, Optional
from app.ingestion.schemas import DocumentPage, ExtractedDocument, KeyEntities
import base64
import io
import json
from pdf2image import convert_from_path
from PIL import Image
import ollama
from app.ingestion.schemas import ExtractedDocument, DocumentPage, ExtractedTable, KeyEntities

def extract_scanned_image(image_path: str, document_id: str) -> ExtractedDocument:
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_base64 = encode_image(img)
    page_text = call_vision_llm(img_base64, page_num=1)
    doc_page = DocumentPage(
        page_number=1,
        page_text=page_text,
        tables=[],
        key_entities=KeyEntities()
    )
    return ExtractedDocument(
        document_id=document_id,
        pages=[doc_page],
        source_type="scanned"
    )

def extract_text_native_pdf(pdf_path: str, document_id: str) -> ExtractedDocument:
    """
    Extract content from a text-native PDF directly using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Create a DocumentPage with raw text; tables and entities left empty for now
        doc_page = DocumentPage(
            page_number=page_num + 1,
            page_text=text,
            tables=[],           # can be enhanced later with table extraction
            key_entities=KeyEntities()
        )
        pages.append(doc_page)
    
    doc.close()
    
    return ExtractedDocument(
        document_id=document_id,
        pages=pages,
        source_type="text_native"
    )

# Placeholder for scanned extraction – will implement later
# app/ingestion/extractor.py (add these functions)


def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_vision_llm(image_base64: str, page_num: int) -> str:
    response = ollama.chat(
        model="moondream:latest",
        messages=[{
            "role": "user",
            "content": "Describe all the text, data, and content visible in this image.",
            "images": [image_base64]
        }],
        options={"num_predict": 512},
        keep_alive="1h"
    )
    return response["message"]["content"]

def parse_vision_response(page_num: int, response_text: str) -> DocumentPage:
    # Strip markdown code fences if the model wrapped the JSON
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        # Remove closing fence
        text = re.sub(r'\n?```$', '', text.strip())
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last-ditch: try to find a JSON object inside the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse JSON for page {page_num}: {text[:300]}") from e
        else:
            raise ValueError(f"No JSON found in vision response for page {page_num}: {text[:300]}")

    tables = [ExtractedTable(**t) for t in data.get("tables", [])]
    entities = KeyEntities(**data.get("key_entities", {}))
    return DocumentPage(
        page_number=page_num,
        page_text=data.get("page_text", ""),
        tables=tables,
        key_entities=entities
    )

def extract_scanned_pdf(pdf_path: str, document_id: str, dpi: int = 150) -> ExtractedDocument:
    images = convert_from_path(pdf_path, dpi=dpi)
    pages = []
    for idx, img in enumerate(images):
        page_num = idx + 1
        try:
            img_base64 = encode_image(img)
            page_text = call_vision_llm(img_base64, page_num=page_num)
            doc_page = DocumentPage(
                page_number=page_num,
                page_text=page_text,
                tables=[],
                key_entities=KeyEntities()
            )
            pages.append(doc_page)
        except Exception as e:
            print(f"Page {page_num} failed: {e}")
            pages.append(DocumentPage(
                page_number=page_num,
                page_text=f"[Error extracting page {page_num}: {e}]",
                tables=[],
                key_entities=KeyEntities()
            ))
    return ExtractedDocument(
        document_id=document_id,
        pages=pages,
        source_type="scanned"
    )

def describe_query_image(image_path: str) -> str:
    img = Image.open(image_path)        # fix: actually load the image
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_base64 = encode_image(img)      # fix: encode it before using
    response = ollama.chat(
        model="moondream:latest",
        messages=[{
            "role": "user",
            "content": "Describe the key content in this image briefly.",
            "images": [img_base64]
        }],
        options={"num_predict": 256}
    )
    return response["message"]["content"]