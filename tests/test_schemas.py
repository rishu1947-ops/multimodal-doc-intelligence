# test_schemas.py
from app.ingestion.schemas import (
    ExtractedTable, KeyEntities, DocumentPage, ExtractedDocument
)
from datetime import datetime

def test_schemas():
    # Create a table
    table = ExtractedTable(headers=["Date", "Amount"], rows=[["2024-01-01", "$100"], ["2024-01-02", "$200"]])
    
    # Create entities
    entities = KeyEntities(dates=["2024-01-01"], amounts=["$100"], names=["John Doe"], identifiers=["INV-001"])
    
    # Create a page
    page = DocumentPage(
        page_number=1,
        page_text="Invoice summary...",
        tables=[table],
        key_entities=entities
    )
    
    # Create full document
    doc = ExtractedDocument(
        document_id="doc123",
        pages=[page],
        source_type="scanned"
    )
    
    # Convert to dict (simulates JSON response from LLM)
    print("Document model as dict:")
    print(doc.model_dump())
    
    # Validate that we can parse back from dict
    doc2 = ExtractedDocument.model_validate(doc.model_dump())
    assert doc2.document_id == "doc123"
    assert doc2.pages[0].tables[0].headers[0] == "Date"
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_schemas()