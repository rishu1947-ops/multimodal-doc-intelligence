# app/ingestion/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExtractedTable(BaseModel):
    """A table extracted from a document page."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)

class KeyEntities(BaseModel):
    """Key entities like dates, amounts, names, identifiers."""
    dates: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    names: List[str] = Field(default_factory=list)
    identifiers: List[str] = Field(default_factory=list)

class DocumentPage(BaseModel):
    """Structured content for a single document page."""
    page_number: int
    page_text: str
    tables: List[ExtractedTable] = Field(default_factory=list)
    key_entities: KeyEntities = Field(default_factory=KeyEntities)

class ExtractedDocument(BaseModel):
    """Full extraction output for a whole document."""
    document_id: str
    pages: List[DocumentPage]
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_type: str  # "text_native" or "scanned"