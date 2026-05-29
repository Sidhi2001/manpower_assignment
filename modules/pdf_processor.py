"""
PDF extraction and chunking.

Strategy: per-page text extraction via PyMuPDF, then character-based
sliding-window chunking (800 chars, 100-char overlap) within each page.
Keeping chunks page-local ensures citations are accurate.
"""

import fitz  
from typing import List, Dict


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def process_pdf(pdf_path: str) -> List[Dict]:
    """
    Return a flat list of chunk dicts, each with:
      - text: str
      - page: int (1-indexed)
      - chunk_id: str  (unique, used as ChromaDB document ID)
    """
    doc = fitz.open(pdf_path)
    all_chunks: List[Dict] = []

    for page_idx in range(len(doc)):
        page_text = doc[page_idx].get_text()
        if not page_text.strip():
            continue

        page_num = page_idx + 1
        for i, chunk_text in enumerate(_chunk_text(page_text)):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": f"p{page_num}_c{i}",
                }
            )

    doc.close()
    return all_chunks


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
