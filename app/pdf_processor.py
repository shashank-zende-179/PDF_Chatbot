import fitz  # PyMuPDF
from typing import List, Dict

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Extracts text from PDF, keeping track of page numbers."""
    doc = fitz.open(file_path)
    pages_data = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_data.append({"page": i + 1, "text": text})
    return pages_data

def chunk_text(pages_data: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Splits text into chunks, prepending page metadata."""
    chunks = []
    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page"]
        
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i: i + chunk_size]
            chunk_text_str = " ".join(chunk_words)
            chunks.append(f"[Page {page_num}] {chunk_text_str}")
            
    return chunks
