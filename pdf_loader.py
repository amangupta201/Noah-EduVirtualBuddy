import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
import hashlib
import os

def load_pdf_text(path: str) -> str:
    """Load text from a single PDF file."""
    try:
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """Chunk text into smaller pieces based on sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_size:
            chunk += " " + sentence
        else:
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = sentence
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def file_hash(path: str) -> str:
    """Generate a hash for a file to ensure uniqueness."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# pdf_loader.py
def process_pdfs(pdf_paths: list) -> dict:
    """Process multiple PDFs and return their text content as a dictionary."""
    pdf_data = {}
    for path in pdf_paths:
        text = load_pdf_text(path)
        pdf_data[path] = text
    return pdf_data
