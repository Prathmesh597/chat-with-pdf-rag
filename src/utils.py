# src/utils.py

import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import requests

EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"  # Correct full model name

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and returns clean text from a given PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return full_text

def chunk_text(text: str, chunk_size: int = 300) -> list:
    """Splits text into smaller chunks of approximately chunk_size words."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# src/utils.py (fixed)

def get_embeddings_from_ollama_batch(text_list: list) -> np.ndarray:
    """Generates embeddings one by one (fast loop) for multiple texts using Ollama embedding model."""
    embeddings = []
    for text in text_list:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text:137m-v1.5-fp16",
                "prompt": text
            }
        )
        if response.status_code == 200:
            emb = response.json()["embedding"]
            embeddings.append(emb)
        else:
            raise Exception(f"Embedding API error: {response.text}")
    return np.array(embeddings)

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Creates a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, save_path: str):
    """Saves FAISS index to disk."""
    faiss.write_index(index, save_path)

def load_faiss_index(load_path: str) -> faiss.IndexFlatL2:
    """Loads FAISS index from disk."""
    return faiss.read_index(load_path)
