# src/app_ui.py

import gradio as gr
import requests
import os
import numpy as np
from utils import extract_text_from_pdf, chunk_text, get_embeddings_from_ollama_batch, create_faiss_index, save_faiss_index, load_faiss_index

VECTORSTORE_PATH = "vectorstore/faiss_index/index.faiss"
CHUNK_SIZE = 300  # Optimal for speed and context

EMBEDDING_MODEL = "nomic-embed-text:137m-v1.5-fp16"
ANSWER_MODEL = "gemma3:4b"

# Global variables
chunks = []
embeddings = None
faiss_index = None

def process_pdf(file):
    global chunks, embeddings, faiss_index

    if not os.path.exists("vectorstore/faiss_index/"):
        os.makedirs("vectorstore/faiss_index/")

    try:
        pdf_path = file.name
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE)
        embeddings = get_embeddings_from_ollama_batch(chunks)
        faiss_index = create_faiss_index(embeddings)
        save_faiss_index(faiss_index, VECTORSTORE_PATH)

        return "✅ PDF processed successfully! You can now ask questions."
    except Exception as e:
        return f"❌ Error processing PDF: {e}"

def query_model(question):
    if faiss_index is None or not chunks:
        return "❌ Please upload and process a PDF first."

    try:
        # Generate embedding for the user's question
        query_embedding = get_embeddings_from_ollama_batch([question])
        D, I = faiss_index.search(query_embedding, k=1)
        retrieved_chunk = chunks[I[0][0]]

        # Limit the size of context to avoid memory overload
        context_text = retrieved_chunk[:2000]  # Limit to 2000 characters

        # Prepare prompt for gemma3:4b
        prompt = f"Context:\n{context_text}\n\nQuestion:\n{question}\nAnswer:"

        # Call Ollama LLM with controlled settings
        payload = {
            "model": ANSWER_MODEL,
            "prompt": prompt,
            "options": {
                "temperature": 0.2,    # More focused answers
                "num_predict": 300     # Limit maximum tokens to generate (faster answers)
            },
            "stream": False
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "⚠️ No answer returned by model.")
        else:
            return f"❌ Error contacting LLM: {response.text}"
    except Exception as e:
        return f"❌ Error answering question: {e}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Chat with PDF using RAG + Ollama")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_button = gr.Button("Process PDF")
    upload_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=10)

    upload_button.click(process_pdf, inputs=[pdf_input], outputs=[upload_output])
    ask_button.click(query_model, inputs=[question_input], outputs=[answer_output])

def launch_app():
    demo.launch(server_name="localhost", server_port=7860, share=False)  # Updated for direct localhost
