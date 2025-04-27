# Chat-with-PDF
# Chat with PDF using RAG, Ollama, and Gradio

Offline Retrieval-Augmented Generation (RAG) system using Local LLMs and Embedding Models  
Built with: **Python**, **Gradio**, **FAISS**, **Ollama**, **Deep Learning Models**

---

## 1. Project Info

This project allows you to chat with your PDF documents by using local language models without any internet connection.

### How it works:

1. **You upload a PDF**
2. **The PDF is split into small text chunks**
3. **Each chunk is embedded locally using `nomic-embed-text:137m-v1.5-fp16` via Ollama**
4. **A FAISS index is built for fast retrieval**
5. **When you ask a question:**
   - The system retrieves the most relevant chunk.
   - The chunk + question are sent to `gemma3:4b` model via Ollama.
   - The model generates a full answer.

### Key Features:
✅ **100% Local, Free, and Private**  
✅ **No external APIs or paid services needed**  
✅ **Light enough for mid-range GPUs (4GB VRAM tested)**  

---

## 2. Pre-requirements

| **Item**           | **Version/Requirement**         |
|---------------------|--------------------------------|
| Python              | >= 3.10                        |
| Ollama              | >= 0.6.6                       |
| VRAM Minimum        | 4GB GPU VRAM    |
| VRAM Recommended    | 8GB GPU VRAM or higher         |
| RAM Recommended     | 16GB RAM or higher             |

---

## 3. Ollama Setup

Ollama is a local LLM runtime. Download and install Ollama from here: [https://ollama.com/download](https://ollama.com/download)

After installation, open CMD and check version:

```bash
ollama --version
```
It should show >= 0.6.6

## 4. Required Models

```bash
ollama pull nomic-embed-text:137m-v1.5-fp16
ollama pull gemma3:4b
```
## 5. Installation and Execution Process
### Clone the Project

```bash
git clone https://github.com/your-username/chat-with-pdf-rag.git
cd chat-with-pdf-rag
```
### Create Python Virtual Environment
```bash
python -m venv venv
```
### Activate Virtual Environment on Windows:
```bash
venv\Scripts\activate
```
### On Linux/macOS:
```bash
source venv/bin/activate
```
### Install Python Requirements
```bash
pip install -r requirements.txt
```
## Start Ollama Server
Make sure your Ollama desktop app is running.
Ollama will run automatically at http://localhost:11434.

### Run the Project
```bash
cd src
python main.py
```
✅ Your browser will automatically open: http://localhost:7860/

## 6. How to Use
- Upload any PDF (small size recommended for testing)
- Click Process PDF
- Type a question related to the document
- Get detailed answers!

## 7. Screenshots
![Uploading PDF](screenshots/Screenshot%201.jpg)

![Chatting with Document](screenshots/Screenshot%202.jpg)



