# RAG Assistant

A Retrieval-Augmented Generation (RAG) app built with LangChain, ChromaDB, and Groq (Llama 3.3 70B).

## What it does
Upload any PDF or TXT file and ask questions — the AI answers using ONLY your documents.

## Tech Stack
- **LLM**: Llama 3.3 70B via Groq (free)
- **Embeddings**: all-MiniLM-L6-v2 (HuggingFace, free, local)
- **Vector DB**: ChromaDB (local)
- **Orchestration**: LangChain
- **UI**: Streamlit

## How RAG works
1. Your document is split into chunks (~500 chars each)
2. Each chunk is converted to a vector (numbers representing meaning)
3. When you ask a question, the most similar chunks are retrieved
4. The LLM reads those chunks and answers grounded in YOUR documents

## Setup

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant

### 2. Create conda environment
conda create -n rag python=3.11
conda activate rag

### 3. Install dependencies
pip install -r requirements.txt
pip install langchain-groq langchain-text-splitters

### 4. Get your free Groq API key
Sign up at console.groq.com - completely free

### 5. Run the app
streamlit run app.py

## Usage
1. Paste your Groq API key in the sidebar
2. Upload a PDF or TXT file
3. Click Build Knowledge Base
4. Ask questions in the chat!
