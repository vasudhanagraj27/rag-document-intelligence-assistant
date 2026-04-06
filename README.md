# RAG-Based Document Intelligence Assistant

A Retrieval-Augmented Generation (RAG) application for document-based question answering using LangChain, ChromaDB, and Groq (Llama 3.3 70B).

## What it does
Upload PDF or TXT documents and ask questions — the system retrieves relevant context and generates accurate, source-grounded answers strictly from the provided documents.

## Tech Stack
- **LLM**: Llama 3.3 70B via Groq  
- **Embeddings**: all-MiniLM-L6-v2 (HuggingFace, local)  
- **Vector DB**: ChromaDB (local)  
- **Orchestration**: LangChain  
- **UI**: Streamlit  

## How RAG works
1. Documents are split into chunks (~500 characters each)  
2. Each chunk is converted into vector embeddings representing semantic meaning  
3. For each query, the most relevant chunks are retrieved using similarity search  
4. The LLM generates responses grounded strictly in the retrieved context  

## Key Features
- Retrieval-Augmented Generation (RAG) pipeline  
- Semantic search using vector embeddings (ChromaDB)  
- Context-aware LLM responses with reduced hallucination  
- Streamlit-based interactive UI  

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/vasudhanagraj27/rag-document-intelligence-assistant.git
cd rag-document-intelligence-assistant
```

### 2. Create and activate environment
```bash
conda create -n rag python=3.11
conda activate rag
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-groq langchain-text-splitters
```

### 4. Get Groq API key
Sign up at https://console.groq.com and generate a free API key.

### 5. Run the application
```bash
streamlit run app.py
```

## Usage
1. Enter your Groq API key in the sidebar  
2. Upload a PDF or TXT file  
3. Click **Build Knowledge Base**  
4. Ask questions in the chat  

## Why this project
This project demonstrates building an end-to-end LLM application using Retrieval-Augmented Generation (RAG) to enable accurate, context-grounded question answering over custom documents.

## Future Improvements
- Add support for multiple documents and document collections  
- Implement evaluation metrics for response quality  
- Add chat history persistence and session management  
- Deploy on cloud for public access  
