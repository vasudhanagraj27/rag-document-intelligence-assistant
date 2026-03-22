import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
CHROMA_DIR    = "./chroma_db"

def ingest(file_paths):
    print(f"\nLoading {len(file_paths)} file(s)...")
    all_docs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  Skipping - file not found: {path}")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        else:
            print(f"  Skipping unsupported type: {path}")
            continue
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  Loaded: {path} ({len(docs)} page(s))")
    if not all_docs:
        print("\nNo documents loaded. Exiting.")
        return
    print(f"\nSplitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  {len(chunks)} chunks created")
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    print(f"\nStoring in ChromaDB at {CHROMA_DIR}...")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"\nDone! {len(chunks)} chunks stored.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()
    ingest(args.files)
