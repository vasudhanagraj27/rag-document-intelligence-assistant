import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3
CHROMA_DIR    = "./chroma_db"

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; }
.source-box { background: #1e1e2e; border-radius: 8px; padding: 8px 12px;
              font-size: 0.78rem; color: #cdd6f4; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

def build_vectorstore(all_docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    import shutil
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR,
    )
    return vectorstore, len(chunks)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain(vectorstore, api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    prompt = PromptTemplate.from_template("""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I don't have enough information in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:""")
    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=api_key)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

with st.sidebar:
    st.markdown("## Setup")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if api_key:
        st.session_state["api_key"] = api_key
    st.divider()
    st.markdown("## Upload Documents")
    uploaded_files = st.file_uploader(
        label="Drop files here", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    ingest_btn = st.button("Build Knowledge Base", disabled=not uploaded_files, use_container_width=True)
    if ingest_btn:
        if not st.session_state.get("api_key"):
            st.error("Enter your Groq API key first.")
        else:
            embeddings = load_embeddings()
            all_docs = []
            progress = st.progress(0, text="Loading files...")
            for i, uf in enumerate(uploaded_files):
                suffix = ".pdf" if uf.name.lower().endswith(".pdf") else ".txt"
                file_type = "pdf" if suffix == ".pdf" else "txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                docs = load_document(tmp_path, file_type)
                all_docs.extend(docs)
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files), text=f"Loaded {uf.name}")
            progress.progress(1.0, text="Building vector index...")
            vectorstore, n_chunks = build_vectorstore(all_docs, embeddings)
            chain, retriever = build_qa_chain(vectorstore, st.session_state["api_key"])
            st.session_state["chain"] = chain
            st.session_state["retriever"] = retriever
            st.session_state["messages"] = []
            progress.empty()
            st.success(f"{len(uploaded_files)} file(s) indexed into {n_chunks} chunks!")
    st.divider()
    if st.session_state.get("chain"):
        st.success("Knowledge base is ready")
    else:
        st.info("Upload docs and click Build to start")
    st.divider()
    st.caption(f"Model: {LLM_MODEL}")
    st.caption(f"Embeddings: {EMBED_MODEL}")
    st.caption(f"Top-K chunks: {TOP_K}")

st.markdown('<p class="main-title">RAG Assistant</p>', unsafe_allow_html=True)
st.caption("Ask anything about your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("View retrieved chunks"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f'<div class="source-box"><b>Chunk {idx}:</b> {src}</div>', unsafe_allow_html=True)

if question := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.get("chain"):
        st.error("Please upload documents and click Build Knowledge Base first.")
        st.stop()
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            source_docs = st.session_state["retriever"].invoke(question)
            answer = st.session_state["chain"].invoke(question)
            sources = [doc.page_content[:300] for doc in source_docs]
        st.write(answer)
        if sources:
            with st.expander("View retrieved chunks"):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f'<div class="source-box"><b>Chunk {idx}:</b> {src}</div>', unsafe_allow_html=True)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
