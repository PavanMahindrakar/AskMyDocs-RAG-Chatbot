# RAG Pipeline with FAISS + Streamlit (Local LLM via Ollama)

A **Retrieval-Augmented Generation (RAG)** system using FAISS for vector storage, Streamlit for a ChatGPT-style interface, and a local LLM via Ollama (Mistral) for private, on-device inference.

---

## Tech Stack
- **LLM (generation):** Mistral via Ollama  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)  
- **Vector DB:** FAISS (persisted locally)  
- **Frameworks:** LangChain (community + HuggingFace + Ollama integrations)  
- **UI:** Streamlit (chat interface + sidebar uploads)  
- **Loaders:** PyPDFLoader (PDF), TextLoader (TXT)  
- **Memory:** ConversationBufferMemory (session-level chat memory)  

---

## Features
- Upload PDFs/TXTs → automatic embeddings → store in FAISS  
- Chat with your documents using a ChatGPT-style interface  
- Incremental knowledge growth; new uploads are added without overwriting  
- Local inference ensures privacy and no cloud dependency  
- Session memory enables coherent follow-up conversations  

---

## Folder Structure

rag_project/
├─ data/ # Uploaded PDFs/TXTs
├─ index/ # FAISS vector DB (persistent)
├─ ingest.py # Ingestion pipeline (embed + store docs)
├─ query.py # Retrieval + generation + memory
├─ ui.py # Streamlit chat app
├─ inspect_faiss.py # Inspect FAISS index (optional)
├─ requirements.txt # Dependencies
└─ venv/ # Python virtual environment


---

## Getting Started

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Install Ollama

curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull mistral     # or: ollama pull gemma:2b


# Run the App

streamlit run ui.py

# Usage

    Open the Streamlit app URL

    Use “Manage Documents” in the sidebar to upload PDFs/TXTs

    Ask questions in the chat; answers are grounded in uploaded documents

    Knowledge base grows incrementally as new documents are added

# Implementation Details

    Ingestion: PyPDFLoader & TextLoader → embeddings → FAISS

    Retrieval + Generation: FAISS retriever → LangChain prompt → Ollama LLM

    Memory: ConversationBufferMemory stores chat history for context

# Achievements

    Fully functional RAG pipeline with FAISS

    ChatGPT-style UI for Q&A over documents

    Incremental knowledge base growth

    Fully local inference (privacy-friendly)

    Optional FAISS inspection utility
