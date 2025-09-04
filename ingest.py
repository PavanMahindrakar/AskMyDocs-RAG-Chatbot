import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # updated import

def ingest_files(data_dir="data", index_dir="index"):
    """
    Ingest PDF and TXT files from the data directory and update the FAISS index.
    If the index already exists, new documents are added instead of overwriting.
    """

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Collect new documents
    documents = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("⚠️ No new documents found in the data folder.")
        return

    # Load existing FAISS index if available
    if os.path.exists(index_dir):
        db = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(documents)   # ✅ appends new docs
    else:
        db = FAISS.from_documents(documents, embedding_model)

    # Save updated index
    db.save_local(index_dir)

    print("✅ Ingestion complete. Index updated successfully.")

