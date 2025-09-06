import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

TRACK_FILE = "ingested_files.json"

def load_ingested_files():
    """Load list of already ingested files from JSON tracker."""
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_ingested_files(files):
    """Save updated list of ingested files."""
    with open(TRACK_FILE, "w") as f:
        json.dump(files, f, indent=4)

def ingest_files(data_dir="data", index_dir="index"):
    """
    Ingest PDF and TXT files from the data directory and update the FAISS index.
    Skips files already ingested using JSON tracker.
    """

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load previously ingested files
    ingested_files = load_ingested_files()

    # Collect new documents
    documents = []
    new_files = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        # Skip already ingested files
        if file_path in ingested_files:
            continue

        # Load PDF or TXT
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            docs = loader.load()
        else:
            continue

        # Add source metadata for each chunk
        for doc in docs:
            doc.metadata["source"] = file_path
            documents.append(doc)

        new_files.append(file_path)

    if not documents:
        print("⚠️ No new documents found in the data folder.")
        return

    # Load existing FAISS index if available
    if os.path.exists(index_dir):
        db = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(documents)   # append only new docs
    else:
        db = FAISS.from_documents(documents, embedding_model)

    # Save updated index
    db.save_local(index_dir)

    # Update tracker
    ingested_files.extend(new_files)
    save_ingested_files(ingested_files)

    print(f"✅ Ingestion complete. Added {len(new_files)} new file(s):")
    for f in new_files:
        print(" -", f)


if __name__ == "__main__":
    ingest_files()
