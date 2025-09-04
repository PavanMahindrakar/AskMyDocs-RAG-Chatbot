from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load your FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("index", embedding_model, allow_dangerous_deserialization=True)

# Get all documents stored in docstore
all_docs = list(db.docstore._dict.values())

print(f"📂 Total documents stored: {len(all_docs)}\n")

# Print a preview of stored docs
for i, doc in enumerate(all_docs[:5]):  # show first 5 docs
    print(f"--- Document {i+1} ---")
    print("Source:", doc.metadata.get("source", "unknown"))
    print("Content preview:", doc.page_content[:200], "...\n")
