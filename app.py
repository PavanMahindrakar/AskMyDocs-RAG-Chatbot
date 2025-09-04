from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Initialize FastAPI app
app = FastAPI()

# Load embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index (ensure you've run ingest.py before starting API)
db = FAISS.load_local("index", embedding_model, allow_dangerous_deserialization=True)

# Load LLM (Ollama must be running locally)
llm = OllamaLLM(model="mistral")  # you can change to llama2, etc.

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Request body model
class QueryRequest(BaseModel):
    query: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "RAG API is running 🚀. Use POST /query to ask questions."}

# Query endpoint
@app.post("/query")
def ask_question(request: QueryRequest):
    result = qa.invoke({"query": request.query})
    return {"answer": result["result"]}
