from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_chatbot(index_dir="index"):
    """
    Load FAISS index + Ollama model + memory for conversational RAG
    """

    # Load embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # top 2 docs

    # Ollama model
    llm = OllamaLLM(model="mistral")

    # Memory (stores chat history)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    return qa
