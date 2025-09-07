# query.py
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def get_chatbot(index_dir="index"):
    """
    Load FAISS index + Ollama model + memory for conversational RAG.
    """

    # Load embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 chunks

    # Ollama LLM
    llm = OllamaLLM(model="mistral")

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
	output_key="answer"  # <-- store only the 'answer' text
    )

    # Optional: custom prompt template to ground answers in retrieved documents
    prompt_template = """
Use the following document chunks to answer the question as accurately as possible.

If the answer is not in the documents, reply with:
"I could not find relevant information in the provided documents."

Document Chunks:
{context}

Question: {question}
Answer:
"""



    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        return_source_documents=True,  # ✅ must be True
        combine_docs_chain_kwargs={"prompt": PROMPT}  # use prompt to force grounding
    )

    return qa
