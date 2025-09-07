import streamlit as st
import os
from ingest import ingest_files
from query import get_chatbot 

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="💬 RAG Chat", page_icon="🧠")
st.title("💬 AskMyDocs")

# ------------------- Initialize Chatbot & Memory -------------------
if "qa" not in st.session_state:
    st.session_state.qa = get_chatbot()  # persistent chatbot with memory

# ------------------- Initialize Chat History -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ------------------- Sidebar: File Uploader -------------------
with st.sidebar.expander("📂 Manage Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Drag & drop or browse files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)  # Ensure data folder exists
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        st.success("✅ Files uploaded successfully!")

        # Auto-ingest after upload
        with st.spinner("🔄 Updating FAISS index..."):
            ingest_files()
        st.success("📂 Index updated with new files!")

# ------------------- Chat UI -------------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ------------------- User Input Box -------------------
if query := st.chat_input("Ask a question about your documents..."):
    # Show user query
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa({
    "question": query,
    "chat_history": []   # ✅ pass empty since memory=None
})

            answer = response["answer"]
            st.markdown(answer)

    # Save assistant response
    st.session_state["messages"].append({"role": "assistant", "content": answer})

