import streamlit as st
import requests
import os

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

API_URL = "http://localhost:8000"

st.title("📄 PDF-based RAG Chatbot")
st.markdown("Upload a PDF and ask questions! Answers are based **strictly** on the document content.")

# Sidebar for PDF upload and API Key setup
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key (Free)", type="password", help="Get your free key at https://console.groq.com")
    
    st.divider()
    
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if st.button("Process PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF (extracting text, chunking, generating embeddings locally)...\nThis may take a minute if downloading standard models for the first time."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}. Is FastAPI running on port 8000?")
        else:
            st.warning("Please upload a PDF first.")

# Chat interface
st.header("💬 Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Source Snippets"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**\n{source}")

# User input handling
if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                req_data = {"query": prompt, "api_key": api_key if api_key else ""}
                response = requests.post(f"{API_URL}/ask", json=req_data)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("context_sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Source Snippets"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**\n{source}")
                                
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("Error getting answer from API.")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}. Is FastAPI running?")
