from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from app.pdf_processor import extract_text_from_pdf, chunk_text
from app.embedding import get_embeddings, get_embedding
from app.retriever import FAISSRetriever
from app.llm import generate_answer

app = FastAPI(title="PDF RAG API")

# Initialize retriever
retriever = FAISSRetriever()

class QueryRequest(BaseModel):
    query: str
    api_key: str = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    # Save file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process PDF
        pages_data = extract_text_from_pdf(file_path)
        chunks = chunk_text(pages_data, chunk_size=500, overlap=100)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
            
        # Generate embeddings
        embeddings = get_embeddings(chunks)
        
        # Store in FAISS
        retriever.add_chunks(chunks, embeddings)
        
        return {"message": f"Successfully processed {file.filename} into {len(chunks)} chunks and stored embeddings."}
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    
    # Embed query
    query_emb = get_embedding(query)
    
    # Retrieve chunks
    relevant_chunks = retriever.search(query_emb, top_k=3)
    
    # Generate answer
    answer = generate_answer(query, relevant_chunks, request.api_key)
    
    return {
        "answer": answer,
        "context_sources": relevant_chunks
    }
