import os
from groq import Groq

def generate_answer(query: str, context_chunks: list, api_key: str = None) -> str:
    """Generates an answer based ONLY on the provided context."""
    if not context_chunks:
        return "Please upload a document first so I have context to answer from."
        
    context = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
You are a helpful answering assistant. You must answer the user's question strictly based on the context provided below.
If the answer is not found in the context, say exactly: "Answer not found in document".
Do not hallucinate or use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""
    
    key_to_use = api_key or os.getenv("GROQ_API_KEY")
    if not key_to_use:
        return "Error: Groq API Key is missing. Please enter it in the Streamlit UI sidebar."
        
    try:
        client = Groq(api_key=key_to_use)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a precise assistant that answers only from the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"
