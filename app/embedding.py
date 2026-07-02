from sentence_transformers import SentenceTransformer
import numpy as np

# Automatically download and load model natively
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    """Generates an embedding for a piece of text."""
    embedding = model.encode(text)
    return np.array(embedding).astype('float32')

def get_embeddings(texts: list) -> np.ndarray:
    """Generates embeddings for a list of texts."""
    embeddings = model.encode(texts)
    return np.array(embeddings).astype('float32')
