import faiss
import numpy as np
import pickle
import os

class FAISSRetriever:
    def __init__(self, embedding_dim: int = 384, index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl"):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.embedding_dim = embedding_dim
        
        # Load index and chunks if they exist, to append on existing data
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunks = []
            
    def add_chunks(self, chunks: list, embeddings: np.ndarray):
        """Adds text chunks and their embeddings to the FAISS index."""
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        # Save back to disk to persist memory
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
            
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        """Searches for the most similar chunks to the query."""
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                results.append(self.chunks[i])
        return results
