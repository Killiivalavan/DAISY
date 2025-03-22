"""
Vector store for RAG system.
Manages the vector database for document storage and retrieval.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from src.utils.config import VECTOR_DB_DIR

class VectorStore:
    """Manages the vector database for document storage and retrieval."""
    
    def __init__(self, vector_db_dir: Optional[str] = None):
        """
        Initialize vector store with directory path.
        
        Args:
            vector_db_dir: Path to store vector database files
        """
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        self.index_file = os.path.join(self.vector_db_dir, "faiss_index.bin")
        self.metadata_file = os.path.join(self.vector_db_dir, "metadata.json")
        self.mapping_file = os.path.join(self.vector_db_dir, "id_mapping.pkl")
        
        # Make sure the directory exists
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        # Initialize or load existing index
        self.faiss_index = None
        self.document_metadata = []
        self.id_to_document_map = {}
        
        # Try to load existing index if available
        self.load_index()
    
    def load_index(self) -> bool:
        """
        Load existing index and metadata if available.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if index and metadata files exist
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file) and os.path.exists(self.mapping_file):
                # Load FAISS index
                self.faiss_index = faiss.read_index(self.index_file)
                
                # Load document metadata
                with open(self.metadata_file, 'r') as f:
                    self.document_metadata = json.load(f)
                
                # Load ID to document mapping
                with open(self.mapping_file, 'rb') as f:
                    self.id_to_document_map = pickle.load(f)
                
                print(f"Loaded existing vector database with {len(self.document_metadata)} documents.")
                return True
            return False
        except Exception as e:
            print(f"Error loading vector database: {e}")
            # Reset to empty state
            self.faiss_index = None
            self.document_metadata = []
            self.id_to_document_map = {}
            return False
    
    def save_index(self):
        """Save index and metadata to disk."""
        try:
            if self.faiss_index is not None:
                # Save FAISS index
                faiss.write_index(self.faiss_index, self.index_file)
                
                # Save document metadata
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.document_metadata, f, indent=2)
                
                # Save ID to document mapping
                with open(self.mapping_file, 'wb') as f:
                    pickle.dump(self.id_to_document_map, f)
                
                print(f"Saved vector database with {len(self.document_metadata)} documents.")
        except Exception as e:
            print(f"Error saving vector database: {e}")
    
    def create_index(self, dimension: int):
        """
        Create a new FAISS index with the specified dimension.
        
        Args:
            dimension: Dimension of the embeddings
        """
        # Create a new L2 index
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.document_metadata = []
        self.id_to_document_map = {}
    
    def add_documents(self, documents_with_embeddings: List[Dict[str, Any]]):
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents_with_embeddings: List of document chunks with text, 
                                       metadata, and embeddings
        """
        if not documents_with_embeddings:
            return
        
        # Get the dimension from the first embedding
        first_embedding = documents_with_embeddings[0].get("embedding")
        if first_embedding is None:
            print("No embedding found in the first document.")
            return
        
        # Initialize index if it doesn't exist
        if self.faiss_index is None:
            self.create_index(len(first_embedding))
        
        # Prepare embeddings as numpy array
        embeddings = []
        for i, doc in enumerate(documents_with_embeddings):
            embedding = doc.get("embedding")
            if embedding is not None:
                # Convert to numpy array if needed
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Ensure the embedding is in the correct shape (1D array)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                # Add to embeddings list
                embeddings.append(embedding)
                
                # Create metadata entry without the embedding (to save space)
                doc_without_embedding = {k: v for k, v in doc.items() if k != "embedding"}
                self.document_metadata.append(doc_without_embedding)
                
                # Map document index to metadata index
                doc_id = len(self.id_to_document_map)
                self.id_to_document_map[doc_id] = len(self.document_metadata) - 1
        
        if embeddings:
            # Convert to numpy array and ensure it's the right format
            embeddings_array = np.array(embeddings).astype(np.float32)
            
            # Add to the index
            self.faiss_index.add(embeddings_array)
            
            # Save updated index
            self.save_index()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of similar documents with distances
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Convert to numpy array if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Ensure the embedding is in the correct shape
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Reshape for FAISS if needed
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Perform the search
        distances, indices = self.faiss_index.search(query_embedding, min(k, self.faiss_index.ntotal))
        
        # Extract results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                metadata_idx = self.id_to_document_map.get(int(idx))
                if metadata_idx is not None and metadata_idx < len(self.document_metadata):
                    result = {
                        "distance": float(distances[0][i]),
                        "document": self.document_metadata[metadata_idx]
                    }
                    results.append(result)
        
        return results 