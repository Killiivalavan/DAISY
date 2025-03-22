"""
Embedding generator for RAG system.
Generates embeddings for text chunks using sentence-transformers.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from src.utils.config import EMBEDDING_MODEL

class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence-transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator with a model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding as a numpy array
        """
        try:
            # Generate embedding with the model
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector in case of error
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple text chunks.
        
        Args:
            chunks: List of document chunks with text and metadata
            
        Returns:
            Document chunks with added embeddings
        """
        if not chunks:
            return []
            
        chunked_with_embeddings = []
        total_chunks = len(chunks)
        batch_size = min(32, total_chunks)  # Process in smaller batches
        print(f"Generating embeddings for {total_chunks} chunks in batches of {batch_size}")
        
        # Use batching for better performance
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch = chunks[i:batch_end]
            
            if (i % (batch_size * 2) == 0) or (i + batch_size >= total_chunks):
                print(f"  Processing chunks {i+1}-{batch_end} of {total_chunks} ({(i+1)/total_chunks*100:.1f}%)")
            
            start_time = time.time()
            
            # Extract text from chunks
            texts = [chunk.get("text", "") for chunk in batch]
            
            try:
                # Generate embeddings in batch
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                
                # Add embeddings to chunks
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    new_chunk = {
                        **chunk,
                        "embedding": embedding
                    }
                    chunked_with_embeddings.append(new_chunk)
                
                # Report on first batch time
                if i == 0:
                    batch_time = time.time() - start_time
                    estimated_total = batch_time * (total_chunks / batch_size)
                    print(f"  First batch took {batch_time:.2f}s. Estimated total time: {estimated_total:.2f}s")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add chunks without embeddings in case of error
                for chunk in batch:
                    chunked_with_embeddings.append(chunk)
        
        print(f"Generated embeddings for {len(chunked_with_embeddings)} chunks")
        return chunked_with_embeddings 