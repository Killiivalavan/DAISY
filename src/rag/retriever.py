"""
Retriever for RAG system.
Retrieves relevant documents based on query embeddings.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from src.rag.embedding_generator import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.document_tracker import DocumentTracker
from src.utils.config import MAX_DOCS_TO_RETRIEVE, DOCUMENT_TRACKING_FILE

class Retriever:
    """Retrieves relevant documents based on user queries."""
    
    def __init__(self, 
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 vector_store: Optional[VectorStore] = None,
                 document_tracker: Optional[DocumentTracker] = None,
                 max_docs: Optional[int] = None):
        """
        Initialize retriever with embedding generator and vector store.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            document_tracker: DocumentTracker instance
            max_docs: Maximum number of documents to retrieve
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.document_tracker = document_tracker or DocumentTracker(DOCUMENT_TRACKING_FILE)
        self.max_docs = max_docs or MAX_DOCS_TO_RETRIEVE
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: User query text
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search for similar documents
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=self.max_docs
        )
        
        return search_results
    
    def get_formatted_context(self, query: str) -> str:
        """
        Get formatted context from retrieved documents.
        
        Args:
            query: User query
            
        Returns:
            Formatted context string for the LLM
        """
        results = self.retrieve(query)
        
        if not results:
            return "No relevant documents found."
        
        # Sort by distance (smaller is better)
        sorted_results = sorted(results, key=lambda x: x.get("distance", float("inf")))
        
        # Format the context
        context_parts = []
        
        for i, result in enumerate(sorted_results):
            doc = result.get("document", {})
            text = doc.get("text", "")
            
            # Get metadata for citation
            metadata = doc.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown Document")
            
            # Add formatted document chunk
            context_parts.append(f"[Document {i+1} - {file_name}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def get_rag_prompt(self, query: str, system_prompt: str) -> List[Dict[str, str]]:
        """
        Get RAG-enhanced prompt for the LLM.
        
        Args:
            query: User query
            system_prompt: Original system prompt
            
        Returns:
            Message list with RAG context
        """
        context = self.get_formatted_context(query)
        
        # Create enhanced system prompt with context
        rag_system_prompt = f"{system_prompt}\n\n" \
                            f"Today's date is: {self.get_current_date()}\n" \
                            f"You have access to the following document excerpts to help answer the user's question.\n" \
                            f"---\n{context}\n---\n" \
                            f"Always cite your sources by referencing the document number when providing information from these documents.\n" \
                            f"If the documents don't contain relevant information, just say so and answer based on your knowledge.\n"
        
        # Create messages for the LLM
        messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": query}
        ]
        
        return messages
    
    def get_current_date(self) -> str:
        """Get current date formatted as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def process_documents(self, force_reprocess=False):
        """
        Process and index documents for the RAG system.
        
        Args:
            force_reprocess: If True, reprocess all documents regardless of tracking status
            
        Returns:
            Number of document chunks processed
        """
        from src.rag.document_loader import DocumentLoader
        from src.rag.document_processor import DocumentProcessor
        
        print("Starting document processing...")
        print("Step 1/6: Loading sentence-transformers model (may take a minute on first run)...")
        
        # Ensure the embedding model is loaded (this can take time on first run)
        _ = self.embedding_generator.model
        print("Model loaded successfully.")
        
        # Load documents
        print("Step 2/6: Scanning for PDF documents...")
        loader = DocumentLoader()
        pdf_files = loader.scan_directory(include_subdirs=True)
        
        if not pdf_files:
            print("No PDF files found in the documents directory.")
            return 0
            
        print(f"Found {len(pdf_files)} PDF files.")
        
        # Filter out already processed documents if not forcing reprocessing
        if not force_reprocess:
            print("Step 3/6: Checking for new or modified documents...")
            unprocessed_files = self.document_tracker.get_unprocessed_documents(pdf_files)
            
            if not unprocessed_files:
                print("All documents are already processed and up-to-date.")
                return 0
                
            print(f"Found {len(unprocessed_files)} new or modified documents to process.")
            pdf_files = unprocessed_files
        else:
            print("Step 3/6: Processing all documents (force reprocess enabled)...")
            
        # Get metadata for documents to process
        documents = loader.get_document_metadata(pdf_files)
        
        # Process documents
        print(f"Step 4/6: Extracting text from {len(documents)} documents...")
        processor = DocumentProcessor()
        all_chunks = []
        processed_docs = {}  # Track processed documents and their chunk counts
        
        for i, doc in enumerate(documents):
            file_path = doc.get('file_path')
            file_name = doc.get('file_name', f'Document {i+1}')
            print(f"  Processing {file_name} ({i+1}/{len(documents)})...")
            
            chunks = processor.process_document(doc)
            chunk_count = len(chunks)
            
            if chunk_count > 0:
                print(f"  Extracted {chunk_count} text chunks from {file_name}")
                all_chunks.extend(chunks)
                processed_docs[file_path] = chunk_count
            else:
                print(f"  No text chunks were extracted from {file_name}")
        
        if not all_chunks:
            print("No text chunks were extracted from the documents.")
            return 0
            
        # Generate embeddings
        print(f"Step 5/6: Generating embeddings for {len(all_chunks)} text chunks...")
        print("  This may take a few minutes for large documents...")
        chunks_with_embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Add to vector store
        print("Step 6/6: Storing embeddings in vector database...")
        self.vector_store.add_documents(chunks_with_embeddings)
        
        # Mark documents as processed and save tracking data
        for file_path, chunk_count in processed_docs.items():
            self.document_tracker.mark_document_processed(file_path, chunk_count)
        
        self.document_tracker.save_tracking_data()
        
        print(f"Successfully processed and indexed {len(all_chunks)} document chunks from {len(processed_docs)} documents.")
        
        # Report overall stats
        stats = self.document_tracker.get_stats()
        print(f"Total documents in tracking system: {stats['total_processed']}")
        print(f"Total document chunks in tracking system: {stats['total_chunks']}")
        
        return len(all_chunks) 