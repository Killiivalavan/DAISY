"""
Document tracker for RAG system.
Tracks which documents have been processed to avoid reprocessing.
"""
import os
import json
import time
from typing import Dict, List, Optional, Set

class DocumentTracker:
    """Tracks which documents have been processed to avoid duplicates."""
    
    def __init__(self, tracking_file: str):
        """
        Initialize document tracker with tracking file path.
        
        Args:
            tracking_file: Path to store document tracking data
        """
        self.tracking_file = tracking_file
        self.processed_docs = {}
        self.load_tracking_data()
        
    def load_tracking_data(self) -> None:
        """Load tracking data from disk if available."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    self.processed_docs = json.load(f)
                print(f"Loaded tracking data for {len(self.processed_docs)} documents.")
            except Exception as e:
                print(f"Error loading tracking data: {e}")
                self.processed_docs = {}
    
    def save_tracking_data(self) -> None:
        """Save tracking data to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            
            with open(self.tracking_file, 'w') as f:
                json.dump(self.processed_docs, f, indent=2)
            print(f"Saved tracking data for {len(self.processed_docs)} documents.")
        except Exception as e:
            print(f"Error saving tracking data: {e}")
    
    def is_document_processed(self, file_path: str) -> bool:
        """
        Check if document has been processed and not modified since.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document has been processed and not modified, False otherwise
        """
        if file_path not in self.processed_docs:
            return False
            
        try:
            # Get last modification time of the file
            last_modified = os.path.getmtime(file_path)
            last_processed = self.processed_docs[file_path].get("last_processed", 0)
            
            # If file has been modified since last processing, it needs reprocessing
            return last_modified <= last_processed
        except Exception:
            # If we can't get file info, assume it needs processing
            return False
    
    def mark_document_processed(self, file_path: str, chunk_count: int) -> None:
        """
        Mark document as processed with current timestamp.
        
        Args:
            file_path: Path to the document file
            chunk_count: Number of chunks extracted from document
        """
        self.processed_docs[file_path] = {
            "last_processed": time.time(),
            "chunk_count": chunk_count,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path)
        }
        
    def get_unprocessed_documents(self, file_paths: List[str]) -> List[str]:
        """
        Filter list of file paths to only include unprocessed documents.
        
        Args:
            file_paths: List of document file paths to check
            
        Returns:
            List of file paths that need processing
        """
        return [path for path in file_paths if not self.is_document_processed(path)]
        
    def get_stats(self) -> Dict:
        """
        Get stats about tracked documents.
        
        Returns:
            Dictionary with stats
        """
        if not self.processed_docs:
            return {"total_processed": 0, "total_chunks": 0}
            
        total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_docs.values())
        return {
            "total_processed": len(self.processed_docs),
            "total_chunks": total_chunks
        } 