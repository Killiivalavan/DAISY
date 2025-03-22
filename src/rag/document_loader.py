"""
Document loader for RAG system.
Scans directories for PDF files and loads them.
"""
import os
from typing import List, Dict, Optional
from src.utils.config import DOCUMENTS_DIR

class DocumentLoader:
    """Scans directories for PDF files and loads them."""
    
    def __init__(self, documents_dir: Optional[str] = None):
        """Initialize document loader with directory path."""
        self.documents_dir = documents_dir or DOCUMENTS_DIR
        
    def scan_directory(self, include_subdirs: bool = True) -> List[str]:
        """
        Scan directory for PDF files.
        
        Args:
            include_subdirs: Whether to scan subdirectories as well
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        # Make sure the directory exists
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir, exist_ok=True)
            return pdf_files
        
        # Walk through the directory and subdirectories if needed
        if include_subdirs:
            for root, _, files in os.walk(self.documents_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            # Only scan the main directory
            for file in os.listdir(self.documents_dir):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(self.documents_dir, file))
        
        return pdf_files
    
    def get_document_metadata(self, file_paths: List[str]) -> List[Dict]:
        """
        Get metadata for each document.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of document metadata dictionaries
        """
        documents = []
        
        for file_path in file_paths:
            try:
                relative_path = os.path.relpath(file_path, self.documents_dir)
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                doc_metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "relative_path": relative_path,
                    "file_size": file_size,
                    "file_type": "pdf"
                }
                
                documents.append(doc_metadata)
            except Exception as e:
                print(f"Error processing document {file_path}: {e}")
        
        return documents 