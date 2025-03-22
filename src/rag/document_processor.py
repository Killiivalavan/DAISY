"""
Document processor for RAG system.
Extracts text from PDF files and splits into chunks.
"""
from typing import List, Dict, Any, Optional
import PyPDF2
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    """Processes PDF documents by extracting text and splitting into chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """Initialize document processor with chunk settings."""
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        start_time = time.time()
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                print(f"    Extracting text from {total_pages} pages...")
                
                for page_num in range(total_pages):
                    if page_num % 10 == 0 and page_num > 0:
                        print(f"    Processed {page_num}/{total_pages} pages...")
                    
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
                
                extraction_time = time.time() - start_time
                print(f"    Extracted {len(text)} characters in {extraction_time:.2f}s")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            metadata: Document metadata to include with each chunk
            
        Returns:
            List of document chunks with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        try:
            # Split the text into chunks
            print(f"    Splitting text into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
            text_chunks = self.text_splitter.split_text(text)
            
            # Create document chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "chunk_count": len(text_chunks)
                    }
                }
                chunks.append(chunk)
            
            print(f"    Created {len(chunks)} chunks from document")
            return chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            return []
    
    def process_document(self, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document by extracting text and splitting into chunks.
        
        Args:
            doc_metadata: Document metadata dictionary with file_path
            
        Returns:
            List of document chunks with text and metadata
        """
        file_path = doc_metadata.get("file_path")
        if not file_path:
            return []
        
        # Extract text from the PDF
        text = self.extract_text_from_pdf(file_path)
        
        if not text:
            print(f"    No text was extracted from {file_path}")
            return []
            
        # Split text into chunks with metadata
        document_chunks = self.split_text(text, doc_metadata)
        
        return document_chunks 