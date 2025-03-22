# DAISY RAG Documents Directory

This directory is used to store PDF documents for DAISY's Retrieval-Augmented Generation (RAG) system.

## How It Works

1. Place your PDF files in this directory or any subdirectory.
2. Run `python daisy.py --process-docs` to process and index the documents.
3. The system will:
   - Scan this directory and all subdirectories for PDFs
   - Extract text from each PDF
   - Split the text into manageable chunks
   - Generate embeddings for each chunk
   - Store the embeddings in a vector database

## Usage

After processing documents, DAISY will automatically include relevant information from these documents when answering your questions. You can refer to specific documents in your queries, or DAISY will retrieve relevant documents based on semantic similarity.

## Commands

- **Process Documents**: `python daisy.py --process-docs`
- **Use DAISY with RAG**: `python daisy.py` (RAG is enabled by default)
- **Disable RAG**: `python daisy.py --no-rag`

You can also tell DAISY "process documents" or "index documents" by voice to trigger document processing.

## Notes

- The vector database is stored in the `vector_db` directory at the project root.
- Larger PDFs may take longer to process.
- The system works best with text-based PDFs rather than scanned documents. 