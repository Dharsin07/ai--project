# Vector Database System Documentation

## Overview

This document describes the enhanced vector database system implemented in the AI Research Assistant project. The system provides advanced document processing, embedding generation, and semantic search capabilities using state-of-the-art AI models.

## Architecture

### Core Components

1. **PDF Text Extraction**: Extracts readable text from PDF documents with page number tracking
2. **Intelligent Chunking**: Splits text into meaningful chunks of 200-300 words with sentence boundary detection
3. **Embedding Generation**: Creates vector embeddings using Sentence Transformers
4. **Vector Database**: Stores and indexes embeddings using FAISS for efficient similarity search
5. **Metadata Management**: Tracks document and chunk metadata for enhanced search results

### Technology Stack

- **Sentence Transformers**: `all-MiniLM-L6-v2` model for high-quality embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **PyPDF2**: PDF text extraction
- **FastAPI**: REST API endpoints
- **Google Gemini**: AI-powered question answering

## API Endpoints

### Document Management

#### `POST /rag/upload`
Upload and process a PDF document for vector database storage.

**Request**: Multipart form data with PDF file
**Response**: 
```json
{
  "message": "Successfully processed PDF with vector database",
  "filename": "document.pdf",
  "chunks_created": 45,
  "total_characters": 12500,
  "pages_processed": 10,
  "vector_db_size": 45,
  "embedding_dimension": 384,
  "document_metadata": {
    "document_name": "document.pdf",
    "total_pages": 10,
    "total_chunks": 45,
    "upload_timestamp": "2024-01-25T10:30:00",
    "file_size": 1024000
  }
}
```

#### `GET /documents`
List all uploaded documents and their metadata.

**Response**:
```json
{
  "documents": [...],
  "total_chunks": 45,
  "vector_db_size": 45
}
```

#### `DELETE /documents`
Clear all uploaded documents from the vector database.

**Response**:
```json
{
  "message": "All documents cleared successfully",
  "timestamp": "2024-01-25T10:30:00"
}
```

### Search and Query

#### `POST /vector/search`
Search the vector database directly with semantic similarity.

**Request**:
```json
{
  "query": "What are the main findings?",
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "What are the main findings?",
  "matches": [
    {
      "chunk_text": "The main findings indicate that...",
      "similarity_score": 0.892,
      "chunk_id": "abc12345",
      "document_name": "research.pdf",
      "page_number": 5,
      "chunk_index": 12,
      "rank": 1
    }
  ],
  "answer": "Based on the document, the main findings are..."
}
```

#### `POST /rag/query`
Query the RAG system with context-aware question answering.

**Request**:
```json
{
  "question": "What is the conclusion of this study?"
}
```

**Response**:
```json
{
  "answer": "The study concludes that...",
  "matched_chunks": ["chunk1", "chunk2", "chunk3"],
  "question": "What is the conclusion of this study?",
  "metadata": [
    {
      "chunk_id": "abc12345",
      "document_name": "research.pdf",
      "page_number": 8,
      "chunk_index": 20,
      "text_preview": "The study concludes...",
      "word_count": 245
    }
  ]
}
```

### System Status

#### `GET /health`
Enhanced health check with vector database status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-25T10:30:00",
  "gemini_status": "connected",
  "vector_database": {
    "initialized": true,
    "size": 45,
    "dimension": 384
  },
  "embedding_model": {
    "model_loaded": true,
    "model_name": "all-MiniLM-L6-v2"
  },
  "stored_chunks": 45,
  "documents_loaded": 1,
  "rag_status": "ready"
}
```

## Processing Pipeline

### 1. PDF Text Extraction
```python
def extract_text_from_pdf_with_pages(pdf_bytes: bytes) -> tuple[List[str], List[int]]:
    """
    Extracts text from PDF while preserving page numbers.
    Returns list of text content and corresponding page numbers.
    """
```

### 2. Intelligent Chunking
```python
def split_text_into_meaningful_chunks(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of 200-300 words with intelligent sentence boundary detection.
    Ensures chunks are meaningful and maintain context.
    """
```

### 3. Embedding Generation
```python
def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates vector embeddings for text chunks using Sentence Transformers.
    Returns normalized embeddings for cosine similarity.
    """
```

### 4. Vector Database Creation
```python
def create_vector_database(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Creates FAISS vector database with normalized embeddings.
    Supports efficient similarity search with cosine similarity.
    """
```

### 5. Semantic Search
```python
def search_vector_database(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches vector database for semantically similar chunks.
    Returns ranked results with similarity scores and metadata.
    """
```

## Metadata Structure

### Document Metadata
```python
class DocumentMetadata(BaseModel):
    document_name: str      # Original filename
    total_pages: int        # Number of pages processed
    total_chunks: int       # Number of chunks created
    upload_timestamp: str   # Upload time
    file_size: int          # File size in bytes
```

### Chunk Metadata
```python
class ChunkMetadata(BaseModel):
    chunk_id: str          # Unique identifier
    document_name: str     # Source document
    page_number: int       # Page number
    chunk_index: int       # Index within document
    text_preview: str      # Preview of content
    word_count: int        # Number of words
```

## Usage Examples

### Basic Upload and Search
```python
import requests

# Upload PDF
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/rag/upload', files=files)

# Search vector database
search_data = {
    "query": "machine learning applications",
    "top_k": 3
}
response = requests.post('http://localhost:8000/vector/search', json=search_data)
```

### Advanced RAG Query
```python
# Query with context
query_data = {
    "question": "What are the limitations mentioned in the study?"
}
response = requests.post('http://localhost:8000/rag/query', json=query_data)

# Access metadata
data = response.json()
for metadata in data['metadata']:
    print(f"Found in {metadata['document_name']}, page {metadata['page_number']}")
```

## Performance Characteristics

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Speed**: ~1000 chunks/second (CPU)
- **Quality**: High semantic understanding

### Vector Database
- **Index**: FAISS IndexFlatIP (Inner Product)
- **Similarity**: Cosine similarity (normalized embeddings)
- **Scalability**: Millions of vectors
- **Search Speed**: Sub-millisecond for typical queries

### Chunking Strategy
- **Chunk Size**: 200-300 words (configurable)
- **Overlap**: 50 words (configurable)
- **Boundary Detection**: Sentence endings
- **Minimum Size**: 50 words (filter)

## Testing

### Automated Testing
Run the test script to verify functionality:

```bash
# Test with a PDF file
python test_vector_db.py path/to/your/document.pdf

# Test without PDF (health check only)
python test_vector_db.py
```

### Manual Testing
1. Start the server: `python main.py`
2. Upload a PDF via the web interface or API
3. Test search functionality with various queries
4. Verify metadata and similarity scores

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

### Chunking Parameters
```python
# In main.py, modify these parameters:
CHUNK_SIZE = 250        # Words per chunk
CHUNK_OVERLAP = 50       # Overlap between chunks
MIN_CHUNK_SIZE = 50      # Minimum words to keep a chunk
```

## Troubleshooting

### Common Issues

1. **Embedding Model Loading Error**
   - Ensure sufficient RAM (at least 1GB available)
   - Check internet connection for first-time download

2. **PDF Processing Failure**
   - Verify PDF is text-based (not scanned images)
   - Check file permissions and disk space

3. **Vector Search Returns Empty**
   - Ensure document was uploaded successfully
   - Check vector database status via `/health` endpoint

4. **Memory Issues**
   - Process documents in smaller batches
   - Reduce chunk size for very large documents

### Debug Information
Enable debug mode by setting `DEBUG=true` in environment variables to see detailed processing logs.

## Future Enhancements

### Planned Features
1. **Multiple Document Support**: Upload and search across multiple documents simultaneously
2. **Hybrid Search**: Combine semantic and keyword search
3. **Document Summarization**: Generate document-level summaries
4. **Export Functionality**: Export search results and embeddings
5. **Advanced Filtering**: Filter by document, page range, or metadata

### Performance Optimizations
1. **GPU Acceleration**: CUDA support for embedding generation
2. **Quantization**: Reduce memory usage with quantized embeddings
3. **Caching**: Cache frequent queries and results
4. **Batch Processing**: Process multiple documents in parallel

## Security Considerations

1. **File Upload Validation**: Verify file types and sizes
2. **Memory Management**: Prevent memory exhaustion attacks
3. **API Rate Limiting**: Implement request throttling
4. **Data Privacy**: Clear sensitive data after processing

## Conclusion

The vector database system provides a robust foundation for advanced document analysis and semantic search capabilities. By leveraging state-of-the-art embedding models and efficient vector operations, it enables accurate and fast retrieval of relevant information from large document collections.
