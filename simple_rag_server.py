"""
Simple RAG Server without emoji characters
Fixed version that handles the /research endpoint
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import re
import base64
import hashlib
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Document processing
import PyPDF2
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env():
    env_file = Path(__file__).parent / "env.txt"
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.replace('\0', '').strip()
                        value = value.replace('\0', '').strip()
                        if key and value:
                            os.environ[key] = value
            print(f"Loaded environment from {env_file}")
            return
        except Exception as e:
            print(f"Error loading env.txt: {e}")

load_env()

# Configuration
@dataclass
class Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    max_chunks_per_query: int = 5
    similarity_threshold: float = 0.3
    collection_name: str = "rag_documents"
    chroma_persist_directory: str = "./chroma_db"

config = Config()

# FastAPI app
app = FastAPI(title="Simple RAG Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Data models
class ChunkMetadata(BaseModel):
    chunk_index: int
    document_id: str
    file_type: str
    similarity_score: Optional[float] = None
    source: str

class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata

class ResearchRequest(BaseModel):
    topic: str
    summary_type: str = "short"
    source_count: int = 5

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    confidence_score: float
    sources: List[RetrievedChunk]
    processing_time_ms: float
    fallback_used: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float

# Global variables
current_document_id = None
document_store = {}
start_time = time.time()

# Simple embedding service
class SimpleEmbeddingService:
    def __init__(self):
        self.model_name = config.embedding_model
    
    def encode(self, text: str) -> List[float]:
        """Simple hash-based embedding fallback"""
        if not text.strip():
            return []
        
        # Create a simple hash-based embedding
        words = text.lower().split()
        vector = []
        
        for i, word in enumerate(words[:384]):
            # Create a hash for each word and convert to a float
            word_hash = hashlib.md5(word.encode()).hexdigest()
            # Convert hex hash to float between -1 and 1
            float_val = int(word_hash[:8], 16) / (2**32 - 1) * 2 - 1
            vector.append(float_val)
        
        # Pad or truncate to 384 dimensions
        while len(vector) < 384:
            vector.append(0.0)
        
        return vector[:384]

# Simple vector database
class SimpleVectorDB:
    def __init__(self):
        self.embedding_service = SimpleEmbeddingService()
        self.documents = {}
        self.embeddings = {}
    
    def add_document(self, doc_id: str, chunks: List[str]):
        """Add document chunks to the database"""
        self.documents[doc_id] = chunks
        self.embeddings[doc_id] = [self.embedding_service.encode(chunk) for chunk in chunks]
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
    
    def query_similar(self, query_text: str, n_results: int = 5, threshold: float = 0.3):
        """Query for similar chunks"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_service.encode(query_text)
        results = []
        
        for doc_id, chunks in self.documents.items():
            if doc_id not in self.embeddings:
                continue
            
            doc_embeddings = self.embeddings[doc_id]
            
            for i, chunk_embedding in enumerate(doc_embeddings):
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= threshold:
                    results.append(type('Result', (), {
                        'content': chunks[i],
                        'similarity_score': similarity,
                        'chunk_index': i,
                        'document_id': doc_id
                    })())
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:n_results]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(y * y for y in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def get_stats(self):
        """Get database statistics"""
        total_chunks = sum(len(chunks) for chunks in self.documents.values())
        return {
            "type": "simple",
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_service.model_name
        }

# Initialize vector database
vector_db = SimpleVectorDB()

# Helper functions
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {e}")
                continue
        
        if text.strip():
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        else:
            return "No text could be extracted from PDF"
            
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return f"Error processing PDF: {str(e)}"

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def smart_fallback_answer(question: str) -> str:
    """Generate a smart fallback answer when no documents are available"""
    return (
        f"I don't have access to any documents to answer your question about '{question}'. "
        f"Please upload a document first, and I'll be happy to help you find information."
    )

# API Endpoints
@app.post("/research", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest):
    """Research endpoint that matches frontend expectations"""
    global current_document_id
    
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    start_time_ms = time.time()
    
    try:
        # If no document is uploaded, use fallback response
        if not current_document_id or current_document_id not in vector_db.documents:
            fallback_response = (
                f"I don't have access to any documents to research '{request.topic}'. "
                f"Please upload a document first, and I'll be happy to help you research this topic."
            )
            processing_time = (time.time() - start_time_ms) * 1000
            
            return ResearchResponse(
                topic=request.topic,
                summary=fallback_response,
                confidence_score=0.0,
                sources=[],
                processing_time_ms=processing_time,
                fallback_used=True
            )
        
        # Perform similarity search
        search_results = vector_db.query_similar(
            query_text=request.topic,
            n_results=request.source_count,
            threshold=config.similarity_threshold
        )
        
        if not search_results:
            fallback_response = (
                f"I couldn't find relevant information about '{request.topic}' "
                f"in the uploaded documents. Please try rephrasing your question or "
                f"uploading relevant documents."
            )
            processing_time = (time.time() - start_time_ms) * 1000
            
            return ResearchResponse(
                topic=request.topic,
                summary=fallback_response,
                confidence_score=0.0,
                sources=[],
                processing_time_ms=processing_time,
                fallback_used=True
            )
        
        # Generate a simple summary based on the found chunks
        context = "\n\n".join([result.content for result in search_results])
        
        # Simple summary generation
        if request.summary_type == "short":
            summary = f"Based on the uploaded documents, here's a brief summary about '{request.topic}':\n\n{context[:500]}..."
        else:
            summary = f"Based on the uploaded documents, here's a detailed analysis about '{request.topic}':\n\n{context}"
        
        # Calculate confidence score
        confidence_score = sum(result.similarity_score for result in search_results) / len(search_results)
        
        # Convert search results to RetrievedChunk format
        sources = []
        for i, result in enumerate(search_results):
            metadata = ChunkMetadata(
                chunk_index=result.chunk_index,
                document_id=result.document_id,
                file_type="pdf",
                similarity_score=result.similarity_score,
                source=f"Document chunk {i+1}"
            )
            sources.append(RetrievedChunk(
                content=result.content,
                metadata=metadata
            ))
        
        processing_time = (time.time() - start_time_ms) * 1000
        
        return ResearchResponse(
            topic=request.topic,
            summary=summary,
            confidence_score=confidence_score,
            sources=sources,
            processing_time_ms=processing_time,
            fallback_used=False
        )
        
    except Exception as e:
        logger.error(f"Error processing research request: {e}")
        processing_time = (time.time() - start_time_ms) * 1000
        
        return ResearchResponse(
            topic=request.topic,
            summary=f"An error occurred while researching '{request.topic}': {str(e)}",
            confidence_score=0.0,
            sources=[],
            processing_time_ms=processing_time,
            fallback_used=True
        )

@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    global current_document_id
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        # Split into chunks
        chunks = split_text_into_chunks(text)
        
        # Generate document ID
        doc_id = f"{file.filename}_{int(time.time())}"
        
        # Add to vector database
        vector_db.add_document(doc_id, chunks)
        
        # Set as current document
        current_document_id = doc_id
        
        processing_time = 0.0  # Simple implementation
        
        return {
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_characters": len(text),
            "document_id": doc_id,
            "file_type": "pdf" if file.filename.lower().endswith('.pdf') else "text",
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    stats = vector_db.get_stats()
    return {
        "status": "running",
        "current_document_id": current_document_id,
        "vector_db": stats,
        "config": {
            "embedding_model": config.embedding_model,
            "max_chunks_per_query": config.max_chunks_per_query,
            "similarity_threshold": config.similarity_threshold
        }
    }

# Static file serving
@app.get("/style.css")
async def get_style_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_script_js():
    return FileResponse("script.js", media_type="application/javascript")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    
    print("Simple RAG Server Starting...")
    print("=" * 50)
    print(f"Embedding: {vector_db.embedding_service.model_name}")
    print(f"Collection: {config.collection_name}")
    print(f"Running on port 8002")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
