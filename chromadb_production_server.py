"""
ChromaDB-Enhanced RAG Server
Production-ready RAG system using ChromaDB for persistent vector storage
"""

import os
import json
import requests
import time
import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from pathlib import Path

# Load environment variables from env.txt or .env
def load_env():
    # Try .env first (standard format)
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        # Clean up any null characters
                        key = key.replace('\0', '').strip()
                        value = value.replace('\0', '').strip()
                        if key and value:
                            os.environ[key] = value
            print(f"[OK] Loaded environment from {env_file}")
            return
        except Exception as e:
            print(f"[WARNING] Error loading .env: {e}")
    
    # Fallback to env.txt
    env_file = Path(__file__).parent / "env.txt"
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        # Clean up any null characters
                        key = key.replace('\0', '').strip()
                        value = value.replace('\0', '').strip()
                        if key and value:
                            os.environ[key] = value
            print(f"[OK] Loaded environment from {env_file}")
        except Exception as e:
            print(f"[WARNING] Error loading env.txt: {e}")
    else:
        print("[WARNING] No environment file found (.env or env.txt)")

load_env()

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Document processing
import PyPDF2
import io

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
    print("[OK] ChromaDB available")
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[ERROR] ChromaDB not available, install with: pip install chromadb")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("[OK] Sentence Transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[ERROR] Sentence Transformers not available")

# OCR imports
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    print("[OK] OCR available")
except ImportError:
    OCR_AVAILABLE = False
    print("[ERROR] OCR not available, install with: pip install pytesseract pillow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chromadb_rag_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class ChromaDBRAGConfig:
    """Configuration for ChromaDB RAG system"""
    similarity_threshold: float = 0.1
    max_chunks_per_query: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "rag_documents"
    enable_hybrid_search: bool = True
    ocr_enabled: bool = OCR_AVAILABLE

config = ChromaDBRAGConfig()

# FastAPI app
app = FastAPI(
    title="ChromaDB RAG Server",
    description="Production-ready RAG with ChromaDB vector storage",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*"  # Fallback for development
    ],
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

class RAGQueryRequest(BaseModel):
    question: str
    similarity_threshold: Optional[float] = None
    max_chunks: Optional[int] = None

class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    sources: List[RetrievedChunk]
    processing_time_ms: float

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

class RAGUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    total_characters: int
    document_id: str
    file_type: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    embedding_service: str
    ocr_available: bool
    chromadb_available: bool
    document_count: int
    collection_name: str
    uptime_seconds: float

# Enhanced Embedding Service
class ChromaEmbeddingService:
    """Embedding service compatible with ChromaDB"""
    
    def __init__(self):
        self.model_name = config.embedding_model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"[OK] Loaded sentence-transformers model: {self.model_name}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load sentence-transformers: {e}")
                self.model = None
        else:
            logger.warning("[WARNING] Sentence Transformers not available")
            self.model = None
    
    def encode(self, text: str) -> List[float]:
        """Encode text into embedding vector"""
        if not text.strip():
            return []
        
        if self.model is not None:
            try:
                return self.model.encode(text, convert_to_numpy=False).tolist()
            except Exception as e:
                logger.error(f"[ERROR] Embedding error: {e}")
                return self._fallback_embedding(text)
        else:
            return self._fallback_embedding(text)
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently"""
        if not texts:
            return []
        
        if self.model is not None:
            try:
                # Process in batches
                all_embeddings = []
                for i in range(0, len(texts), 32):
                    batch = texts[i:i + 32]
                    batch_embeddings = self.model.encode(batch, convert_to_numpy=False)
                    all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                return all_embeddings
            except Exception as e:
                logger.error(f"[ERROR] Batch embedding error: {e}")
                return [self._fallback_embedding(text) for text in texts]
        else:
            return [self._fallback_embedding(text) for text in texts]
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback hash-based embedding"""
        words = text.lower().split()
        vector = []
        
        for i, word in enumerate(words[:384]):
            hash_val = sum(ord(c) for c in word) + i * 31
            normalized = hash_val / 1000.0
            vector.append(normalized)
        
        while len(vector) < 384:
            vector.append(0.0)
        
        return vector[:384]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Fallback dimension

# ChromaDB Vector Database
class ChromaVectorDB:
    """ChromaDB-based vector database with persistent storage"""
    
    def __init__(self):
        self.embedding_service = ChromaEmbeddingService()
        self.client = None
        self.collection = None
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available")
            return
        
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(path=config.chroma_persist_directory)
            logger.info(f"✅ ChromaDB client initialized: {config.chroma_persist_directory}")
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection(name=config.collection_name)
                logger.info(f"✅ Using existing collection: {config.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=config.collection_name,
                    embedding_function=None  # We'll provide our own embeddings
                )
                logger.info(f"✅ Created new collection: {config.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            self.client = None
            self.collection = None
    
    def add_document(self, chunks: List[str], doc_id: str, file_type: str = "text") -> str:
        """Add document chunks to ChromaDB"""
        if not self.collection:
            logger.error("❌ ChromaDB collection not available")
            return doc_id
        
        logger.info(f"📄 Adding document {doc_id} with {len(chunks)} chunks")
        
        try:
            # Generate embeddings for all chunks
            embeddings = self.embedding_service.encode_batch(chunks)
            
            # Prepare data for ChromaDB
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            documents = chunks
            metadatas = [
                {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": file_type,
                    "created_at": datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"✅ Successfully added document {doc_id} to ChromaDB")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Error adding document to ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")
    
    def query_similar(self, query: str, doc_id: str, top_k: int = 5, 
                     similarity_threshold: float = None) -> List[Tuple[int, float]]:
        """Query for similar chunks using ChromaDB"""
        if not self.collection:
            logger.warning("⚠️ ChromaDB collection not available")
            return []
        
        if similarity_threshold is None:
            similarity_threshold = config.similarity_threshold
        
        logger.info(f"🔍 Querying ChromaDB for: '{query}' (threshold: {similarity_threshold})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode(query)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results to filter
                where={"document_id": doc_id},  # Filter by document ID
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            if not results['documents'][0]:
                logger.info("📊 No results found in ChromaDB")
                return []
            
            # Convert ChromaDB results to our format
            processed_results = []
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                
                if similarity >= similarity_threshold:
                    processed_results.append((i, similarity))
            
            # Sort by similarity and return top_k
            processed_results.sort(key=lambda x: x[1], reverse=True)
            top_results = processed_results[:top_k]
            
            logger.info(f"📊 Found {len(top_results)} chunks above threshold {similarity_threshold}")
            return top_results
            
        except Exception as e:
            logger.error(f"❌ ChromaDB query error: {e}")
            return []
    
    def get_document_content(self, doc_id: str) -> List[str]:
        """Get all chunks for a document"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[[0.0] * self.embedding_service.get_embedding_dimension()],  # Dummy embedding
                n_results=1000,  # Get all results
                where={"document_id": doc_id},
                include=["documents"]
            )
            
            return results['documents'][0] if results['documents'][0] else []
            
        except Exception as e:
            logger.error(f"❌ Error getting document content: {e}")
            return []
    
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB"""
        if not self.collection:
            logger.warning("⚠️ ChromaDB collection not available")
            return
        
        try:
            # Get all chunk IDs for the document
            results = self.collection.query(
                query_embeddings=[[0.0] * self.embedding_service.get_embedding_dimension()],  # Dummy embedding
                n_results=1000,
                where={"document_id": doc_id},
                include=["metadatas"]
            )
            
            if results['metadatas'][0]:
                # Extract chunk IDs
                chunk_indices = [meta['chunk_index'] for meta in results['metadatas'][0]]
                ids_to_delete = [f"{doc_id}_chunk_{i}" for i in chunk_indices]
                
                # Delete from ChromaDB
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"🗑️ Deleted document {doc_id} from ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.collection:
            return {
                "type": "chromadb",
                "collection_name": config.collection_name,
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
                "embedding_model": self.embedding_service.model_name,
                "chromadb_available": False
            }
        
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get unique document count
            results = self.collection.query(
                query_embeddings=[[0.0] * self.embedding_service.get_embedding_dimension()],
                n_results=1000,
                include=["metadatas"]
            )
            
            unique_docs = set()
            if results['metadatas'][0]:
                for meta in results['metadatas'][0]:
                    unique_docs.add(meta['document_id'])
            
            return {
                "type": "chromadb",
                "collection_name": config.collection_name,
                "total_documents": len(unique_docs),
                "total_chunks": count,
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
                "embedding_model": self.embedding_service.model_name,
                "chromadb_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                "type": "chromadb",
                "collection_name": config.collection_name,
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
                "embedding_model": self.embedding_service.model_name,
                "chromadb_available": False
            }

# Text processing functions
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF"""
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
                logger.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
                continue
        
        if text.strip():
            logger.info(f"✅ Extracted {len(text)} characters from PDF")
            return text
        else:
            return "No text could be extracted from PDF"
            
    except Exception as e:
        logger.error(f"❌ PDF processing error: {e}")
        return f"Error processing PDF: {str(e)}"

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return "OCR not available. Install with: pip install pytesseract pillow"
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        
        if text.strip():
            logger.info(f"✅ Extracted {len(text)} characters from image")
            return text
        else:
            return "No text found in image"
            
    except Exception as e:
        logger.error(f"❌ OCR error: {e}")
        return f"Error processing image: {str(e)}"

def split_text_into_chunks(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into chunks"""
    if chunk_size is None:
        chunk_size = config.chunk_size
    if overlap is None:
        overlap = config.chunk_overlap
    
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to break at sentence boundary
        chunk = text[start:end]
        last_period = chunk.rfind('. ')
        last_exclamation = chunk.rfind('! ')
        last_question = chunk.rfind('? ')
        
        best_break = max(last_period, last_exclamation, last_question)
        
        if best_break > start + chunk_size // 2:
            end = start + best_break + 2
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return [chunk for chunk in chunks if len(chunk) > 50]

# Smart fallback answer function
def smart_fallback_answer(prompt: str) -> str:
    """Smart fallback that extracts answers from context when no API key is available"""
    try:
        # Extract context and question from prompt
        context_start = prompt.find("CONTEXT:")
        question_start = prompt.find("QUESTION:")
        
        if context_start == -1 or question_start == -1:
            return "The answer is not found in the provided document."
        
        context = prompt[context_start + 8:question_start].strip()
        question = prompt[question_start + 9:].strip()
        
        # Simple keyword-based answer extraction
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Look for question keywords in context
        question_words = set(question_lower.split())
        context_sentences = context.split('.')
        
        best_sentence = ""
        best_score = 0
        
        for sentence in context_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Calculate overlap score
            overlap = len(question_words.intersection(sentence_words))
            score = overlap / len(question_words) if question_words else 0
            
            if score > best_score and score > 0.2:  # At least 20% overlap
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            # Try to extract specific information with better patterns
            if "receipt number" in question_lower or "receipt #" in question_lower:
                # Look for receipt number pattern - more comprehensive
                import re
                patterns = [
                    r'receipt\s*(?:number|#)?\s*[:\-]?\s*([A-Z0-9\-]+)',
                    r'receipt\s*(?:number|#)?\s*[:\-]?\s*([0-9\-]+)',
                    r'([A-Z]{2,4}\d{6,})',  # Common receipt format
                    r'(\d{4}[-\s]?\d{4}[-\s]?\d{4})'  # Number groups
                ]
                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        return f"The receipt number is {match.group(1)}."
            
            if "total" in question_lower or "amount" in question_lower:
                # Look for monetary amounts with better context
                import re
                amount_pattern = r'\$[\d,]+\.?\d*'
                matches = re.findall(amount_pattern, context)
                if matches:
                    # Look for the word "total" near amounts
                    context_parts = context.lower().split('.')
                    for i, part in enumerate(context_parts):
                        if "total" in part and i < len(matches):
                            return f"The total amount is {matches[i]}."
                    # If no explicit total found, return the last amount (usually total)
                    return f"The total amount is {matches[-1]}."
            
            # Return the best sentence as answer
            return best_sentence + "."
        
        # If no good sentence found, try to provide a helpful response
        if any(word in context_lower for word in question_words):
            return "Based on the document, the information is present but a specific answer cannot be extracted."
        
        return "The answer is not found in the provided document."
        
    except Exception as e:
        logger.error(f"❌ Smart fallback error: {e}")
        return "The answer is not found in the provided document."

# Groq API integration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def call_groq_api(prompt: str) -> str:
    """Call Groq API with enhanced error handling"""
    if not GROQ_API_KEY:
        logger.warning("⚠️ Groq API key not configured, using smart fallback")
        return smart_fallback_answer(prompt)
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            logger.info("✅ Groq API call successful")
            return answer
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            logger.error(f"❌ Groq API Error: {response.status_code} - {error_data}")
            return smart_fallback_answer(prompt)
    
    except Exception as e:
        logger.error(f"❌ Groq API Exception: {e}")
        return smart_fallback_answer(prompt)

# Anti-hallucination prompt
STRICT_RAG_PROMPT = """You are a strict AI assistant for document-based question answering.

Instructions:
- Answer ONLY using the provided context.
- Do NOT use any external knowledge.
- Do NOT guess or assume anything.
- If the answer is not clearly present in the context, respond EXACTLY with:
  "The answer is not found in the provided document."
- Keep the answer concise, accurate, and relevant.
- Prefer using exact phrases from the context when possible.

CONTEXT:
{context}

QUESTION:
{question}"""

# Global variables
current_document_id = None
vector_db = ChromaVectorDB()
start_time = time.time()

# API Endpoints
@app.post("/rag/upload", response_model=RAGUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents with ChromaDB storage"""
    global current_document_id
    
    allowed_extensions = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and image files are allowed")
    
    start_time_ms = time.time()
    
    try:
        file_bytes = await file.read()
        file_type = "text"
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
            file_type = "pdf"
        elif any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
            text = extract_text_from_image(file_bytes)
            file_type = "image"
        else:
            text = file_bytes.decode('utf-8')
            file_type = "text"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        # Create chunks
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from file")
        
        # Delete previous document if exists
        if current_document_id:
            vector_db.delete_document(current_document_id)
        
        # Add to ChromaDB
        doc_id = f"doc_{int(time.time())}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
        vector_db.add_document(chunks, doc_id, file_type)
        current_document_id = doc_id
        
        processing_time = (time.time() - start_time_ms) * 1000
        
        logger.info(f"✅ Successfully uploaded {file.filename} ({file_type}) to ChromaDB - {processing_time:.2f}ms")
        
        return RAGUploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            total_characters=len(text),
            document_id=doc_id,
            file_type=file_type,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query RAG system with ChromaDB backend"""
    global current_document_id
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not current_document_id:
        raise HTTPException(
            status_code=400, 
            detail="No document has been uploaded yet. Please upload a document first."
        )
    
    start_time_ms = time.time()
    
    try:
        # Use custom threshold if provided
        threshold = request.similarity_threshold or config.similarity_threshold
        max_chunks = request.max_chunks or config.max_chunks_per_query
        
        # Perform similarity search
        search_results = vector_db.query_similar(
            request.question, 
            current_document_id, 
            top_k=max_chunks,
            similarity_threshold=threshold
        )
        
        if not search_results:
            logger.info("📊 No relevant chunks found above threshold")
            return RAGQueryResponse(
                question=request.question,
                answer="The answer is not found in the provided document.",
                confidence_score=0.0,
                retrieved_chunks=[],
                document_id=current_document_id,
                processing_time_ms=(time.time() - start_time_ms) * 1000,
                fallback_used=True
            )
        
        # Get document content for retrieved chunks
        all_chunks = vector_db.get_document_content(current_document_id)
        
        # Prepare retrieved chunks with metadata
        retrieved_chunks = []
        context_parts = []
        
        for chunk_idx, similarity_score in search_results:
            if chunk_idx < len(all_chunks):
                chunk_content = all_chunks[chunk_idx]
                
                retrieved_chunk = RetrievedChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_idx,
                        document_id=current_document_id,
                        file_type="unknown",
                        similarity_score=similarity_score,
                        source=f"chunk_{chunk_idx}"
                    )
                )
                retrieved_chunks.append(retrieved_chunk)
                context_parts.append(chunk_content)
        
        # Build context
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate prompt
        prompt = STRICT_RAG_PROMPT.format(
            context=context,
            question=request.question
        )
        
        # Get answer from LLM
        answer = call_groq_api(prompt)
        
        # Calculate confidence score based on similarity scores
        confidence_score = sum(score for _, score in search_results) / len(search_results)
        
        processing_time = (time.time() - start_time_ms) * 1000
        
        logger.info(f"🎯 Query processed - Confidence: {confidence_score:.3f}, Time: {processing_time:.2f}ms")
        
        return RAGQueryResponse(
            question=request.question,
            answer=answer.strip(),
            confidence_score=confidence_score,
            retrieved_chunks=retrieved_chunks,
            document_id=current_document_id,
            processing_time_ms=processing_time,
            fallback_used=(answer == "The answer is not found in the provided document.")
        )
    
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    stats = vector_db.get_stats()
    
    return HealthResponse(
        status="healthy",
        embedding_service=stats["embedding_model"],
        ocr_available=OCR_AVAILABLE,
        chromadb_available=CHROMADB_AVAILABLE,
        document_count=stats["total_documents"],
        collection_name=stats["collection_name"],
        uptime_seconds=uptime
    )

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    stats = vector_db.get_stats()
    uptime = time.time() - start_time
    
    return {
        "status": "running",
        "uptime_seconds": uptime,
        "database": stats,
        "embedding": {
            "service": stats["embedding_model"],
            "dimension": stats["embedding_dimension"],
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        },
        "ocr": {
            "available": OCR_AVAILABLE,
            "enabled": config.ocr_enabled
        },
        "config": {
            "similarity_threshold": config.similarity_threshold,
            "max_chunks_per_query": config.max_chunks_per_query,
            "chunk_size": config.chunk_size,
            "collection_name": config.collection_name
        },
        "supported_formats": [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        "groq_api": "configured" if GROQ_API_KEY else "demo_mode"
    }

@app.delete("/rag/document")
async def delete_document():
    """Delete current document"""
    global current_document_id
    
    if not current_document_id:
        raise HTTPException(status_code=400, detail="No document to delete")
    
    vector_db.delete_document(current_document_id)
    deleted_id = current_document_id
    current_document_id = None
    
    logger.info(f"🗑️ Deleted document {deleted_id} from ChromaDB")
    return {"message": f"Document {deleted_id} deleted successfully"}

# Static file serving
@app.get("/style.css")
async def get_style_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_script_js():
    return FileResponse("script.js", media_type="application/javascript")

@app.post("/research", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest):
    """Research endpoint that matches frontend expectations"""
    global current_document_id
    
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    start_time_ms = time.time()
    
    try:
        # If no document is uploaded, use fallback response
        if not current_document_id:
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
        
        # Use the existing RAG system to answer the research query
        rag_request = RAGQueryRequest(
            question=request.topic,
            max_chunks=request.source_count
        )
        
        # Perform the query using existing logic
        threshold = config.similarity_threshold
        max_chunks = request.source_count or config.max_chunks_per_query
        
        # Perform similarity search
        search_results = vector_db.query_similar(
            query_text=request.topic,
            n_results=max_chunks,
            threshold=threshold
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
        
        # Generate response using LLM
        context = "\n\n".join([result.content for result in search_results])
        
        # Adjust prompt based on summary type
        if request.summary_type == "short":
            prompt = f"Based on the following context, provide a brief summary about '{request.topic}':\n\n{context}"
        else:
            prompt = f"Based on the following context, provide a detailed analysis about '{request.topic}':\n\n{context}"
        
        # Generate response
        llm_response = await generate_llm_response(prompt)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(search_results)
        
        # Convert search results to RetrievedChunk format
        sources = []
        for i, result in enumerate(search_results):
            metadata = ChunkMetadata(
                chunk_index=i,
                document_id=current_document_id,
                file_type="unknown",
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
            summary=llm_response,
            confidence_score=confidence_score,
            sources=sources,
            processing_time_ms=processing_time,
            fallback_used=False
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing research request: {e}")
        processing_time = (time.time() - start_time_ms) * 1000
        
        return ResearchResponse(
            topic=request.topic,
            summary=f"An error occurred while researching '{request.topic}': {str(e)}",
            confidence_score=0.0,
            sources=[],
            processing_time_ms=processing_time,
            fallback_used=True
        )

# Graceful shutdown
def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info("🛑 Shutting down ChromaDB RAG server gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 ChromaDB RAG Server Starting...")
    print("=" * 50)
    print(f"📊 Embedding: {vector_db.embedding_service.model_name}")
    print(f"🔍 OCR Available: {OCR_AVAILABLE}")
    print(f"💾 ChromaDB Available: {CHROMADB_AVAILABLE}")
    print(f"📁 Collection: {config.collection_name}")
    print(f"💾 Persist Directory: {config.chroma_persist_directory}")
    print(f"⚙️ Similarity Threshold: {config.similarity_threshold}")
    print(f"🌐 Running on port 8002")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
