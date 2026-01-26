import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional, Dict, Any
import random
from datetime import datetime
import re
import PyPDF2
import numpy as np
import io
import math
import json
import uuid
import pickle
from pathlib import Path
from collections import Counter

load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment variables")
    GEMINI_AVAILABLE = False
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        GEMINI_AVAILABLE = True
        print("✅ Gemini API initialized successfully")
    except Exception as e:
        print(f"⚠️ Gemini API initialization failed: {e}")
        GEMINI_AVAILABLE = False

app = FastAPI(title="AI Research Assistant API", description="Powered by Google Gemini API & Simple Vector Store")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple persistent vector store (ChromaDB-like)
class SimpleVectorStore:
    def __init__(self, db_path: str = "./vector_store"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.vectors_file = self.db_path / "vectors.pkl"
        self.metadata_file = self.db_path / "metadata.pkl"
        self.documents_file = self.db_path / "documents.pkl"
        self.vocabulary_file = self.db_path / "vocabulary.pkl"
        
        self.vectors = []
        self.metadata = []
        self.documents = []
        self.vocabulary = []
        
        self.load()
    
    def load(self):
        """Load data from disk"""
        try:
            if self.vectors_file.exists():
                with open(self.vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            if self.documents_file.exists():
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            if self.vocabulary_file.exists():
                with open(self.vocabulary_file, 'rb') as f:
                    self.vocabulary = pickle.load(f)
            print(f"✅ Loaded {len(self.vectors)} vectors from disk")
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
            self.vectors = []
            self.metadata = []
            self.documents = []
            self.vocabulary = []
    
    def save(self):
        """Save data to disk"""
        try:
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.vocabulary_file, 'wb') as f:
                pickle.dump(self.vocabulary, f)
            print(f"✅ Saved {len(self.vectors)} vectors to disk")
        except Exception as e:
            print(f"⚠️ Error saving vector store: {e}")
    
    def add(self, documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: List[str], vocabulary: List[str]):
        """Add documents to the store"""
        # Update vocabulary if new
        if not self.vocabulary:
            self.vocabulary = vocabulary
        elif len(vocabulary) > len(self.vocabulary):
            self.vocabulary = vocabulary
        
        for i, (doc, embedding, metadata, doc_id) in enumerate(zip(documents, embeddings, metadatas, ids)):
            self.documents.append(doc)
            self.vectors.append(embedding)
            self.metadata.append(metadata)
        self.save()
    
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store"""
        if not self.vectors:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Ensure dimensions match
        if len(query_embedding) != len(self.vectors[0]):
            print(f"⚠️ Dimension mismatch: query {len(query_embedding)} vs stored {len(self.vectors[0])}")
            # Pad or truncate query embedding to match stored vectors
            if len(query_embedding) < len(self.vectors[0]):
                query_embedding = np.pad(query_embedding, (0, len(self.vectors[0]) - len(query_embedding)))
            else:
                query_embedding = query_embedding[:len(self.vectors[0])]
        
        # Handle zero vector case
        if np.linalg.norm(query_embedding) == 0:
            print("⚠️ Query embedding is zero vector, using random embedding")
            query_embedding = np.random.rand(len(self.vectors[0]))
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate cosine similarity
        similarities = []
        for vector in self.vectors:
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            vector_norm = vector / np.linalg.norm(vector)
            
            # Cosine similarity
            similarity = np.dot(query_norm, vector_norm)
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results_documents = []
        results_metadata = []
        results_distances = []
        
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return positive similarities
                results_documents.append(self.documents[idx])
                results_metadata.append(self.metadata[idx])
                results_distances.append(1 - similarities[idx])  # Convert to distance
        
        return {
            "documents": [results_documents],
            "metadatas": [results_metadata],
            "distances": [results_distances]
        }
    
    def get_by_metadata(self, where_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Get documents by metadata filter"""
        filtered_indices = []
        for i, metadata in enumerate(self.metadata):
            match = True
            for key, value in where_filter.items():
                if metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_indices.append(i)
        
        filtered_metadata = [self.metadata[i] for i in filtered_indices]
        filtered_ids = [meta.get('chunk_id', f'chunk_{i}') for i, meta in zip(filtered_indices, filtered_metadata)]
        
        return {
            "ids": filtered_ids,
            "metadatas": [filtered_metadata]
        }
    
    def delete(self, ids: List[str] = None):
        """Delete documents by IDs or clear all"""
        if ids is None:
            # Clear all
            self.vectors = []
            self.metadata = []
            self.documents = []
            self.vocabulary = []
        else:
            # Delete specific IDs
            indices_to_keep = []
            for i, metadata in enumerate(self.metadata):
                if metadata.get('chunk_id') not in ids:
                    indices_to_keep.append(i)
            
            self.vectors = [self.vectors[i] for i in indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            self.documents = [self.documents[i] for i in indices_to_keep]
        
        self.save()
    
    def count(self) -> int:
        """Return the number of documents"""
        return len(self.documents)
    
    def get_vocabulary(self) -> List[str]:
        """Get the current vocabulary"""
        return self.vocabulary

# Initialize vector store
vector_store = SimpleVectorStore()

class SummarizeRequest(BaseModel):
    text: str

class ResearchRequest(BaseModel):
    topic: str
    summary_type: str = "detailed"
    source_count: int = 5

class Source(BaseModel):
    title: str
    authors: str
    url: str
    year: int
    relevance_score: float

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[Source]
    keywords: List[str]
    timestamp: str
    processing_time: float

class DocumentMetadata(BaseModel):
    document_name: str
    total_pages: int
    total_chunks: int
    upload_timestamp: str
    file_size: int

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text_preview: str
    word_count: int

class RAGQueryRequest(BaseModel):
    question: str

class RAGQueryResponse(BaseModel):
    answer: str
    matched_chunks: List[str]
    question: str
    metadata: List[ChunkMetadata]

# Mock data for sources
MOCK_SOURCES = [
    {
        "title": "Advances in AI Research: A Comprehensive Review",
        "authors": "Johnson, M., Smith, A., & Williams, R.",
        "url": "https://arxiv.org/abs/2024.12345",
        "year": 2024
    },
    {
        "title": "The Future of Artificial Intelligence: Trends and Implications",
        "authors": "Chen, L., Rodriguez, K., & Thompson, J.",
        "url": "https://nature.com/articles/ai2024",
        "year": 2024
    }
]

def generate_simple_embeddings(texts: List[str], vocabulary: List[str] = None) -> np.ndarray:
    """Generate simple TF-IDF-like embeddings without external dependencies"""
    print("🔄 Generating simple embeddings (no external dependencies)")
    
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
        'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
    }
    
    if vocabulary is None:
        # Create vocabulary from all texts (for new documents)
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            filtered_words = [word for word in words if word not in stop_words]
            all_words.extend(filtered_words)
        
        # Get top 500 most common words as vocabulary
        word_counts = Counter(all_words)
        vocabulary = [word for word, count in word_counts.most_common(500)]
    
    # Create embeddings using the provided vocabulary
    embeddings = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        word_count = Counter(filtered_words)
        
        # Create TF-IDF-like vector
        vector = np.zeros(len(vocabulary))
        total_words = len(filtered_words)
        
        for i, vocab_word in enumerate(vocabulary):
            if vocab_word in word_count:
                # Simple term frequency
                tf = word_count[vocab_word] / total_words if total_words > 0 else 0
                vector[i] = tf
        
        embeddings.append(vector)
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms
    
    print(f"✅ Generated {len(embeddings_array)} embeddings with {len(vocabulary)} dimensions")
    return embeddings_array, vocabulary

def generate_mock_sources(topic: str, count: int) -> List[Source]:
    """Generate mock sources based on the topic"""
    sources = []
    selected_sources = random.sample(MOCK_SOURCES, min(count, len(MOCK_SOURCES)))
    
    for source_data in selected_sources:
        title = source_data["title"].replace("AI", topic.split()[0] if topic.split() else "AI")
        source = Source(
            title=title,
            authors=source_data["authors"],
            url=source_data["url"],
            year=source_data["year"],
            relevance_score=round(0.7 + random.random() * 0.3, 2)
        )
        sources.append(source)
    
    return sources

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
        'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
    }
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word.capitalize() for word, _ in keywords[:max_keywords]]

def extract_text_from_pdf_with_pages(pdf_bytes: bytes) -> tuple[List[str], List[int]]:
    """Extract text from PDF bytes with page numbers"""
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pages_text = []
    page_numbers = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            pages_text.append(text)
            page_numbers.append(page_num)
    
    return pages_text, page_numbers

def split_text_into_meaningful_chunks(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """Split text into meaningful chunks of 200-300 words"""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        sentence_endings = ['. ', '! ', '? ', '\n']
        best_break = -1
        
        for i in range(min(50, len(chunk_words))):
            word_idx = len(chunk_words) - 1 - i
            if word_idx >= 0:
                word = chunk_words[word_idx]
                if any(word.endswith(ending) for ending in sentence_endings):
                    best_break = word_idx + 1
                    break
        
        if best_break > 0:
            chunk_words = chunk_words[:best_break]
            chunk_text = ' '.join(chunk_words)
            start = start + best_break - overlap
        else:
            start = end - overlap
        
        chunks.append(chunk_text.strip())
        
        if start >= len(words):
            break
    
    return [chunk for chunk in chunks if len(chunk.split()) >= 50]

async def generate_with_gemini(prompt: str) -> str:
    """Generate response using Gemini API"""
    if not GEMINI_AVAILABLE:
        return f"This is a mock response because Gemini API is not available. The prompt was: {prompt[:100]}..."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        return f"Error generating response: {str(e)}"

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize text using Gemini AI"""
    if len(req.text) < 100:
        raise HTTPException(status_code=400, detail="Text too short for summarization")

    try:
        prompt = f"""
        Please summarize the following text in a clear and concise manner:
        
        Text: {req.text}
        
        Guidelines:
        - Provide a comprehensive summary
        - Keep it under 150 words
        - Focus on the main points and key information
        - Use clear and accessible language
        
        Summary:
        """
        
        summary = await generate_with_gemini(prompt)
        return {"summary": summary.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed")

@app.post("/research", response_model=ResearchResponse)
async def generate_research_brief(request: ResearchRequest):
    """Generate research brief using Gemini AI"""
    start_time = datetime.now()
    
    try:
        if not request.topic or len(request.topic.strip()) < 10:
            raise HTTPException(status_code=400, detail="Topic must be at least 10 characters long")
        
        if request.summary_type not in ["short", "detailed"]:
            raise HTTPException(status_code=400, detail="Summary type must be 'short' or 'detailed'")
        
        if not 1 <= request.source_count <= 10:
            raise HTTPException(status_code=400, detail="Source count must be between 1 and 10")
        
        if request.summary_type == "short":
            prompt = f"""
            Answer this question accurately and concisely: "{request.topic}"
            
            Guidelines:
            - Provide a direct, accurate answer
            - Be specific and factual
            - Keep it under 150 words
            - Focus on most important information
            - Use clear, accessible language
            
            Question: {request.topic}
            Answer:
            """
        else:
            prompt = f"""
            Provide a comprehensive answer to this question: "{request.topic}"
            
            Structure your response to include:
            1. **Direct Answer**: Clear response to question
            2. **Key Details**: Important supporting information
            3. **Context**: Background or relevant context if needed
            4. **Examples**: Specific examples if applicable
            5. **Additional Insights**: Related information that adds value
            
            Guidelines:
            - Be accurate and factual
            - Provide depth and detail (300-500 words)
            - Use clear headings and structure
            - Be comprehensive but focused
            
            Question: {request.topic}
            Detailed Answer:
            """
        
        summary_text = await generate_with_gemini(prompt)
        summary_text = summary_text.strip()
        
        sources = generate_mock_sources(request.topic, request.source_count)
        keywords = extract_keywords(summary_text)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        response = ResearchResponse(
            topic=request.topic,
            summary=summary_text,
            sources=sources,
            keywords=keywords,
            timestamp=end_time.isoformat(),
            processing_time=round(processing_time, 2)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing research request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/rag/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF for RAG with Simple Vector Store"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)
        
        print(f"🔄 Processing PDF: {file.filename} ({file_size} bytes)")
        
        pages_text, page_numbers = extract_text_from_pdf_with_pages(pdf_bytes)
        
        if not pages_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        print(f"📄 Extracted text from {len(pages_text)} pages")
        
        all_chunks = []
        chunk_metadata = []
        
        for page_idx, (page_text, page_num) in enumerate(zip(pages_text, page_numbers)):
            page_chunks = split_text_into_meaningful_chunks(page_text)
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_id = str(uuid.uuid4())[:8]
                
                metadata = {
                    'chunk_id': chunk_id,
                    'document_name': file.filename,
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'word_count': len(chunk.split())
                }
                
                all_chunks.append(chunk)
                chunk_metadata.append(metadata)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from PDF")
        
        print(f"🔢 Created {len(all_chunks)} chunks")
        
        # Generate simple embeddings with consistent vocabulary
        embeddings, vocabulary = generate_simple_embeddings(all_chunks)
        
        # Clear existing documents for this file
        existing_docs = vector_store.get_by_metadata({"document_name": file.filename})
        if existing_docs['ids']:
            vector_store.delete(ids=existing_docs['ids'])
        
        # Add to vector store with vocabulary
        chunk_ids = [meta['chunk_id'] for meta in chunk_metadata]
        vector_store.add(all_chunks, embeddings, chunk_metadata, chunk_ids, vocabulary)
        
        total_characters = sum(len(chunk) for chunk in all_chunks)
        
        print(f"✅ Successfully processed {file.filename}")
        print(f"📊 Stats: {len(pages_text)} pages, {len(all_chunks)} chunks, {total_characters} characters")
        
        return {
            "message": f"Successfully processed PDF with Simple Vector Store",
            "filename": file.filename,
            "chunks_created": len(all_chunks),
            "total_characters": total_characters,
            "pages_processed": len(pages_text),
            "vector_db_size": vector_store.count(),
            "embedding_dimension": embeddings.shape[1],
            "document_metadata": {
                "document_name": file.filename,
                "total_pages": len(pages_text),
                "total_chunks": len(all_chunks),
                "upload_timestamp": datetime.now().isoformat(),
                "file_size": file_size
            }
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system using Simple Vector Store"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        print(f"🔍 Searching Vector Store for: {request.question}")
        
        # Generate query embedding using stored vocabulary
        stored_vocabulary = vector_store.get_vocabulary()
        query_embedding, _ = generate_simple_embeddings([request.question], stored_vocabulary)
        query_embedding = query_embedding[0]  # Get the single embedding
        
        # Search vector store
        results = vector_store.query(query_embedding, n_results=3)
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        matched_chunks = results['documents'][0]
        matched_metadata = []
        
        for metadata in results['metadatas'][0]:
            chunk_meta = ChunkMetadata(
                chunk_id=metadata['chunk_id'],
                document_name=metadata['document_name'],
                page_number=metadata['page_number'],
                chunk_index=metadata['chunk_index'],
                text_preview=metadata['text_preview'],
                word_count=metadata['word_count']
            )
            matched_metadata.append(chunk_meta)
        
        context = "\n\n---\n\n".join(matched_chunks)
        
        print(f"📝 Found {len(matched_chunks)} relevant chunks")
        
        prompt = f"""
        Answer the following question using ONLY the provided context. 
        If the answer is not present in the context, say "Not found in the document."
        
        Context:
        {context}
        
        Question: {request.question}
        
        Answer:
        """
        
        answer = await generate_with_gemini(prompt)
        
        return RAGQueryResponse(
            answer=answer.strip(),
            matched_chunks=matched_chunks,
            question=request.question,
            metadata=matched_metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_status": "connected" if GEMINI_AVAILABLE else "disconnected",
        "vector_database": {
            "type": "Simple Vector Store",
            "initialized": True,
            "size": vector_store.count(),
            "persistent": True,
            "path": "./vector_store"
        },
        "embedding_model": {
            "model_loaded": True,
            "model_name": "Simple TF-IDF Embeddings (No External Dependencies)"
        },
        "stored_chunks": vector_store.count(),
        "rag_status": "ready" if vector_store.count() > 0 else "no_document"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        if vector_store.count() == 0:
            return {
                "documents": [],
                "total_chunks": 0,
                "vector_db_size": 0
            }
        
        # Group by document name
        doc_groups = {}
        for metadata in vector_store.metadata:
            doc_name = metadata['document_name']
            if doc_name not in doc_groups:
                doc_groups[doc_name] = {
                    "document_name": doc_name,
                    "total_pages": set(),
                    "total_chunks": 0,
                    "upload_timestamp": metadata.get('upload_timestamp', datetime.now().isoformat()),
                    "file_size": metadata.get('file_size', 0)
                }
            doc_groups[doc_name]["total_pages"].add(metadata['page_number'])
            doc_groups[doc_name]["total_chunks"] += 1
        
        documents = []
        for doc_data in doc_groups.values():
            doc_data["total_pages"] = len(doc_data["total_pages"])
            documents.append(doc_data)
        
        return {
            "documents": documents,
            "total_chunks": vector_store.count(),
            "vector_db_size": vector_store.count()
        }
    except Exception as e:
        print(f"Error listing documents: {e}")
        return {
            "documents": [],
            "total_chunks": 0,
            "vector_db_size": 0
        }

@app.delete("/documents")
async def clear_documents():
    """Clear all uploaded documents"""
    try:
        vector_store.delete()
        print("✅ All documents cleared from Vector Store")
        
        return {
            "message": "All documents cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear documents")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Research Assistant API",
        "version": "2.0.0",
        "vector_store": "Simple Vector Store (No External Dependencies)",
        "model": "Google Gemini Pro" if GEMINI_AVAILABLE else "Mock Responses",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting AI Research Assistant API - No External Dependencies")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: {'Google Gemini Pro' if GEMINI_AVAILABLE else 'Mock Responses'}")
    print(f"🗄️ Vector Store: Simple Vector Store (Persistent)")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"⚠️ No PyTorch, no Sentence Transformers - completely dependency free!")
    
    uvicorn.run(
        "main_no_deps:app",
        host=host,
        port=port,
        reload=debug
    )
