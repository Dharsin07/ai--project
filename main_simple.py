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
from collections import Counter

load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment variables")
    print("Please set up your .env file with a valid Gemini API key")
    # Don't raise error for demo purposes

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_AVAILABLE = True
    print("✅ Gemini API initialized successfully")
except Exception as e:
    print(f"⚠️ Gemini API initialization failed: {e}")
    GEMINI_AVAILABLE = False

app = FastAPI(title="AI Research Assistant API", description="Simplified Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components (simplified)
stored_chunks = []
document_metadata = []
chunk_metadata = []

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
    },
    {
        "title": "AI Applications in Modern Society",
        "authors": "Anderson, S., Kumar, P., & Martinez, D.",
        "url": "https://science.org/article/2024/ai-society",
        "year": 2023
    },
    {
        "title": "Ethical Considerations in AI Development",
        "authors": "Taylor, E., Brown, M., & Davis, L.",
        "url": "https://ieee.org/ai-ethics-2024",
        "year": 2024
    },
    {
        "title": "Methodological Approaches to AI Research",
        "authors": "Wilson, R., Garcia, A., & Lee, H.",
        "url": "https://acm.org/ai-methods-2024",
        "year": 2023
    }
]

def generate_mock_sources(topic: str, count: int) -> List[Source]:
    """Generate mock sources based on the topic"""
    sources = []
    selected_sources = random.sample(MOCK_SOURCES, min(count, len(MOCK_SOURCES)))
    
    for i, source_data in enumerate(selected_sources):
        # Customize title based on topic
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

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def split_text_into_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence or word boundary
        chunk = text[start:end]
        last_period = chunk.rfind('. ')
        last_space = chunk.rfind(' ')
        
        if last_period > end - 100:
            end = start + last_period + 2
        elif last_space > end - 50:
            end = start + last_space
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def simple_text_similarity(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple text similarity using keyword matching"""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        score = intersection / union if union > 0 else 0
        
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

async def generate_with_gemini(prompt: str) -> str:
    """Generate response using Gemini API"""
    if not GEMINI_AVAILABLE:
        # Fallback mock response
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
    """Upload and process PDF for RAG (simplified version)"""
    global stored_chunks, document_metadata, chunk_metadata
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)
        
        print(f"🔄 Processing PDF: {file.filename} ({file_size} bytes)")
        
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from PDF")
        
        # Create metadata for chunks
        new_chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                chunk_id=f"chunk_{i}_{hash(chunk) % 10000}",
                document_name=file.filename,
                page_number=1,  # Simplified - not tracking individual pages
                chunk_index=i,
                text_preview=chunk[:100] + "..." if len(chunk) > 100 else chunk,
                word_count=len(chunk.split())
            )
            new_chunk_metadata.append(metadata)
        
        stored_chunks = chunks
        chunk_metadata = new_chunk_metadata
        
        # Create document metadata
        doc_metadata = DocumentMetadata(
            document_name=file.filename,
            total_pages=1,  # Simplified
            total_chunks=len(chunks),
            upload_timestamp=datetime.now().isoformat(),
            file_size=file_size
        )
        document_metadata = [doc_metadata]
        
        total_characters = sum(len(chunk) for chunk in chunks)
        
        print(f"✅ Successfully processed {file.filename}")
        print(f"📊 Stats: {len(chunks)} chunks, {total_characters} characters")
        
        return {
            "message": f"Successfully processed PDF (simplified version)",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_characters": total_characters,
            "pages_processed": 1,  # Simplified
            "vector_db_size": len(chunks),  # Mock vector DB size
            "embedding_dimension": 384,  # Mock dimension
            "document_metadata": doc_metadata.dict()
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system (simplified version)"""
    global stored_chunks, chunk_metadata
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not stored_chunks:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet")
    
    try:
        print(f"🔍 Searching for: {request.question}")
        
        matched_chunks = simple_text_similarity(request.question, stored_chunks, top_k=3)
        
        if not matched_chunks:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        context = "\n\n---\n\n".join(matched_chunks)
        
        prompt = f"""
        Answer the following question using ONLY the provided context. 
        If the answer is not present in the context, say "Not found in the document."
        
        Context:
        {context}
        
        Question: {request.question}
        
        Answer:
        """
        
        answer = await generate_with_gemini(prompt)
        
        # Get metadata for matched chunks
        matched_metadata = []
        for chunk in matched_chunks:
            # Find corresponding metadata
            for meta in chunk_metadata:
                if meta.text_preview in chunk or chunk[:50] in meta.text_preview:
                    matched_metadata.append(meta)
                    break
        
        return RAGQueryResponse(
            answer=answer.strip(),
            matched_chunks=matched_chunks,
            question=request.question,
            metadata=matched_metadata[:3]  # Limit to 3 metadata items
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
            "initialized": len(stored_chunks) > 0,
            "size": len(stored_chunks),
            "dimension": 384  # Mock dimension
        },
        "embedding_model": {
            "model_loaded": False,  # Simplified version
            "model_name": "simplified_text_similarity"
        },
        "stored_chunks": len(stored_chunks),
        "documents_loaded": len(document_metadata),
        "rag_status": "ready" if stored_chunks else "no_document"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": [doc.dict() for doc in document_metadata],
        "total_chunks": len(stored_chunks),
        "vector_db_size": len(stored_chunks)
    }

@app.delete("/documents")
async def clear_documents():
    """Clear all uploaded documents"""
    global stored_chunks, document_metadata, chunk_metadata
    
    stored_chunks = []
    document_metadata = []
    chunk_metadata = []
    
    return {
        "message": "All documents cleared successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Research Assistant API - Simplified Version",
        "version": "1.0.0-simplified",
        "model": "Google Gemini Pro" if GEMINI_AVAILABLE else "Mock Responses",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting AI Research Assistant API - Simplified Version")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: {'Google Gemini Pro' if GEMINI_AVAILABLE else 'Mock Responses'}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"⚠️ Note: This is a simplified version without advanced vector database")
    
    uvicorn.run(
        "main_simple:app",
        host=host,
        port=port,
        reload=debug
    )
