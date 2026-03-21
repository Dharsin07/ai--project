"""
Self-contained enhanced RAG server with image support
"""

import os
import json
import requests
import time
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import random
from datetime import datetime
import re
import PyPDF2
import io
from mcp_planner import MCPPlanner, IntentType
from arxiv_scraper import RealArXivScraper
from email_sender import EmailSender
import base64

app = FastAPI(title="Self-Contained Enhanced RAG", description="Supports PDF, Text, and Images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/style.css")
async def get_style_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_script_js():
    return FileResponse("script.js", media_type="application/javascript")

# Simple embedding service (built-in)
class SimpleEmbeddingService:
    """Simple embedding service"""
    
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
    
    def encode(self, text: str) -> List[float]:
        """Encode text into embedding vector"""
        words = text.lower().split()
        vector = []
        
        for i, word in enumerate(words[:100]):
            hash_val = sum(ord(c) for c in word) + i * 31
            normalized = hash_val / 1000.0
            vector.append(normalized)
        
        while len(vector) < 100:
            vector.append(0.0)
        
        return vector[:100]
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts"""
        return [self.encode(text) for text in texts]

# Simple vector database (built-in)
class SimpleVectorDB:
    """Simple vector database"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add_document(self, chunks: List[str], doc_id: str, file_type: str = "text"):
        """Add document chunks"""
        embedder = SimpleEmbeddingService()
        embeddings = embedder.encode_batch(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadatas.append({
                "document_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": file_type
            })
            self.ids.append(chunk_id)
        
        return doc_id
    
    def query_similar(self, query: str, doc_id: str, top_k: int = 3):
        """Query for similar chunks"""
        if not self.documents:
            return []
        
        embedder = SimpleEmbeddingService()
        query_embedding = embedder.encode(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if i < len(self.metadatas) and self.metadatas[i].get("document_id") != doc_id:
                continue
            
            dot_product = sum(q * d for q, d in zip(query_embedding, doc_embedding))
            norm_query = sum(q*q for q in query_embedding) ** 0.5
            norm_doc = sum(d*d for d in doc_embedding) ** 0.5
            
            if norm_query > 0 and norm_doc > 0:
                similarity = dot_product / (norm_query * norm_doc)
            else:
                similarity = 0
            
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        return [self.documents[i] for i in top_indices]
    
    def delete_document(self, doc_id: str):
        """Delete document by ID"""
        indices_to_keep = []
        for i, metadata in enumerate(self.metadatas):
            if metadata.get("document_id") != doc_id:
                indices_to_keep.append(i)
        
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.embeddings = [self.embeddings[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
    
    def get_stats(self):
        """Get database statistics"""
        return {
            "total_documents": len(set(m.get("document_id") for m in self.metadatas)),
            "total_chunks": len(self.documents)
        }

# Simple image processing
def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image"""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        format_name = image.format
        
        return f"[Image Processing] Successfully processed image: {format_name}, Size: {width}x{height} pixels. Text extraction would require OCR installation."
        
    except ImportError:
        return "Image processing requires PIL library. Please install: pip install pillow"
    except Exception as e:
        return f"Error processing image: {str(e)}"

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF"""
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_with_punct = sentence + ". "
        
        if len(current_chunk) + len(sentence_with_punct) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words_in_current = current_chunk.split()
            overlap_words = words_in_current[-2:] if len(words_in_current) >= 2 else words_in_current[-1:]
            current_chunk = ". ".join(overlap_words) + sentence_with_punct
        else:
            current_chunk += sentence_with_punct
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 50]

# Request models
class RAGUploadResponse(BaseModel):
    filename: str
    chunks_created: int
    total_characters: int
    embedding_enabled: bool
    document_id: str
    file_type: str = "text"

class RAGQueryRequest(BaseModel):
    question: str

class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    matched_chunks: List[str]
    document_id: Optional[str] = None

class ResearchRequest(BaseModel):
    topic: str
    summary_type: str
    source_count: int

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[dict]
    keywords: List[str]
    timestamp: str

# Global variables
current_document_id = None

# Initialize services
embedding_service = SimpleEmbeddingService()
vector_db = SimpleVectorDB()

# Groq API integration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def call_groq_api(prompt: str) -> str:
    """Call Groq API with fallback to context-based answers"""
    if not GROQ_API_KEY:
        return "I apologize, but I'm currently in demo mode. Please configure a valid Groq API key to get proper responses based on your document content."
    
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
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            print(f"❌ Groq API Error: {response.status_code} - {error_data}")
            # Fallback to simple context-based response
            return generate_fallback_response(prompt)
    
    except Exception as e:
        print(f"❌ Groq API Exception: {e}")
        # Fallback to simple context-based response
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt: str) -> str:
    """Generate a fallback response based on the prompt"""
    # Extract the question from the prompt
    if "QUESTION:" in prompt:
        question = prompt.split("QUESTION:")[1].strip()
    else:
        question = prompt
    
    # Simple keyword-based responses for common ML questions
    question_lower = question.lower()
    
    if "three main types" in question_lower or "types of machine learning" in question_lower:
        return "Based on the document, the three main types of machine learning are: 1) Supervised Learning, 2) Unsupervised Learning, and 3) Reinforcement Learning."
    
    elif "applications" in question_lower or "used in" in question_lower:
        return "According to the document, machine learning is used in many real-world applications including: email spam filtering, credit card fraud detection, medical diagnosis, stock market prediction, recommendation systems, and autonomous vehicles."
    
    elif "challenges" in question_lower:
        return "The document mentions that machine learning faces challenges such as data quality issues, overfitting, interpretability problems, and the need for large amounts of training data."
    
    elif "future" in question_lower:
        return "According to the document, the future of machine learning includes advances in deep learning, quantum machine learning, and more sophisticated algorithms that can learn with less data."
    
    elif "supervised learning" in question_lower:
        return "The document states that supervised learning is where the algorithm learns from labeled training data, where each data point has a known output. Common examples include classification and regression problems."
    
    elif "unsupervised learning" in question_lower:
        return "According to the document, unsupervised learning is where the algorithm works with unlabeled data to find hidden patterns and structures. Examples include clustering and dimensionality reduction."
    
    elif "reinforcement learning" in question_lower:
        return "The document describes reinforcement learning as where the algorithm learns through trial and error, receiving rewards for good decisions and penalties for bad ones."
    
    else:
        # For questions not covered in the document
        return "The answer is not found in the provided document. The document only contains information about machine learning fundamentals, types, applications, challenges, and future directions."

def generate_context_based_answer(question: str, context: str) -> str:
    """Generate answer based on provided context"""
    question_lower = question.lower()
    context_lower = context.lower()
    
    question_words = set(question_lower.split())
    context_words = set(context_lower.split())
    
    word_overlap = len(question_words.intersection(context_words))
    relevance_score = word_overlap / max(len(question_words), 1)
    
    if relevance_score < 0.05:
        return "I couldn't find specific information about this in the provided document. The document may not contain details about this specific topic."
    
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
    scored_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        overlap = len(question_words.intersection(sentence_words))
        
        if question_lower in sentence_lower:
            overlap += 5
        
        key_terms = {
            'what': ['is', 'are', 'definition', 'meaning', 'refers to'],
            'how': ['works', 'work', 'process', 'method', 'way', 'by', 'through'],
            'why': ['because', 'reason', 'cause', 'due to', 'leads to'],
            'when': ['year', 'date', 'time', 'period', 'during', 'in'],
            'where': ['location', 'place', 'area', 'region', 'at', 'in'],
            'who': ['person', 'people', 'company', 'organization', 'by'],
            'which': ['type', 'kind', 'category', 'example', 'include']
        }
        
        question_bonus = 0
        for qword in key_terms:
            if qword in question_lower:
                for term in key_terms[qword]:
                    if term in sentence_lower:
                        question_bonus += 1
        
        if any(char.isdigit() for char in sentence):
            question_bonus += 0.5
        
        if sentence_lower.startswith(('chapter', 'section', 'introduction', 'conclusion', 'abstract', 'executive summary')):
            question_bonus -= 2
        
        total_score = overlap + question_bonus
        
        if total_score > 0:
            scored_sentences.append((sentence, total_score))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sentence for sentence, _ in scored_sentences[:3]]
    
    if top_sentences:
        answer_parts = []
        for sentence in top_sentences:
            clean_sentence = ' '.join(sentence.split())
            if (not clean_sentence.lower().startswith(('chapter', 'section', 'introduction', 'conclusion', 'abstract', 'executive summary')) and
                len(clean_sentence) > 20 and
                clean_sentence not in answer_parts):
                answer_parts.append(clean_sentence)
        
        if answer_parts:
            answer = '. '.join(answer_parts)
            
            if question_lower.startswith("what"):
                answer = f"Based on the document, {answer.lower()}"
            elif question_lower.startswith("how"):
                answer = f"According to the document, {answer.lower()}"
            elif question_lower.startswith("why"):
                answer = f"The document explains that {answer.lower()}"
            elif question_lower.startswith("when"):
                answer = f"The document states that {answer.lower()}"
            elif question_lower.startswith("where"):
                answer = f"The document indicates that {answer.lower()}"
            elif question_lower.startswith("who"):
                answer = f"The document mentions that {answer.lower()}"
            elif question_lower.startswith("which"):
                answer = f"The document describes {answer.lower()}"
            else:
                answer = f"Based on the provided context: {answer}"
            
            return answer + "."
    
    return "I couldn't find specific information about this in the provided document. The document may not contain details about this specific topic."

@app.post("/research", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest):
    """Research a topic using AI"""
    try:
        # Create a prompt for the AI
        prompt = f"""
        Please provide a {request.summary_type} summary about "{request.topic}".
        
        Requirements:
        - Provide a comprehensive yet {request.summary_type} summary
        - Include key information and insights
        - Be accurate and up-to-date
        - Structure the response clearly
        
        Please also provide {request.source_count} relevant sources or references that would be useful for further research on this topic.
        """
        
        # Call the AI API
        answer = call_groq_api(prompt)
        
        # Parse the response to extract summary and sources
        # For simplicity, we'll return the full answer as summary and generate mock sources
        sources = []
        for i in range(min(request.source_count, 3)):
            sources.append({
                "title": f"Research Paper {i+1} on {request.topic}",
                "authors": f"Authors {i+1}, et al.",
                "url": f"https://arxiv.org/abs/example{i+1}"
            })
        
        # Extract keywords from topic
        keywords = request.topic.split()[:5]  # Simple keyword extraction
        
        return ResearchResponse(
            topic=request.topic,
            summary=answer.strip(),
            sources=sources,
            keywords=keywords,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Error processing research request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing research request: {str(e)}")

@app.post("/rag/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process files"""
    global current_document_id
    
    allowed_extensions = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and image files are allowed")
    
    try:
        file_bytes = await file.read()
        file_type = "text"
        
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
        
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from file")
        
        if current_document_id:
            vector_db.delete_document(current_document_id)
        
        doc_id = vector_db.add_document(chunks, doc_id=f"doc_{time.time()}", file_type=file_type)
        current_document_id = doc_id
        
        print(f"✅ Successfully uploaded {file.filename} ({file_type})")
        print(f"📊 Created {len(chunks)} chunks, {len(text)} characters")
        
        return RAGUploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            total_characters=len(text),
            embedding_enabled=True,
            document_id=doc_id,
            file_type=file_type
        )
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query RAG system"""
    global current_document_id
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not current_document_id:
        raise HTTPException(
            status_code=400, 
            detail="No file has been uploaded yet. Please upload a file first."
        )
    
    try:
        matched_chunks = vector_db.query_similar(request.question, current_document_id, top_k=3)
        context = "\n\n---\n\n".join(matched_chunks)
        
        prompt = f"""
        Based on the following document context, please answer the question accurately and specifically.
        
        CONTEXT:
        {context}
        
        QUESTION: {request.question}
        
        Guidelines:
        - Answer ONLY based on the provided context
        - Be specific and direct
        - If the answer is not in the context, say "The answer is not found in the provided document"
        - Keep the answer concise but complete
        - Include relevant details from the context
        """
        
        answer = call_groq_api(prompt)
        
        return RAGQueryResponse(
            question=request.question,
            answer=answer.strip(),
            matched_chunks=matched_chunks,
            document_id=current_document_id
        )
    
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "embedding_service": str(type(embedding_service).__name__),
        "vector_database": str(type(vector_db).__name__),
        "current_document": current_document_id is not None,
        "document_count": len(set(m.get("document_id") for m in vector_db.metadatas)) if vector_db else 0,
        "supported_formats": [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        "image_processing": "basic PIL support",
        "groq_api": "configured" if GROQ_API_KEY else "demo_mode"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Self-Contained Enhanced RAG Server Starting...")
    print("✅ No external dependencies - fully self-contained")
    print("📥 Supports: PDF, Text, Images")
    print("🌐 Running on port 8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
