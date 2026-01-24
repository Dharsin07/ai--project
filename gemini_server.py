import os
import json
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
from datetime import datetime
import re
import PyPDF2
import io

app = FastAPI(title="AI Research Assistant API", description="Powered by Google Gemini API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for PDF chunks
stored_chunks = []
stored_filename = ""

# Get API key from environment or .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    GEMINI_API_KEY = line.split('=', 1)[1].strip()
                    break
    except:
        pass

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("⚠️  WARNING: No valid Gemini API key found. Please set GEMINI_API_KEY in your .env file")
    GEMINI_API_KEY = None

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

class RAGQueryRequest(BaseModel):
    question: str

class RAGQueryResponse(BaseModel):
    answer: str
    matched_chunks: List[str]
    question: str

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

def call_gemini_api(prompt: str) -> str:
    """Call Gemini API using REST API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text']
            return content.strip()
        else:
            raise HTTPException(status_code=500, detail="No response from Gemini API")
            
    except requests.exceptions.RequestException as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {str(e)}")

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
    import re
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    words = re.findall(r'\b\w{4,}\b', text.lower())
    words = [word for word in words if word not in common_words]
    
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word.capitalize() for word, _ in sorted_words[:max_keywords]]

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add sentence with period
        sentence_with_punct = sentence + ". "
        
        # If adding this sentence exceeds chunk size, save current chunk
        if len(current_chunk) + len(sentence_with_punct) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap (last few sentences)
            sentences_in_chunk = re.split(r'[.!?]+', current_chunk)
            overlap_sentences = sentences_in_chunk[-2:] if len(sentences_in_chunk) >= 2 else sentences_in_chunk[-1:]
            current_chunk = ". ".join(overlap_sentences) + ". " + sentence_with_punct
        else:
            current_chunk += sentence_with_punct
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 50]

def simple_text_similarity(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Improved text similarity using keyword matching and scoring"""
    # Extract meaningful words from query (remove stop words)
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    
    # Common stop words to ignore
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who', 'which'}
    
    # Remove stop words from query
    query_words = query_words - stop_words
    
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\b\w{3,}\b', chunk.lower()))
        
        # Calculate multiple similarity scores
        # 1. Jaccard similarity (word overlap)
        intersection = len(query_words.intersection(chunk_words))
        union = len(query_words.union(chunk_words))
        jaccard_score = intersection / union if union > 0 else 0
        
        # 2. Keyword frequency score (how many times query words appear)
        keyword_score = 0
        for word in query_words:
            keyword_score += chunk.lower().count(word)
        
        # 3. Exact phrase matching bonus
        exact_phrase_bonus = 0
        if query.lower() in chunk.lower():
            exact_phrase_bonus = 2.0
        
        # 4. Question word matching bonus
        question_bonus = 0
        if any(word in query.lower() for word in ['what', 'when', 'where', 'why', 'how', 'who', 'which']):
            # Look for sentences that might contain answers
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    question_bonus += 0.5
        
        # Combined score
        final_score = (jaccard_score * 0.3) + (keyword_score * 0.4) + (exact_phrase_bonus * 0.2) + (question_bonus * 0.1)
        
        scored_chunks.append((chunk, final_score, intersection))
    
    # Sort by score and return top_k
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out chunks with no relevance
    relevant_chunks = [chunk for chunk, score, intersection in scored_chunks if intersection > 0 or score > 0.1]
    
    return relevant_chunks[:top_k] if relevant_chunks else [scored_chunks[0][0]] if scored_chunks else []

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
        
        if GEMINI_API_KEY:
            summary = call_gemini_api(prompt)
        else:
            # Fallback mock response
            summary = f"This text discusses important concepts and provides valuable information. The main points include key details about the topic, supporting evidence, and conclusions drawn from the analysis. The content appears to be well-structured and informative."
        
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
        # Validate input
        if not request.topic or len(request.topic.strip()) < 10:
            raise HTTPException(status_code=400, detail="Topic must be at least 10 characters long")
        
        if request.summary_type not in ["short", "detailed"]:
            raise HTTPException(status_code=400, detail="Summary type must be 'short' or 'detailed'")
        
        if not 1 <= request.source_count <= 10:
            raise HTTPException(status_code=400, detail="Source count must be between 1 and 10")
        
        # Generate summary using Gemini
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
        
        if GEMINI_API_KEY:
            summary_text = call_gemini_api(prompt)
        else:
            # Fallback mock response
            summary_text = f"{request.topic} represents an important area of study and research. This topic has gained significant attention due to its relevance and impact on various fields. Key aspects include fundamental principles, practical applications, and future developments. Research in this area continues to evolve, offering new insights and opportunities for advancement."
        
        summary_text = summary_text.strip()
        
        print(f"✅ Generated response for: {request.topic}")
        
        # Generate sources
        sources = generate_mock_sources(request.topic, request.source_count)
        
        # Extract keywords from the AI response
        keywords = extract_keywords(summary_text)
        
        # Calculate processing time
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
    """Upload and process PDF for RAG"""
    global stored_chunks, stored_filename
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        pdf_bytes = await file.read()
        
        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Split into chunks
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from PDF")
        
        # Store chunks
        stored_chunks = chunks
        stored_filename = file.filename
        
        return {
            "message": f"Successfully processed PDF",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_characters": len(text)
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system with Gemini AI"""
    global stored_chunks
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not stored_chunks:
        raise HTTPException(
            status_code=400, 
            detail="No PDF has been uploaded yet. Please upload a PDF document first using the 'Upload Document' section above."
        )
    
    try:
        print(f"🔍 Processing question: {request.question}")
        print(f"📚 Available chunks: {len(stored_chunks)}")
        
        # Find similar chunks using text similarity
        matched_chunks = simple_text_similarity(request.question, stored_chunks, top_k=3)
        print(f"🎯 Matched chunks: {len(matched_chunks)}")
        
        # Combine context from matched chunks
        context = "\n\n---\n\n".join(matched_chunks)
        print(f"📝 Context length: {len(context)} characters")
        
        # Generate answer using Gemini API
        if GEMINI_API_KEY:
            prompt = f"""
            Based on the following document context, please answer the question accurately and specifically.
            
            CONTEXT:
            {context}
            
            QUESTION: {request.question}
            
            Guidelines:
            - Answer ONLY based on the provided context
            - Be specific and direct
            - If the answer is not in the context, say "The answer is not found in the provided document"
            - Include relevant details from the context
            - Keep the answer concise but complete
            
            Answer:
            """
            
            answer = call_gemini_api(prompt)
        else:
            # Fallback to simple keyword matching
            answer = generate_fallback_answer(request.question, context)
        
        print(f"💭 Generated answer length: {len(answer)} characters")
        
        return RAGQueryResponse(
            answer=answer.strip(),
            matched_chunks=matched_chunks,
            question=request.question
        )
        
    except Exception as e:
        print(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed")

def generate_fallback_answer(question: str, context: str) -> str:
    """Fallback answer generation when API is not available"""
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who', 'which'}
    question_keywords = question_words - stop_words
    
    sentences = re.split(r'[.!?]+', context)
    sentence_scores = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        sentence_lower = sentence.lower()
        score = 0
        keyword_matches = 0
        
        for keyword in question_keywords:
            if keyword in sentence_lower:
                keyword_matches += sentence_lower.count(keyword)
                score += sentence_lower.count(keyword) * 3
        
        sentence_scores.append((sentence, score, keyword_matches))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_sentences = [(s, score, matches) for s, score, matches in sentence_scores if matches > 0 and score > 0]
    
    if relevant_sentences:
        top_sentences = [s[0] for s in relevant_sentences[:2]]
        return ". ".join(top_sentences) + "."
    
    return f"I couldn't find specific information related to your question in the uploaded document."

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_status": "connected" if GEMINI_API_KEY else "no_api_key",
        "rag_status": "ready" if stored_chunks else "no_document",
        "current_document": stored_filename if stored_filename else None,
        "chunks_stored": len(stored_chunks)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Research Assistant API",
        "version": "1.0.0",
        "model": "Google Gemini API" if GEMINI_API_KEY else "Fallback Mode",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting AI Research Assistant API")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: {'Google Gemini API' if GEMINI_API_KEY else 'Fallback Mode'}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    
    if not GEMINI_API_KEY:
        print("⚠️  WARNING: Running in fallback mode. Set GEMINI_API_KEY in .env for AI responses.")
    
    uvicorn.run(
        "gemini_server:app",
        host=host,
        port=port,
        reload=debug
    )
