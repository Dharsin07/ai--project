from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
from datetime import datetime
import re
import PyPDF2
import io

app = FastAPI(title="Mock AI Research Assistant API")

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

# Mock data
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
    """Extract keywords from text"""
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

def generate_answer_from_context(question: str, context: str) -> str:
    """Generate highly specific answer based on question and context"""
    question_lower = question.lower()
    
    # Extract key terms from question
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'does', 'did', 'will', 'would'}
    question_keywords = question_words - stop_words
    
    print(f"🔑 Question keywords: {question_keywords}")
    
    # Split context into sentences
    sentences = re.split(r'[.!?]+', context)
    sentence_scores = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\b\w{3,}\b', sentence_lower))
        
        # Calculate relevance score
        score = 0
        keyword_matches = 0
        
        # Keyword matching (most important)
        for keyword in question_keywords:
            if keyword in sentence_lower:
                keyword_matches += sentence_lower.count(keyword)
                score += sentence_lower.count(keyword) * 3
        
        # Question type specific matching
        if any(word in question_lower for word in ['what', 'define', 'describe', 'explain']):
            # Look for definitions and descriptions
            if any(pattern in sentence_lower for pattern in ['is defined as', 'refers to', 'means', 'is a', 'are', 'definition', 'describes']):
                score += 5
            elif any(pattern in sentence_lower for pattern in ['is', 'are', 'provides', 'contains', 'includes']):
                score += 2
                
        elif any(word in question_lower for word in ['when', 'time', 'date', 'year', 'period']):
            # Look for time-related information
            if any(pattern in sentence_lower for pattern in ['year', 'date', 'time', 'when', 'during', 'period', 'in', 'since', 'until']):
                score += 4
                
        elif any(word in question_lower for word in ['where', 'location', 'place', 'located']):
            # Look for location information
            if any(pattern in sentence_lower for pattern in ['located', 'place', 'location', 'where', 'at', 'in', 'on']):
                score += 4
                
        elif any(word in question_lower for word in ['how', 'process', 'method', 'way', 'steps']):
            # Look for process information
            if any(pattern in sentence_lower for pattern in ['process', 'method', 'steps', 'how', 'by', 'through', 'using', 'first', 'then', 'next']):
                score += 4
                
        elif any(word in question_lower for word in ['why', 'reason', 'cause', 'because', 'purpose']):
            # Look for causal information
            if any(pattern in sentence_lower for pattern in ['because', 'reason', 'cause', 'due', 'leads', 'results', 'purpose', 'in order to']):
                score += 4
                
        elif any(word in question_lower for word in ['who', 'person', 'author', 'creator']):
            # Look for people/author information
            if any(pattern in sentence_lower for pattern in ['who', 'by', 'author', 'created', 'developed', 'wrote']):
                score += 4
        
        # Bonus for sentences that start with relevant words
        if sentence_words and question_keywords:
            first_word = list(sentence_words)[0]
            if first_word in question_keywords:
                score += 2
        
        sentence_scores.append((sentence, score, keyword_matches))
    
    # Sort by score
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 Top 5 sentence scores:")
    for i, (sentence, score, matches) in enumerate(sentence_scores[:5]):
        print(f"  {i+1}. Score: {score}, Matches: {matches}, Text: {sentence[:100]}...")
    
    # Filter sentences with actual keyword matches
    relevant_sentences = [(s, score, matches) for s, score, matches in sentence_scores if matches > 0 and score > 0]
    
    if relevant_sentences:
        # Take the top 1-2 most relevant sentences
        top_sentences = [s[0] for s in relevant_sentences[:2]]
        answer = ". ".join(top_sentences)
        
        # If answer is too short, add one more relevant sentence
        if len(answer) < 80 and len(relevant_sentences) > 2:
            answer += ". " + relevant_sentences[2][0]
        
        return answer + "."
    
    # If no sentences with keyword matches, try broader search
    if sentence_scores:
        best_sentence = sentence_scores[0][0]
        return f"I found this information in the document that may relate to your question: {best_sentence}."
    
    return "I couldn't find specific information related to your question in the uploaded document."

def generate_mock_summary(topic: str, summary_type: str) -> str:
    """Generate mock summary"""
    if summary_type == "short":
        return f"{topic} represents a significant area of contemporary research with broad implications across multiple domains. Recent studies indicate that {topic.lower()} has evolved rapidly, driven by technological advancements and changing societal needs. Key findings suggest that {topic.lower()} offers substantial benefits while also presenting unique challenges that require careful consideration."
    else:
        return f"""{topic} has emerged as a transformative field with far-reaching implications across academic, industrial, and societal domains. This comprehensive analysis examines the current state of {topic.lower()}, its historical development, and future prospects.

Recent advancements in {topic.lower()} have been propelled by converging technologies and interdisciplinary collaboration. Studies indicate that {topic.lower()} addresses critical challenges while creating new opportunities for exploration and discovery. The integration of {topic.lower()} with emerging technologies has resulted in synergistic effects, amplifying its impact and potential applications.

Technical innovations in {topic.lower()} have addressed many previous limitations, enabling more sophisticated applications and implementations. Research indicates that {topic.lower()} systems now demonstrate enhanced performance, reliability, and scalability across diverse use cases. These improvements have facilitated broader adoption and integration of {topic.lower()} solutions in various industries and sectors.

The socioeconomic impact of {topic.lower()} extends beyond technical achievements, influencing workforce development, educational curricula, and policy frameworks. Studies highlight both positive outcomes and potential challenges associated with {topic.lower()} deployment, emphasizing the need for balanced approaches that maximize benefits while mitigating risks."""

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Mock summarize endpoint"""
    if len(req.text) < 100:
        raise HTTPException(status_code=400, detail="Text too short for summarization")

    summary = f"This is a mock summary of the provided text. The original text discusses important concepts that have been condensed into this summary. Key points include the main arguments, supporting evidence, and conclusions presented in the source material."
    
    return {"summary": summary}

@app.post("/research", response_model=ResearchResponse)
async def generate_research_brief(request: ResearchRequest):
    """Generate mock research brief"""
    start_time = datetime.now()
    
    if not request.topic or len(request.topic.strip()) < 10:
        raise HTTPException(status_code=400, detail="Topic must be at least 10 characters long")
    
    if request.summary_type not in ["short", "detailed"]:
        raise HTTPException(status_code=400, detail="Summary type must be 'short' or 'detailed'")
    
    if not 1 <= request.source_count <= 10:
        raise HTTPException(status_code=400, detail="Source count must be between 1 and 10")
    
    # Generate mock summary
    summary_text = generate_mock_summary(request.topic, request.summary_type)
    
    # Generate sources
    sources = generate_mock_sources(request.topic, request.source_count)
    
    # Extract keywords (simplified)
    keywords = [request.topic.split()[0].capitalize(), "Research", "Development", "Technology", "Innovation"][:5]
    
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
    """Query the RAG system with real PDF content"""
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
        
        # Generate answer based on context
        answer = generate_answer_from_context(request.question, context)
        print(f"💭 Generated answer length: {len(answer)} characters")
        
        return RAGQueryResponse(
            answer=answer.strip(),
            matched_chunks=matched_chunks,
            question=request.question
        )
        
    except Exception as e:
        print(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_status": "mock_mode",
        "rag_status": "ready" if stored_chunks else "no_document",
        "current_document": stored_filename if stored_filename else None,
        "chunks_stored": len(stored_chunks)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mock AI Research Assistant API",
        "version": "1.0.0",
        "model": "Mock Mode",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = "0.0.0.0"
    port = 8000
    
    print(f"🚀 Starting Mock AI Research Assistant API")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: Mock Mode (no real AI)")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "mock_server:app",
        host=host,
        port=port,
        reload=False
    )
