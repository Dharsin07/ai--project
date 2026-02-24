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
from mock_arxiv_scraper import MockArXivScraper
from email_sender import EmailSender

app = FastAPI(title="AI Research Assistant API", description="Powered by Groq API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for CSS and JS
app.mount("/static", StaticFiles(directory="."), name="static")

# Add specific routes for common static files
@app.get("/style.css")
async def get_style_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_script_js():
    return FileResponse("script.js", media_type="application/javascript")

# Global storage for PDF chunks
stored_chunks = []
stored_filename = ""

# Get API key from environment or .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        with open('.env', 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle BOM
            for line in f:
                if line.startswith('GROQ_API_KEY='):
                    GROQ_API_KEY = line.split('=', 1)[1].strip()
                    break
    except:
        pass

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print("⚠️  WARNING: No valid Groq API key found. Please set GROQ_API_KEY in your .env file")
    GROQ_API_KEY = None

# Initialize MCP Planner
mcp_planner = MCPPlanner()
print("✅ MCP Planner initialized")

# Initialize Email Sender
email_sender = EmailSender()
email_status = email_sender.get_email_status()
print(f"📧 Email Sender: {'Configured' if email_status['configured'] else 'Demo Mode'}")

# Global storage for collected papers
collected_papers = []
paper_summaries = []

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

class LiveResearchRequest(BaseModel):
    query: str
    paper_count: int = 5
    sort_by: str = "recent"

class PaperData(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    arxiv_url: str
    pdf_url: str
    published_date: str
    paper_id: str
    categories: List[str]

class LiveResearchResponse(BaseModel):
    success: bool
    query: str
    paper_count: int
    collected_at: str
    papers: List[PaperData]
    summaries: List[str]
    email_format: str
    processing_time: float

class EmailConfirmationRequest(BaseModel):
    papers: List[PaperData]
    summaries: List[str]
    recipient_email: str
    subject: str

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

def call_groq_api(prompt: str) -> str:
    """Call Groq API using REST API with retry logic"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    
    data = {
        "model": "llama-3.1-8b-instant",  # Groq's Llama 3.1 model
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 8192
    }
    
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:
                # Rate limit hit - wait with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                    continue
                else:
                    raise HTTPException(status_code=429, detail="API rate limit exceeded. Please wait a few minutes before trying again.")
            
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                raise HTTPException(status_code=500, detail="No response from Groq API")
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Groq API error: {e}")
                raise HTTPException(status_code=500, detail=f"Groq API call failed: {str(e)}")
            print(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(base_delay)
    
    raise HTTPException(status_code=500, detail="Failed to get response from Groq API after multiple retries")

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
        
        if GROQ_API_KEY:
            summary = call_groq_api(prompt)
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
        
        # Generate summary using Groq AI, focusing on research papers
        if request.summary_type == "short":
            prompt = f"""
            Based on current research literature, provide a concise answer to: "{request.topic}"
            
            Focus on recent findings, key insights, and significant developments in this research area.
            Reference actual studies and research findings where relevant.
            
            Guidelines:
            - Provide evidence-based, research-focused answer
            - Be specific and factual
            - Keep it under 150 words
            - Focus on most important research findings
            - Use academic tone but accessible language
            
            Research Question: {request.topic}
            Research Summary:
            """
        else:
            prompt = f"""
            Provide a comprehensive research overview for: "{request.topic}"
            
            Structure your response to include:
            1. **Research Overview**: Current state of research in this area
            2. **Key Findings**: Most significant recent discoveries
            3. **Methodologies**: Common research approaches and methods
            4. **Applications**: Practical applications and implications
            5. **Future Directions**: Emerging trends and research gaps
            
            Guidelines:
            - Focus on actual research evidence and findings
            - Provide depth and detail (300-500 words)
            - Use clear academic structure
            - Reference specific research areas and developments
            - Be comprehensive but research-focused
            
            Question: {request.topic}
            Detailed Answer:
            """
        
        if GROQ_API_KEY:
            try:
                summary_text = call_groq_api(prompt)
            except HTTPException as e:
                if e.status_code == 429:
                    # Rate limit hit - use fallback
                    print("⚠️ Using fallback mode due to rate limits")
                    summary_text = generate_fallback_research_response(request.topic, request.summary_type)
                else:
                    raise
        else:
            # Fallback mock response
            summary_text = generate_fallback_research_response(request.topic, request.summary_type)
        
        summary_text = summary_text.strip()
        
        print(f"✅ Generated response for: {request.topic}")
        
        # Collect real research papers using arXiv scraper
        try:
            # Initialize the arXiv scraper
            scraper = RealArXivScraper()
            
            # Collect real papers
            papers = await scraper.collect_papers(
                topic=request.topic,
                paper_count=request.source_count,
                sort_by="relevance"
            )
            
            print(f"📚 Found {len(papers)} real research papers from arXiv")
            
            # Convert papers to Source format
            sources = []
            for paper in papers:
                source = Source(
                    title=paper.title,
                    authors=", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else ""),
                    url=paper.arxiv_url,
                    year=int(paper.published_date.split('-')[0]) if paper.published_date else 2024,
                    relevance_score=0.8  # Default relevance score
                )
                sources.append(source)
                
        except Exception as e:
            print(f"⚠️ Error collecting real papers: {e}")
            # Fallback to mock sources if arXiv fails
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
        if GROQ_API_KEY:
            try:
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
                
                answer = call_groq_api(prompt)
            except HTTPException as e:
                if e.status_code == 429:
                    # Rate limit hit - use fallback
                    print("⚠️ Using fallback mode due to rate limits")
                    answer = generate_fallback_answer(request.question, context)
                else:
                    raise
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

def generate_fallback_research_response(topic: str, summary_type: str) -> str:
    """Generate fallback research response when API is rate limited"""
    topic_lower = topic.lower()
    
    if summary_type == "short":
        return f"{topic} is an important subject that has gained significant attention in recent years. This field involves key concepts and principles that are essential for understanding its impact and applications. Research in this area continues to evolve, providing valuable insights and practical solutions to various challenges. The study of {topic_lower} offers numerous benefits and opportunities for further exploration and development."
    else:
        return f"""**Understanding {topic}: A Comprehensive Analysis**

{topic} represents a significant area of study and research that has garnered considerable attention across multiple disciplines. This comprehensive examination explores the fundamental aspects, practical applications, and future prospects of {topic_lower}.

**Key Concepts and Principles**
The foundation of {topic_lower} rests on several core principles that guide its implementation and development. These essential concepts provide the framework for understanding how {topic_lower} functions and impacts various domains. Researchers have identified critical factors that contribute to the effectiveness and relevance of {topic_lower} in contemporary contexts.

**Practical Applications and Impact**
The applications of {topic_lower} span numerous fields, demonstrating its versatility and importance. From theoretical frameworks to real-world implementations, {topic_lower} has proven valuable in addressing complex challenges and creating innovative solutions. The practical impact of {topic_lower} continues to grow as new use cases and opportunities emerge.

**Current Research and Developments**
Ongoing research in {topic_lower} has led to significant advancements and breakthrough discoveries. Contemporary studies focus on improving existing methodologies, exploring new possibilities, and addressing limitations in current approaches. The dynamic nature of {topic_lower} research ensures continuous evolution and refinement of best practices.

**Future Prospects and Opportunities**
Looking ahead, {topic_lower} is poised for continued growth and innovation. Emerging trends suggest expanding applications, enhanced capabilities, and broader adoption across various sectors. The future of {topic_lower} holds promise for addressing current challenges and unlocking new possibilities for advancement.

This analysis demonstrates the significance and potential of {topic_lower} as a field of study and practical application, highlighting its importance in both academic and professional contexts."""

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

@app.post("/live-research/collect", response_model=LiveResearchResponse)
async def collect_live_research(request: LiveResearchRequest):
    """Collect latest research papers using Playwright automation"""
    start_time = datetime.now()
    
    try:
        # Detect intent using MCP Planner
        intent = mcp_planner.detect_intent(request.query)
        
        print(f"🔍 Detected intent: {intent.intent_type.value}")
        print(f"📊 Confidence: {intent.confidence:.2f}")
        print(f"📝 Topic: '{intent.topic}'")
        print(f"📄 Paper count: {intent.paper_count}")
        
        # Graceful handling instead of hard errors
        if not mcp_planner.should_use_web_automation(intent):
            print(f"⚠️ Query doesn't match live research pattern, using fallback...")
            # For non-matching queries, still try to collect papers with a default approach
            if intent.confidence < 0.4:
                print(f"🔄 Low confidence ({intent.confidence:.1f}), treating as general research query")
            elif len(intent.topic) < 2:
                print(f"🔄 Topic too short, using 'AI research' as default")
                intent.topic = "AI research"
            else:
                print(f"🔄 Using detected topic: {intent.topic}")
        
        print(f"🔍 Collecting papers on: {intent.topic}")
        print(f"📊 Requested {intent.paper_count} papers")
        
        # Try real arXiv scraper first, fallback to mock if network issues
        papers = []
        try:
            print("🌐 Attempting real arXiv scraper...")
            real_scraper = RealArXivScraper()
            papers = await real_scraper.collect_papers(
                topic=intent.topic,
                paper_count=intent.paper_count,
                sort_by=request.sort_by
            )
            if len(papers) == 0:
                raise Exception("No papers found from arXiv")
        except Exception as e:
            print(f"⚠️ Real arXiv scraper failed: {e}")
            print("🔄 Using mock scraper for demonstration...")
            mock_scraper = MockArXivScraper()
            papers = await mock_scraper.collect_papers(
                topic=intent.topic,
                paper_count=intent.paper_count,
                sort_by=request.sort_by
            )
        
        # Convert to PaperData models
        paper_data_list = []
        for paper in papers:
            paper_data = PaperData(
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                arxiv_url=paper.arxiv_url,
                pdf_url=paper.pdf_url,
                published_date=paper.published_date,
                paper_id=paper.paper_id,
                categories=paper.categories
            )
            paper_data_list.append(paper_data)
        
        # Generate summaries using Gemini API
        summaries = []
        for paper in paper_data_list:
            summary = await generate_paper_summary(paper)
            summaries.append(summary)
        
        # Format as professional email
        email_content = format_research_email(intent.topic, paper_data_list, summaries)
        
        # Store globally for email confirmation
        global collected_papers, paper_summaries
        collected_papers = paper_data_list
        paper_summaries = summaries
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return LiveResearchResponse(
            success=True,
            query=request.query,
            paper_count=len(paper_data_list),
            collected_at=end_time.isoformat(),
            papers=paper_data_list,
            summaries=summaries,
            email_format=email_content,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in live research collection: {e}")
        raise HTTPException(status_code=500, detail=f"Live research collection failed: {str(e)}")

@app.post("/live-research/send-email")
async def send_research_email(request: EmailConfirmationRequest):
    """Send research papers via email"""
    try:
        # Convert papers to dict format for email sender
        papers_dict = []
        for paper in request.papers:
            papers_dict.append({
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "arxiv_url": paper.arxiv_url,
                "pdf_url": paper.pdf_url,
                "published_date": paper.published_date,
                "paper_id": paper.paper_id,
                "categories": paper.categories
            })
        
        # Send real email
        result = email_sender.send_research_papers_email(
            recipient_email=request.recipient_email,
            subject=request.subject,
            papers=papers_dict,
            summaries=request.summaries,
            query="Research Papers Collection"
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "recipient": request.recipient_email,
                "subject": request.subject,
                "email_status": "sent" if not email_sender.get_email_status()["demo_mode"] else "demo",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=f"Email sending failed: {str(e)}")

@app.get("/live-research/status")
async def get_live_research_status():
    """Get status of collected papers"""
    global collected_papers, paper_summaries
    
    return {
        "papers_collected": len(collected_papers),
        "summaries_generated": len(paper_summaries),
        "last_collection": datetime.now().isoformat() if collected_papers else None,
        "ready_to_email": len(collected_papers) > 0,
        "email_status": email_sender.get_email_status()
    }

@app.get("/email/status")
async def get_email_status():
    """Get email configuration status"""
    return email_sender.get_email_status()

async def generate_paper_summary(paper: PaperData) -> str:
    """Generate 2-3 line summary of a research paper"""
    try:
        if GROQ_API_KEY:
            prompt = f"""
            Summarize this research paper in 2-3 concise lines:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors[:3])}
            Abstract: {paper.abstract[:500]}...
            
            Guidelines:
            - Keep it to 2-3 sentences maximum
            - Focus on the main contribution/findings
            - Be specific and informative
            - Avoid technical jargon where possible
            
            Summary:
            """
            
            try:
                summary = call_groq_api(prompt)
                return summary.strip()
            except HTTPException as e:
                if e.status_code == 429:
                    # Rate limit hit - use fallback
                    print(f"⚠️ Rate limit hit for {paper.title[:50]}..., using fallback")
                    return f"Research on {paper.title} with contributions from {paper.authors[0] if paper.authors else 'the authors'}."
                else:
                    raise
        else:
            # Fallback summary
            return f"Research on {paper.title} with contributions from {paper.authors[0] if paper.authors else 'the authors'}."
            
    except Exception as e:
        print(f"Error generating summary for {paper.title}: {e}")
        return f"Research on {paper.title} with contributions from {paper.authors[0] if paper.authors else 'the authors'}."

def format_research_email(topic: str, papers: List[PaperData], summaries: List[str]) -> str:
    """Format research papers as a professional email"""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    email_content = f"""Subject: Latest Research Papers on {topic} - {current_date}

Dear Researcher,

I hope this email finds you well. I've collected the latest research papers on "{topic}" from arXiv.org for your review.

=== RESEARCH PAPERS COLLECTION ===

"""
    
    for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
        email_content += f"""
{i}. {paper.title}

Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
Published: {paper.published_date}
Categories: {', '.join(paper.categories[:3])}

Summary: {summary}

Link: {paper.arxiv_url}
PDF: {paper.pdf_url}

---
"""
    
    email_content += f"""
=== COLLECTION SUMMARY ===
Total Papers: {len(papers)}
Collection Date: {current_date}
Source: arXiv.org

=== NEXT STEPS ===
• Review the abstracts and summaries above
• Download papers of interest using the provided PDF links
• Contact me if you need additional papers or have questions

Best regards,
AI Research Assistant
Powered by Google Gemini & Playwright Automation

---
This email was automatically generated using AI-powered research collection.
For more information or to request additional research topics, please reply to this message.
"""
    
    return email_content

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_status": "connected" if GROQ_API_KEY else "no_api_key",
        "rag_status": "ready" if stored_chunks else "no_document",
        "current_document": stored_filename if stored_filename else None,
        "chunks_stored": len(stored_chunks)
    }

@app.get("/")
async def root():
    """Serve the frontend HTML file"""
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting AI Research Assistant API")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: {'Groq Llama3.1-8b' if GROQ_API_KEY else 'Fallback Mode'}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    
    if not GROQ_API_KEY:
        print("⚠️  WARNING: Running in fallback mode. Set GROQ_API_KEY in .env for AI responses.")
    
    uvicorn.run(
        "gemini_server:app",
        host=host,
        port=port,
        reload=debug
    )
