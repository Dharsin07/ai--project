import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional
import random
from datetime import datetime
import re

load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyACIGRJJSukfCkCMLS2sUIgn4DGVVAm8YY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Research Assistant API", description="Powered by Google Gemini API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-flash')

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
            relevance_score=round(0.7 + random.random() * 0.3, 2)  # 0.7 to 1.0
        )
        sources.append(source)
    
    return sources

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # Common words to exclude
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
        'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
    }
    
    # Clean and tokenize text
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word.capitalize() for word, _ in keywords[:max_keywords]]

async def generate_with_gemini(prompt: str) -> str:
    """Generate response using Gemini API"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating with Gemini: {e}")
        raise HTTPException(status_code=500, detail="AI generation failed")

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
        
        summary_text = await generate_with_gemini(prompt)
        summary_text = summary_text.strip()
        
        print(f"✅ Generated Gemini response for: {request.topic}")
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_status": "connected"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Research Assistant API",
        "version": "1.0.0",
        "model": "Google Gemini Pro",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting AI Research Assistant API")
    print(f"📍 Server: http://{host}:{port}")
    print(f"🤖 AI Model: Google Gemini Pro")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )
