"""
MCP Planner for Live Research Paper Collector
Detects user intent and decides when to use web automation
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class IntentType(Enum):
    LIVE_RESEARCH_COLLECTION = "live_research_collection"
    REGULAR_RESEARCH = "regular_research"
    RAG_QUERY = "rag_query"
    UNKNOWN = "unknown"

@dataclass
class ResearchIntent:
    intent_type: IntentType
    topic: str
    paper_count: int
    confidence: float
    extracted_params: Dict[str, any]

class MCPPlanner:
    def __init__(self):
        # Keywords that trigger live research collection
        self.live_collection_keywords = [
            "collect", "latest", "recent", "new", "current", "up-to-date",
            "papers", "research papers", "arxiv", "scholar", "publications",
            "find papers", "get papers", "search papers", "latest research",
            "study", "studies", "research", "article", "articles", "journal",
            "publication", "literature", "review", "survey", "analysis"
        ]
        
        # Number extraction patterns
        self.number_patterns = [
            r"(\d+)\s+(?:papers?|research\s+papers?|studies?)",
            r"(?:latest|recent|new)\s+(\d+)\s+(?:papers?|research\s+papers?)",
            r"collect\s+(\d+)\s+(?:papers?|research\s+papers?)",
            r"(\d+)\s+(?:latest|recent|new)\s+(?:papers?|research\s+papers?)"
        ]
    
    def detect_intent(self, user_query: str) -> ResearchIntent:
        """
        Detect user intent and extract parameters
        """
        query_lower = user_query.lower()
        
        # Check for live research collection intent
        live_score = self._calculate_live_collection_score(query_lower)
        
        # Extract topic and parameters
        topic = self._extract_topic(user_query)
        paper_count = self._extract_paper_count(query_lower)
        
        # Determine intent type and confidence
        if live_score >= 0.3:  # Lowered from 0.6 to 0.3
            intent_type = IntentType.LIVE_RESEARCH_COLLECTION
            confidence = live_score
        elif "rag" in query_lower or "document" in query_lower or "pdf" in query_lower:
            intent_type = IntentType.RAG_QUERY
            confidence = 0.8
        else:
            intent_type = IntentType.REGULAR_RESEARCH
            confidence = 0.7
        
        return ResearchIntent(
            intent_type=intent_type,
            topic=topic,
            paper_count=paper_count,
            confidence=confidence,
            extracted_params={
                "original_query": user_query,
                "detected_keywords": self._find_matching_keywords(query_lower),
                "requires_web_automation": intent_type == IntentType.LIVE_RESEARCH_COLLECTION
            }
        )
    
    def _calculate_live_collection_score(self, query: str) -> float:
        """Calculate confidence score for live research collection intent"""
        score = 0.0
        keyword_matches = 0
        
        for keyword in self.live_collection_keywords:
            if keyword in query:
                keyword_matches += 1
                score += 0.3  # Equal score for all research keywords
        
        # Bonus for multiple keywords
        if keyword_matches >= 2:
            score += 0.2
        
        # Bonus for specific patterns
        if any(pattern in query for pattern in ["collect", "latest", "recent", "new"]):
            score += 0.1
        
        if any(pattern in query for pattern in ["papers", "arxiv", "research"]):
            score += 0.1
        
        # Minimum score for any substantive query (3+ words)
        if len(query.split()) >= 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _extract_topic(self, query: str) -> str:
        """Extract the research topic from user query"""
        # Remove collection-related keywords to isolate topic
        topic = query
        
        # Remove common collection phrases
        phrases_to_remove = [
            r"collect\s+(?:latest\s+)?(?:\d+\s+)?research\s+papers?\s+(?:on|about|for|in)?",
            r"find\s+(?:latest\s+)?(?:\d+\s+)?research\s+papers?\s+(?:on|about|for|in)?",
            r"get\s+(?:latest\s+)?(?:\d+\s+)?research\s+papers?\s+(?:on|about|for|in)?",
            r"search\s+(?:for\s+)?(?:latest\s+)?(?:\d+\s+)?research\s+papers?\s+(?:on|about|for|in)?",
            r"(?:latest|recent|new)\s+(?:\d+\s+)?research\s+papers?\s+(?:on|about|for|in)?"
        ]
        
        for phrase in phrases_to_remove:
            topic = re.sub(phrase, "", topic, flags=re.IGNORECASE)
        
        # Clean up the topic
        topic = topic.strip()
        
        # If topic is empty or too short, use the whole query
        if len(topic) < 3:
            topic = query
        
        return topic
    
    def _extract_paper_count(self, query: str) -> int:
        """Extract the number of papers requested"""
        for pattern in self.number_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    count = int(match.group(1))
                    # Limit to reasonable range
                    return min(max(count, 1), 10)
                except (ValueError, IndexError):
                    continue
        
        # Default count
        return 5
    
    def _find_matching_keywords(self, query: str) -> List[str]:
        """Find which keywords matched in the query"""
        matched = []
        for keyword in self.live_collection_keywords:
            if keyword in query:
                matched.append(keyword)
        return matched
    
    def should_use_web_automation(self, intent: ResearchIntent) -> bool:
        """Decide if web automation is needed"""
        return (
            intent.intent_type == IntentType.LIVE_RESEARCH_COLLECTION and
            intent.confidence >= 0.4 and  # Lowered from 0.6 to 0.4
            len(intent.topic) >= 2      # Lowered from 3 to 2
        )
    
    def get_execution_plan(self, intent: ResearchIntent) -> Dict[str, any]:
        """Generate execution plan for the detected intent"""
        if intent.intent_type == IntentType.LIVE_RESEARCH_COLLECTION:
            return {
                "tool": "arxiv_scraper",
                "action": "collect_papers",
                "parameters": {
                    "topic": intent.topic,
                    "paper_count": intent.paper_count,
                    "sort_by": "recent",
                    "source": "arxiv"
                },
                "follow_up_actions": [
                    "summarize_papers",
                    "format_email",
                    "request_confirmation"
                ]
            }
        elif intent.intent_type == IntentType.RAG_QUERY:
            return {
                "tool": "rag_system",
                "action": "query_documents",
                "parameters": {
                    "question": intent.extracted_params.get("original_query", "")
                }
            }
        else:
            return {
                "tool": "gemini_api",
                "action": "generate_research",
                "parameters": {
                    "topic": intent.topic,
                    "summary_type": "detailed"
                }
            }

# Example usage and testing
if __name__ == "__main__":
    planner = MCPPlanner()
    
    test_queries = [
        "Collect latest 5 AI security research papers",
        "Find recent papers on machine learning",
        "What are the benefits of quantum computing?",
        "Summarize the uploaded PDF document",
        "Get 3 new papers about climate change"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        intent = planner.detect_intent(query)
        plan = planner.get_execution_plan(intent)
        
        print(f"Intent: {intent.intent_type.value}")
        print(f"Topic: {intent.topic}")
        print(f"Paper Count: {intent.paper_count}")
        print(f"Confidence: {intent.confidence:.2f}")
        print(f"Use Web Automation: {planner.should_use_web_automation(intent)}")
        print(f"Plan: {plan}")
