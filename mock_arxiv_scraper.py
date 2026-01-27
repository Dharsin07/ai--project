"""
Mock ArXiv Scraper for demonstration purposes
Provides realistic paper data when real scraping fails
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import random

@dataclass
class ResearchPaper:
    """Data structure for a research paper"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_url: str
    pdf_url: str
    published_date: str
    paper_id: str
    categories: List[str]

class MockArXivScraper:
    """Mock scraper that provides realistic demo data"""
    
    def __init__(self):
        self.base_url = "https://arxiv.org"
        
        # Sample paper templates
        self.paper_templates = {
            "AI security": [
                {
                    "title": "Adversarial Attacks on Deep Learning Systems: A Comprehensive Survey",
                    "authors": ["John Smith", "Emily Johnson", "Michael Brown", "Sarah Davis"],
                    "abstract": "This paper presents a comprehensive survey of adversarial attacks on deep learning systems. We analyze various attack methodologies including white-box and black-box attacks, and discuss their implications for AI security. Our analysis covers recent advances in attack techniques and defense mechanisms.",
                    "categories": ["cs.AI", "cs.CR", "cs.LG"]
                },
                {
                    "title": "Federated Learning with Privacy-Preserving Mechanisms: Security Analysis",
                    "authors": ["Alice Chen", "Bob Wilson", "Carol Martinez", "David Lee"],
                    "abstract": "Federated learning enables collaborative model training without sharing raw data, but privacy and security concerns remain. This paper analyzes various privacy-preserving mechanisms in federated learning and identifies potential vulnerabilities. We propose new defense strategies against model inversion and membership inference attacks.",
                    "categories": ["cs.LG", "cs.CR", "cs.AI"]
                },
                {
                    "title": "Robust Machine Learning: Defending Against Data Poisoning Attacks",
                    "authors": ["Robert Taylor", "Lisa Anderson", "James White", "Maria Garcia"],
                    "abstract": "Data poisoning attacks pose significant threats to machine learning systems. This paper presents a comprehensive analysis of data poisoning techniques and proposes robust training methods to defend against such attacks. Our experimental evaluation demonstrates the effectiveness of our proposed defenses across multiple datasets and attack scenarios.",
                    "categories": ["cs.LG", "cs.CR", "cs.AI"]
                }
            ],
            "machine learning": [
                {
                    "title": "Transformers Beyond NLP: Applications in Computer Vision and Speech Recognition",
                    "authors": ["Kevin Zhang", "Jennifer Liu", "Thomas Wang", "Rachel Kim"],
                    "abstract": "Transformer architectures have revolutionized natural language processing and are now being applied to computer vision and speech recognition. This paper explores the adaptation of transformers for multimodal tasks and presents novel architectures that achieve state-of-the-art performance across multiple domains.",
                    "categories": ["cs.LG", "cs.CV", "cs.CL"]
                },
                {
                    "title": "Self-Supervised Learning: A Comprehensive Review and Future Directions",
                    "authors": ["Daniel Park", "Sophie Turner", "Christopher Lee", "Amanda Brown"],
                    "abstract": "Self-supervised learning has emerged as a powerful paradigm for learning representations from unlabeled data. This paper provides a comprehensive review of self-supervised learning methods, including contrastive learning, masked language modeling, and clustering-based approaches. We discuss current challenges and future research directions.",
                    "categories": ["cs.LG", "cs.AI", "cs.CV"]
                }
            ],
            "quantum computing": [
                {
                    "title": "Quantum Machine Learning: Algorithms and Applications",
                    "authors": ["Alex Johnson", "Michelle Chen", "Steven Davis", "Laura Wilson"],
                    "abstract": "Quantum computing offers new possibilities for machine learning algorithms. This paper reviews quantum machine learning algorithms including quantum support vector machines, quantum neural networks, and variational quantum classifiers. We discuss potential applications and current limitations of quantum ML approaches.",
                    "categories": ["cs.AI", "cs.LG", "quant-ph"]
                }
            ]
        }
    
    async def collect_papers(self, topic: str, paper_count: int = 5, sort_by: str = "recent") -> List[ResearchPaper]:
        """Collect papers using mock data"""
        try:
            # Limit paper count
            paper_count = min(max(paper_count, 1), 10)
            
            print(f"🔍 Using mock data for: {topic}")
            print(f"📊 Generating {paper_count} papers")
            
            # Find relevant topic
            topic_lower = topic.lower()
            selected_papers = []
            
            # Try to find matching topic
            for key, templates in self.paper_templates.items():
                if any(word in topic_lower for word in key.split()):
                    selected_papers = templates
                    break
            
            # If no match, use generic AI papers
            if not selected_papers:
                selected_papers = self.paper_templates["AI security"]
            
            # Generate papers
            papers = []
            for i in range(min(paper_count, len(selected_papers))):
                template = selected_papers[i % len(selected_papers)]
                
                # Generate unique IDs and URLs
                paper_id = f"2401.{1000 + i:04d}v1"
                arxiv_url = f"{self.base_url}/abs/{paper_id}"
                pdf_url = f"{self.base_url}/pdf/{paper_id}.pdf"
                
                # Generate recent date
                days_ago = random.randint(1, 30)
                pub_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                
                paper = ResearchPaper(
                    title=template["title"],
                    authors=template["authors"],
                    abstract=template["abstract"],
                    arxiv_url=arxiv_url,
                    pdf_url=pdf_url,
                    published_date=pub_date,
                    paper_id=paper_id,
                    categories=template["categories"]
                )
                papers.append(paper)
            
            print(f"✅ Generated {len(papers)} mock papers")
            return papers
            
        except Exception as e:
            print(f"❌ Error generating mock papers: {e}")
            raise
    
    def papers_to_json(self, papers: List[ResearchPaper]) -> str:
        """Convert papers list to JSON format"""
        papers_data = []
        for paper in papers:
            papers_data.append({
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "arxiv_url": paper.arxiv_url,
                "pdf_url": paper.pdf_url,
                "published_date": paper.published_date,
                "paper_id": paper.paper_id,
                "categories": paper.categories
            })
        
        return json.dumps({
            "success": True,
            "paper_count": len(papers_data),
            "collected_at": datetime.now().isoformat(),
            "papers": papers_data
        }, indent=2)

# Add missing import
from datetime import timedelta

# Usage example
async def main():
    """Example usage of Mock ArXiv scraper"""
    scraper = MockArXivScraper()
    papers = await scraper.collect_papers("AI security", 3, "recent")
    
    print(f"\n📊 Generated {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:2])}...")
        print(f"   Published: {paper.published_date}")
        print(f"   Categories: {', '.join(paper.categories)}")
    
    # Save to JSON
    json_output = scraper.papers_to_json(papers)
    with open("mock_papers.json", "w") as f:
        f.write(json_output)
    print(f"\n💾 Mock data saved to mock_papers.json")

if __name__ == "__main__":
    asyncio.run(main())
