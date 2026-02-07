"""
Real ArXiv Scraper using arXiv API
Gets actual research papers from arXiv.org
"""
   
import asyncio
import json
import re
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

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

class RealArXivScraper:
    """Real arXiv scraper using arXiv API"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def collect_papers(self, topic: str, paper_count: int = 5, sort_by: str = "recent") -> List[ResearchPaper]:
        """
        Collect real research papers from arXiv using API
        
        Args:
            topic: Research topic to search for
            paper_count: Number of papers to collect (max 10)
            sort_by: Sorting method ("recent", "relevance")
        
        Returns:
            List of ResearchPaper objects
        """
        try:
            # Limit paper count to reasonable range
            paper_count = min(max(paper_count, 1), 10)
            
            print(f"🔍 Searching arXiv API for: {topic}")
            print(f"📊 Collecting {paper_count} real papers, sorted by {sort_by}")
            
            # Build arXiv API query
            # Clean the topic - remove collection keywords
            clean_topic = re.sub(r'(collect|latest|recent|new|papers?|research)', '', topic, flags=re.IGNORECASE).strip()
            search_query = clean_topic.replace(" ", "+") if clean_topic else topic.replace(" ", "+")
            sort_param = "submittedDate" if sort_by == "recent" else "relevance"
            
            params = {
                "search_query": f"all:{search_query}",
                "start": 0,
                "max_results": paper_count,
                "sortBy": sort_param,
                "sortOrder": "descending"
            }
            
            print(f"🔍 Search query: {search_query}")
            print(f"📋 Parameters: {params}")
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            print(f"📄 arXiv API response length: {len(response.text)} characters")
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.text)
            
            print(f"✅ Successfully collected {len(papers)} real papers from arXiv")
            return papers
            
        except Exception as e:
            print(f"❌ Error collecting real papers: {e}")
            raise
    
    def _parse_arxiv_response(self, xml_text: str) -> List[ResearchPaper]:
        """Parse arXiv XML API response"""
        papers = []
        
        try:
            print(f"🔍 Parsing XML response...")
            
            # Simple XML parsing (without xml.etree to avoid dependencies)
            entries = self._extract_entries(xml_text)
            print(f"📄 Found {len(entries)} entries in XML")
            
            for i, entry in enumerate(entries):
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
                    print(f"✅ Parsed paper {i+1}: {paper.title[:50]}...")
                else:
                    print(f"⚠️ Failed to parse entry {i+1}")
            
            return papers
            
        except Exception as e:
            print(f"❌ Error parsing arXiv response: {e}")
            print(f"📄 XML preview: {xml_text[:500]}...")
            raise
    
    def _extract_entries(self, xml_text: str) -> List[str]:
        """Extract individual entry blocks from XML"""
        # Find all <entry> blocks
        entry_pattern = r'<entry>(.*?)</entry>'
        entries = re.findall(entry_pattern, xml_text, re.DOTALL)
        return entries
    
    def _parse_entry(self, entry_xml: str) -> Optional[ResearchPaper]:
        """Parse individual arXiv entry"""
        try:
            # Extract title
            title_match = re.search(r'<title>(.*?)</title>', entry_xml, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "No title found"
            
            # Extract authors
            authors = []
            author_matches = re.findall(r'<name>(.*?)</name>', entry_xml)
            for author in author_matches:
                authors.append(author.strip())
            
            # Extract abstract
            abstract_match = re.search(r'<summary>(.*?)</summary>', entry_xml, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else "No abstract found"
            
            # Extract arXiv ID and URLs
            id_match = re.search(r'<id>(.*?)</id>', entry_xml)
            arxiv_url = id_match.group(1).strip() if id_match else ""
            
            # Extract paper ID from URL
            paper_id = ""
            if arxiv_url:
                id_match = re.search(r'arxiv\.org/abs/(.+)', arxiv_url)
                if id_match:
                    paper_id = id_match.group(1)
            
            # Generate PDF URL
            pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf" if arxiv_url else ""
            
            # Extract publication date
            published_match = re.search(r'<published>(.*?)</published>', entry_xml)
            published_date = published_match.group(1).strip() if published_match else "Unknown date"
            
            # Format date to be more readable
            if published_date != "Unknown date":
                try:
                    # Parse ISO date and format
                    dt = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                    published_date = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            # Extract categories
            categories = []
            category_matches = re.findall(r'<term>(.*?)</term>', entry_xml)
            for category in category_matches:
                categories.append(category.strip())
            
            return ResearchPaper(
                title=self._clean_text(title),
                authors=authors[:10],  # Limit authors
                abstract=self._clean_text(abstract),
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                published_date=published_date,
                paper_id=paper_id,
                categories=categories[:5]  # Limit categories
            )
            
        except Exception as e:
            print(f"⚠️ Error parsing entry: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean XML text content"""
        # Remove HTML tags and extra whitespace
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
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

# Usage example
async def main():
    """Example usage of Real ArXiv scraper"""
    scraper = RealArXivScraper()
    papers = scraper.collect_papers("machine learning", 3, "recent")
    
    print(f"\n📊 Collected {len(papers)} real papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:2])}...")
        print(f"   Published: {paper.published_date}")
        print(f"   Categories: {', '.join(paper.categories)}")
        print(f"   arXiv ID: {paper.paper_id}")
    
    # Save to JSON
    json_output = scraper.papers_to_json(papers)
    with open("real_arxiv_papers.json", "w") as f:
        f.write(json_output)
    print(f"\n💾 Real papers data saved to real_arxiv_papers.json")

if __name__ == "__main__":
    asyncio.run(main())
