#!/usr/bin/env python3
"""
Vector Database Creation Demo - Simplified Version
Demonstrates the core concepts without heavy dependencies
"""

import os
import sys
import uuid
import math
import re
from typing import List, Dict, Any, Tuple
import PyPDF2
import io
from datetime import datetime
from collections import Counter

class SimpleVectorDatabaseDemo:
    def __init__(self):
        self.stored_chunks = []
        self.chunk_metadata = []
        self.vectors = []
        
    def extract_text_from_pdf_with_pages(self, pdf_bytes: bytes) -> Tuple[List[str], List[int]]:
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
    
    def split_text_into_meaningful_chunks(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """Split text into meaningful chunks of 200-300 words"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Look for sentence endings near the chunk boundary
            sentence_endings = ['. ', '! ', '? ', '\n']
            best_break = -1
            
            for i in range(min(50, len(chunk_words))):  # Look back up to 50 words
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
        
        return [chunk for chunk in chunks if len(chunk.split()) >= 50]  # Filter very small chunks
    
    def create_simple_tf_idf_vector(self, text: str) -> List[float]:
        """Create a simple TF-IDF-like vector for demonstration"""
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        # Tokenize and clean text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(words)
        total_words = len(words)
        
        # Create a simple feature vector (using top 50 most common words across all chunks)
        if not hasattr(self, 'vocabulary'):
            # This would normally be built from all documents
            self.vocabulary = list(set(words))[:100]  # Limit to 100 features for demo
        
        vector = []
        for vocab_word in self.vocabulary:
            if vocab_word in word_counts:
                # Simple TF (term frequency)
                tf = word_counts[vocab_word] / total_words
                vector.append(tf)
            else:
                vector.append(0.0)
        
        return vector
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and create simple vector database"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🔄 Processing PDF: {pdf_path}")
        
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        file_size = len(pdf_bytes)
        
        # Extract text with page numbers
        pages_text, page_numbers = self.extract_text_from_pdf_with_pages(pdf_bytes)
        
        if not pages_text:
            raise ValueError("No text could be extracted from PDF")
        
        print(f"📄 Extracted text from {len(pages_text)} pages")
        
        # Split into meaningful chunks
        all_chunks = []
        new_chunk_metadata = []
        
        # First pass: collect all text to build vocabulary
        all_text = ' '.join(pages_text)
        sample_words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        filtered_words = [word for word in sample_words if word not in stop_words]
        word_counts = Counter(filtered_words)
        self.vocabulary = [word for word, count in word_counts.most_common(100)]
        
        # Second pass: create chunks and vectors
        for page_idx, (page_text, page_num) in enumerate(zip(pages_text, page_numbers)):
            page_chunks = self.split_text_into_meaningful_chunks(page_text)
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_id = str(uuid.uuid4())[:8]  # Short unique ID
                
                metadata = {
                    'chunk_id': chunk_id,
                    'document_name': os.path.basename(pdf_path),
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'word_count': len(chunk.split())
                }
                
                all_chunks.append(chunk)
                new_chunk_metadata.append(metadata)
                
                # Create vector for this chunk
                vector = self.create_simple_tf_idf_vector(chunk)
                self.vectors.append(vector)
        
        if not all_chunks:
            raise ValueError("No valid chunks created from PDF")
        
        print(f"🔢 Created {len(all_chunks)} chunks")
        print(f"📊 Vocabulary size: {len(self.vocabulary)} words")
        
        # Store chunks and metadata
        self.stored_chunks = all_chunks
        self.chunk_metadata = new_chunk_metadata
        
        total_characters = sum(len(chunk) for chunk in all_chunks)
        
        result = {
            'filename': os.path.basename(pdf_path),
            'chunks_created': len(all_chunks),
            'total_characters': total_characters,
            'pages_processed': len(pages_text),
            'vector_db_size': len(self.vectors),
            'embedding_dimension': len(self.vocabulary),
            'file_size': file_size
        }
        
        print(f"✅ Successfully processed {os.path.basename(pdf_path)}")
        print(f"📊 Stats: {len(pages_text)} pages, {len(all_chunks)} chunks, {total_characters} characters")
        
        return result
    
    def search_vector_database(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for similar chunks"""
        if not self.stored_chunks or not self.vectors:
            return []
        
        # Create vector for query
        query_vector = self.create_simple_tf_idf_vector(query)
        
        # Calculate similarities
        similarities = []
        for i, chunk_vector in enumerate(self.vectors):
            similarity = self.cosine_similarity(query_vector, chunk_vector)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for rank, (idx, similarity_score) in enumerate(similarities[:top_k]):
            if similarity_score > 0:  # Only return results with some similarity
                result = {
                    'chunk_text': self.stored_chunks[idx],
                    'similarity_score': similarity_score,
                    'chunk_id': self.chunk_metadata[idx]['chunk_id'] if idx < len(self.chunk_metadata) else f"chunk_{idx}",
                    'document_name': self.chunk_metadata[idx]['document_name'] if idx < len(self.chunk_metadata) else "unknown",
                    'page_number': self.chunk_metadata[idx]['page_number'] if idx < len(self.chunk_metadata) else 0,
                    'chunk_index': self.chunk_metadata[idx]['chunk_index'] if idx < len(self.chunk_metadata) else idx,
                    'rank': rank + 1
                }
                results.append(result)
        
        return results
    
    def demo_search(self, query: str):
        """Demonstrate search functionality"""
        print(f"\n🔍 Searching for: '{query}'")
        
        results = self.search_vector_database(query, top_k=3)
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"✅ Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print(f"Document: {result['document_name']}")
            print(f"Page: {result['page_number']}")
            print(f"Preview: {result['chunk_text'][:200]}...")

def main():
    """Main demonstration function"""
    print("🚀 Vector Database Creation Demo - Simplified Version")
    print("=" * 60)
    print("This demo shows the core concepts of vector database creation")
    print("using simple TF-IDF vectors instead of neural embeddings.")
    print("=" * 60)
    
    # Create demo instance
    demo = SimpleVectorDatabaseDemo()
    
    # Check if a PDF file was provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            # Process the PDF
            result = demo.process_pdf(pdf_path)
            
            print(f"\n📋 Processing Summary:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            # Demonstrate search
            test_queries = [
                "What is the main topic?",
                "Summarize the key findings",
                "Important concepts mentioned"
            ]
            
            for query in test_queries:
                demo.demo_search(query)
                
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
    else:
        print("⚠️ No PDF file provided.")
        print("Usage: python simple_vector_demo.py path/to/your/document.pdf")
        
        # Create a simple demo with sample text
        print("\n📝 Creating demo with sample text...")
        
        sample_text = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that can perform tasks that typically require human intelligence. Machine learning is a subset of AI 
        that enables systems to learn and improve from experience without being explicitly programmed. 
        Deep learning is a further subset of machine learning that uses neural networks with multiple layers 
        to analyze various forms of data. Natural Language Processing (NLP) is another important area of AI 
        that focuses on enabling computers to understand, interpret, and generate human language. 
        Computer Vision is the field that trains computers to interpret and understand the visual world. 
        Robotics combines AI with physical machines to create intelligent robots that can interact with 
        the physical environment. AI has applications in healthcare, finance, transportation, and many other 
        sectors, revolutionizing how we work and live. The development of AI technologies has accelerated 
        in recent years due to advances in computing power, big data availability, and algorithmic improvements.
        """
        
        # Create chunks from sample text
        chunks = demo.split_text_into_meaningful_chunks(sample_text)
        demo.stored_chunks = chunks
        
        # Build vocabulary from sample text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sample_text.lower())
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        demo.vocabulary = [word for word, count in word_counts.most_common(50)]
        
        # Create metadata for demo chunks
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': f"demo_{i}",
                'document_name': 'sample_text.txt',
                'page_number': 1,
                'chunk_index': i,
                'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                'word_count': len(chunk.split())
            }
            demo.chunk_metadata.append(metadata)
            
            # Create vector for this chunk
            vector = demo.create_simple_tf_idf_vector(chunk)
            demo.vectors.append(vector)
        
        print(f"✅ Created demo with {len(chunks)} chunks")
        print(f"📊 Vocabulary size: {len(demo.vocabulary)} words")
        
        # Demonstrate search
        demo.demo_search("What is machine learning?")
        demo.demo_search("Applications of AI")
        demo.demo_search("Deep learning concepts")
        demo.demo_search("Natural language processing")

if __name__ == "__main__":
    main()
