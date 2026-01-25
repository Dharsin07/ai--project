#!/usr/bin/env python3
"""
Vector Database Creation Demo
Standalone demonstration of the vector database functionality
"""

import os
import sys
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
import PyPDF2
import io
from datetime import datetime

# Try to import the required libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Required libraries not available: {e}")
    print("Please install with: pip install sentence-transformers faiss-cpu")
    LIBRARIES_AVAILABLE = False

class VectorDatabaseDemo:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.stored_chunks = []
        self.chunk_metadata = []
        
        if LIBRARIES_AVAILABLE:
            self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
    
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
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not LIBRARIES_AVAILABLE:
            raise ImportError("sentence-transformers library not available")
        
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("✅ Embeddings generated successfully")
        return embeddings
    
    def create_vector_database(self, embeddings: np.ndarray):
        """Create FAISS vector database"""
        if not LIBRARIES_AVAILABLE:
            raise ImportError("faiss library not available")
        
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized_embeddings)
        
        print(f"✅ Vector database created with {index.ntotal} embeddings")
        self.vector_db = index
        return index
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and create vector database"""
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
        
        if not all_chunks:
            raise ValueError("No valid chunks created from PDF")
        
        print(f"🔢 Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(all_chunks)
        
        # Create vector database
        self.create_vector_database(embeddings)
        
        # Store chunks and metadata
        self.stored_chunks = all_chunks
        self.chunk_metadata = new_chunk_metadata
        
        total_characters = sum(len(chunk) for chunk in all_chunks)
        
        result = {
            'filename': os.path.basename(pdf_path),
            'chunks_created': len(all_chunks),
            'total_characters': total_characters,
            'pages_processed': len(pages_text),
            'vector_db_size': self.vector_db.ntotal,
            'embedding_dimension': embeddings.shape[1],
            'file_size': file_size
        }
        
        print(f"✅ Successfully processed {os.path.basename(pdf_path)}")
        print(f"📊 Stats: {len(pages_text)} pages, {len(all_chunks)} chunks, {total_characters} characters")
        
        return result
    
    def search_vector_database(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for similar chunks"""
        if not LIBRARIES_AVAILABLE:
            raise ImportError("Required libraries not available")
        
        if self.vector_db is None or not self.stored_chunks:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search vector database
        scores, indices = self.vector_db.search(query_embedding, min(top_k, len(self.stored_chunks)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.stored_chunks):
                result = {
                    'chunk_text': self.stored_chunks[idx],
                    'similarity_score': float(score),
                    'chunk_id': self.chunk_metadata[idx]['chunk_id'] if idx < len(self.chunk_metadata) else f"chunk_{idx}",
                    'document_name': self.chunk_metadata[idx]['document_name'] if idx < len(self.chunk_metadata) else "unknown",
                    'page_number': self.chunk_metadata[idx]['page_number'] if idx < len(self.chunk_metadata) else 0,
                    'chunk_index': self.chunk_metadata[idx]['chunk_index'] if idx < len(self.chunk_metadata) else idx,
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def demo_search(self, query: str):
        """Demonstrate search functionality"""
        print(f"\n🔍 Searching for: '{query}'")
        
        if not LIBRARIES_AVAILABLE:
            print("❌ Cannot search: required libraries not available")
            return
        
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
    print("🚀 Vector Database Creation Demo")
    print("=" * 50)
    
    if not LIBRARIES_AVAILABLE:
        print("❌ Required libraries are not available.")
        print("Please install them with:")
        print("pip install sentence-transformers faiss-cpu")
        return
    
    # Create demo instance
    demo = VectorDatabaseDemo()
    
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
        print("Usage: python vector_db_demo.py path/to/your/document.pdf")
        
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
        sectors, revolutionizing how we work and live.
        """
        
        # Create chunks from sample text
        chunks = demo.split_text_into_meaningful_chunks(sample_text)
        demo.stored_chunks = chunks
        
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
        
        # Generate embeddings and create vector database
        embeddings = demo.generate_embeddings(chunks)
        demo.create_vector_database(embeddings)
        
        print(f"✅ Created demo with {len(chunks)} chunks")
        
        # Demonstrate search
        demo.demo_search("What is machine learning?")
        demo.demo_search("Applications of AI")
        demo.demo_search("Deep learning concepts")

if __name__ == "__main__":
    main()
