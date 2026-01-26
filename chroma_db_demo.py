#!/usr/bin/env python3
"""
ChromaDB Demo Script
Demonstrates ChromaDB functionality without API dependencies
"""

import os
import sys
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
import PyPDF2
import io
from datetime import datetime
from collections import Counter

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ChromaDB not available: {e}")
    print("Please install with: pip install chromadb")
    CHROMADB_AVAILABLE = False

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sentence Transformers not available: {e}")
    print("Please install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class ChromaDBDemo:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            self.initialize_chroma()
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.initialize_embedding_model()
    
    def initialize_chroma(self):
        """Initialize ChromaDB client"""
        print("🔄 Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./demo_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="demo_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ ChromaDB initialized successfully")
        print(f"📁 Database path: ./demo_chroma_db")
        print(f"📊 Current collection size: {self.collection.count()}")
    
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
            
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            sentence_endings = ['. ', '! ', '? ', '\n']
            best_break = -1
            
            for i in range(min(50, len(chunk_words))):
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
        
        return [chunk for chunk in chunks if len(chunk.split()) >= 50]
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library not available")
        
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("✅ Embeddings generated successfully")
        return embeddings
    
    def add_document_to_chroma(self, chunks: List[str], metadata_list: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add document chunks to ChromaDB"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb library not available")
        
        print(f"🔄 Adding {len(chunks)} chunks to ChromaDB...")
        
        chunk_ids = [meta['chunk_id'] for meta in metadata_list]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadata_list,
            ids=chunk_ids
        )
        
        print(f"✅ Successfully added {len(chunks)} chunks to ChromaDB")
        print(f"📊 Total collection size: {self.collection.count()}")
    
    def search_chroma(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Search ChromaDB for similar documents"""
        if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        print(f"🔍 Searching ChromaDB for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        print(f"✅ Found {len(results['documents'][0])} results")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"\n--- Result {i+1} ---")
            print(f"Distance: {distance:.4f}")
            print(f"Document: {metadata['document_name']}")
            print(f"Page: {metadata['page_number']}")
            print(f"Preview: {doc[:150]}...")
        
        return results
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and add to ChromaDB"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🔄 Processing PDF: {pdf_path}")
        
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        file_size = len(pdf_bytes)
        
        pages_text, page_numbers = self.extract_text_from_pdf_with_pages(pdf_bytes)
        
        if not pages_text:
            raise ValueError("No text could be extracted from PDF")
        
        print(f"📄 Extracted text from {len(pages_text)} pages")
        
        all_chunks = []
        metadata_list = []
        
        for page_idx, (page_text, page_num) in enumerate(zip(pages_text, page_numbers)):
            page_chunks = self.split_text_into_meaningful_chunks(page_text)
            
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_id = str(uuid.uuid4())[:8]
                
                metadata = {
                    'chunk_id': chunk_id,
                    'document_name': os.path.basename(pdf_path),
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'word_count': len(chunk.split()),
                    'upload_timestamp': datetime.now().isoformat()
                }
                
                all_chunks.append(chunk)
                metadata_list.append(metadata)
        
        if not all_chunks:
            raise ValueError("No valid chunks created from PDF")
        
        print(f"🔢 Created {len(all_chunks)} chunks")
        
        embeddings = self.generate_embeddings(all_chunks)
        
        self.add_document_to_chroma(all_chunks, metadata_list, embeddings)
        
        total_characters = sum(len(chunk) for chunk in all_chunks)
        
        result = {
            'filename': os.path.basename(pdf_path),
            'chunks_created': len(all_chunks),
            'total_characters': total_characters,
            'pages_processed': len(pages_text),
            'chroma_db_size': self.collection.count(),
            'embedding_dimension': embeddings.shape[1],
            'file_size': file_size
        }
        
        print(f"✅ Successfully processed {os.path.basename(pdf_path)}")
        print(f"📊 Stats: {len(pages_text)} pages, {len(all_chunks)} chunks, {total_characters} characters")
        
        return result
    
    def demo_search(self, query: str):
        """Demonstrate search functionality"""
        print(f"\n🔍 Demo search: '{query}'")
        
        results = self.search_chroma(query, n_results=3)
        
        if not results['documents'][0]:
            print("❌ No results found")
            return
        
        print(f"\n📊 Search Results Summary:")
        print(f"   Results found: {len(results['documents'][0])}")
        print(f"   Average distance: {np.mean(results['distances'][0]):.4f}")
        
    def clear_collection(self):
        """Clear all documents from the collection"""
        if not CHROMADB_AVAILABLE:
            print("❌ ChromaDB not available")
            return
        
        print("🗑️ Clearing ChromaDB collection...")
        self.collection.delete()
        print("✅ Collection cleared successfully")
        print(f"📊 Collection size: {self.collection.count()}")

def main():
    """Main demonstration function"""
    print("🚀 ChromaDB Demo")
    print("=" * 50)
    print("This demo shows ChromaDB functionality for RAG applications")
    print("=" * 50)
    
    if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Required libraries are not available.")
        print("Please install them with:")
        print("pip install chromadb sentence-transformers")
        return
    
    demo = ChromaDBDemo()
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            result = demo.process_pdf(pdf_path)
            
            print(f"\n📋 Processing Summary:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
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
        print("Usage: python chroma_db_demo.py path/to/your/document.pdf")
        
        print("\n📝 Creating demo with sample text...")
        
        sample_text = """
        Artificial Intelligence (AI) is transforming the way we interact with technology. 
        Machine learning algorithms enable computers to learn from data and make predictions. 
        Deep learning, a subset of machine learning, uses neural networks to process complex patterns. 
        Natural Language Processing allows machines to understand and generate human language. 
        Computer Vision enables AI systems to interpret and analyze visual information. 
        These technologies are being applied in healthcare, finance, transportation, and education. 
        The future of AI holds promise for solving some of humanity's most challenging problems.
        """
        
        chunks = demo.split_text_into_meaningful_chunks(sample_text)
        
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': f"demo_{i}",
                'document_name': 'sample_text.txt',
                'page_number': 1,
                'chunk_index': i,
                'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                'word_count': len(chunk.split()),
                'upload_timestamp': datetime.now().isoformat()
            }
            metadata_list.append(metadata)
        
        embeddings = demo.generate_embeddings(chunks)
        demo.add_document_to_chroma(chunks, metadata_list, embeddings)
        
        demo.demo_search("What is machine learning?")
        demo.demo_search("Applications of AI")
        demo.demo_search("Deep learning concepts")

if __name__ == "__main__":
    main()
