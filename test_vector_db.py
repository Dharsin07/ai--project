#!/usr/bin/env python3
"""
Vector Database Test Script
Demonstrates the vector database creation and search functionality
"""

import requests
import json
import time
import os
from pathlib import Path

class VectorDatabaseTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test the health check endpoint"""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {data['status']}")
                print(f"   Vector DB: {data['vector_database']}")
                print(f"   Embedding Model: {data['embedding_model']}")
                print(f"   Stored Chunks: {data['stored_chunks']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_pdf_upload(self, pdf_path):
        """Test PDF upload and processing"""
        print(f"\n📄 Testing PDF upload: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                response = self.session.post(f"{self.base_url}/rag/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ PDF upload successful")
                print(f"   Filename: {data['filename']}")
                print(f"   Chunks created: {data['chunks_created']}")
                print(f"   Pages processed: {data['pages_processed']}")
                print(f"   Vector DB size: {data['vector_db_size']}")
                print(f"   Embedding dimension: {data['embedding_dimension']}")
                return True
            else:
                print(f"❌ PDF upload failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ PDF upload error: {e}")
            return False
    
    def test_vector_search(self, query):
        """Test vector search functionality"""
        print(f"\n🔍 Testing vector search: '{query}'")
        
        try:
            payload = {"query": query, "top_k": 3}
            response = self.session.post(
                f"{self.base_url}/vector/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Vector search successful")
                print(f"   Query: {data['query']}")
                print(f"   Answer preview: {data['answer'][:100]}...")
                print(f"   Matches found: {len(data['matches'])}")
                
                for i, match in enumerate(data['matches'][:2], 1):
                    print(f"   Match {i}:")
                    print(f"     Similarity: {match['similarity_score']:.3f}")
                    print(f"     Document: {match['document_name']}")
                    print(f"     Page: {match['page_number']}")
                    print(f"     Preview: {match['chunk_text'][:100]}...")
                
                return True
            else:
                print(f"❌ Vector search failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return False
    
    def test_rag_query(self, question):
        """Test RAG query functionality"""
        print(f"\n🤖 Testing RAG query: '{question}'")
        
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/rag/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ RAG query successful")
                print(f"   Question: {data['question']}")
                print(f"   Answer preview: {data['answer'][:100]}...")
                print(f"   Matched chunks: {len(data['matched_chunks'])}")
                print(f"   Metadata items: {len(data['metadata'])}")
                
                for i, metadata in enumerate(data['metadata'][:2], 1):
                    print(f"   Source {i}:")
                    print(f"     Chunk ID: {metadata.chunk_id}")
                    print(f"     Document: {metadata.document_name}")
                    print(f"     Page: {metadata.page_number}")
                    print(f"     Words: {metadata.word_count}")
                
                return True
            else:
                print(f"❌ RAG query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            return False
    
    def test_list_documents(self):
        """Test document listing"""
        print("\n📋 Testing document listing...")
        
        try:
            response = self.session.get(f"{self.base_url}/documents")
            if response.status_code == 200:
                data = response.json()
                print("✅ Document listing successful")
                print(f"   Total documents: {len(data['documents'])}")
                print(f"   Total chunks: {data['total_chunks']}")
                print(f"   Vector DB size: {data['vector_db_size']}")
                
                for doc in data['documents']:
                    print(f"   Document: {doc['document_name']}")
                    print(f"     Pages: {doc['total_pages']}")
                    print(f"     Chunks: {doc['total_chunks']}")
                    print(f"     Upload time: {doc['upload_timestamp']}")
                
                return True
            else:
                print(f"❌ Document listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Document listing error: {e}")
            return False
    
    def test_clear_documents(self):
        """Test document clearing"""
        print("\n🗑️ Testing document clearing...")
        
        try:
            response = self.session.delete(f"{self.base_url}/documents")
            if response.status_code == 200:
                data = response.json()
                print("✅ Document clearing successful")
                print(f"   Message: {data['message']}")
                print(f"   Timestamp: {data['timestamp']}")
                return True
            else:
                print(f"❌ Document clearing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Document clearing error: {e}")
            return False
    
    def run_full_test(self, pdf_path=None):
        """Run a complete test suite"""
        print("🚀 Starting Vector Database Test Suite")
        print("=" * 50)
        
        # Test health check
        if not self.test_health_check():
            print("❌ Health check failed. Make sure the server is running.")
            return False
        
        # Test PDF upload if provided
        if pdf_path:
            if not self.test_pdf_upload(pdf_path):
                print("❌ PDF upload failed.")
                return False
            
            # Test vector search
            test_queries = [
                "What is the main topic of this document?",
                "Summarize the key findings",
                "What are the important concepts mentioned?"
            ]
            
            for query in test_queries:
                self.test_vector_search(query)
                time.sleep(1)  # Small delay between requests
            
            # Test RAG query
            self.test_rag_query("What is the main conclusion of this document?")
            
            # Test document listing
            self.test_list_documents()
            
            # Test document clearing
            self.test_clear_documents()
        else:
            print("⚠️ No PDF provided for testing. Skipping upload and search tests.")
            print("   To test with a PDF, run: python test_vector_db.py path/to/your/document.pdf")
        
        print("\n✅ Test suite completed!")
        return True

def main():
    import sys
    
    base_url = "http://localhost:8000"
    pdf_path = None
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        base_url = sys.argv[2]
    
    tester = VectorDatabaseTester(base_url)
    tester.run_full_test(pdf_path)

if __name__ == "__main__":
    main()
