"""
Test Vector Database Indexing - LIMITED DOCUMENTS

Standard test to index only a few documents for testing.
This avoids computation overhead while testing the system.
"""

import logging
import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.vector_indexer import LegalKnowledgeBaseIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_limited_indexing():
    """Index only a few documents for testing."""
    
    print("🧪 Testing Vector Database - LIMITED INDEXING")
    print("="*60)
    
    # Use test vector database
    test_db_path = "data/test_vector_db"
    
    try:
        # Initialize indexer
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path=test_db_path,
            collection_name="test_legal_docs"
        )
        
        # Select only 3-5 documents for testing (to limit computation)
        test_documents = [
            "data/raw_legal_docs/Undang-Undang_No.1_Tahun_2004.pdf",
            "data/raw_legal_docs/Undang-Undang_No.8_Tahun_1995.pdf", 
            "data/raw_legal_docs/kp1831998.pdf",
            "data/raw_legal_docs/1_PMK.05_2021.pdf"
        ]
        
        print(f"📄 Indexing {len(test_documents)} documents for testing...")
        
        # Check which files exist
        existing_docs = []
        for doc_path in test_documents:
            if os.path.exists(doc_path):
                existing_docs.append(doc_path)
                print(f"  ✅ Found: {Path(doc_path).name}")
            else:
                print(f"  ❌ Missing: {Path(doc_path).name}")
        
        if not existing_docs:
            print("❌ No documents found for indexing")
            return False
        
        # Index the documents
        print(f"\n🔄 Starting indexing process...")
        success_count = 0
        
        for doc_path in existing_docs:
            try:
                result = indexer.index_document(doc_path)
                if result:
                    success_count += 1
                    print(f"  ✅ Indexed: {Path(doc_path).name}")
                else:
                    print(f"  ❌ Failed: {Path(doc_path).name}")
            except Exception as e:
                print(f"  ❌ Error with {Path(doc_path).name}: {e}")
        
        print(f"\n📊 INDEXING RESULTS:")
        print(f"✅ Successfully indexed: {success_count}/{len(existing_docs)} documents")
        
        # Test search
        if success_count > 0:
            print(f"\n🔍 Testing search functionality...")
            results = indexer.search("hukum Indonesia asas legalitas", k=2)
            print(f"✅ Search returned {len(results)} results")
            
            if results:
                print(f"📋 Sample result: {results[0]['metadata']['source'][:50]}...")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Indexing test failed: {e}")
        return False

def test_search_functionality():
    """Test both internal and external search."""
    
    print("\n🔍 Testing Search Functionality")
    print("="*40)
    
    test_db_path = "data/test_vector_db"
    
    try:
        # Test internal search
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3", 
            db_path=test_db_path,
            collection_name="test_legal_docs"
        )
        
        search_query = "asas legalitas hukum pidana Indonesia"
        print(f"🔍 Testing internal search: '{search_query}'")
        
        results = indexer.search(search_query, k=3)
        print(f"📊 Internal search results: {len(results)} documents found")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['metadata']['source'][:60]}... (score: {result['score']:.3f})")
        else:
            print("  ⚠️ No internal documents found - will trigger external search")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Run limited indexing test."""
    
    print("🚀 VECTOR DATABASE TESTING - LIMITED COMPUTATION")
    print("="*70)
    print("Purpose: Index only 3-5 documents for efficient testing")
    print("Expected: Working vector search with minimal computation")
    print("="*70)
    
    # Test 1: Limited indexing
    indexing_success = test_limited_indexing()
    
    # Test 2: Search functionality  
    search_success = test_search_functionality()
    
    print("\n" + "="*70)
    print("🎯 VECTOR DATABASE TEST SUMMARY")
    print("="*70)
    print(f"✅ Limited Indexing: {'PASSED' if indexing_success else 'FAILED'}")
    print(f"✅ Search Functionality: {'PASSED' if search_success else 'FAILED'}")
    
    if indexing_success and search_success:
        print("\n🎉 Vector database is READY for testing!")
        print("💡 Use this test database for Multi-Agent system testing")
    else:
        print("\n⚠️ Vector database needs fixes")
        
    return indexing_success and search_success

if __name__ == "__main__":
    main()
