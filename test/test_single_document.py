"""
Simple Vector Database Test - ONE DOCUMENT ONLY

Test that processes just ONE PDF to verify the pipeline works
and populate the vector database with minimal computation.
"""

import logging
import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.pdf_parser import LegalDocumentParser
from data_processing.vector_indexer import LegalKnowledgeBaseIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_single_document():
    """Process and index just ONE document to test the pipeline."""
    
    print("🧪 SINGLE DOCUMENT TEST - MINIMAL COMPUTATION")
    print("="*60)
    
    # Select ONE small document for testing
    test_document = "data/raw_legal_docs/1_PMK.05_2021.pdf"
    
    if not os.path.exists(test_document):
        print(f"❌ Document not found: {test_document}")
        return False
    
    try:
        print(f"📄 Processing ONE document: {Path(test_document).name}")
        
        # Step 1: Parse PDF
        parser = LegalDocumentParser(
            input_dir="data/raw_legal_docs",
            output_file="data/temp_parsed.jsonl"
        )
        processed_docs = parser.process_document(Path(test_document))
        
        print(f"✅ Extracted {len(processed_docs)} chunks from PDF")
        
        if not processed_docs:
            print("❌ No content extracted from PDF")
            return False
        
        # Step 2: Take only first 5 chunks to minimize computation
        limited_docs = processed_docs[:5]
        print(f"🔄 Using only {len(limited_docs)} chunks for testing")
        
        # Step 3: Initialize vector database  
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/test_vector_db_single",
            collection_name="test_single_doc"
        )
        
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        print("✅ Vector database initialized")
        
        # Step 4: Index the limited chunks
        print(f"🔄 Indexing {len(limited_docs)} chunks...")
        indexer.index_documents(limited_docs, batch_size=5, embedding_batch_size=2)
        
        print("✅ Documents indexed successfully!")
        
        # Step 5: Test search
        print("🔍 Testing search...")
        results = indexer.search_similar_documents("PMK pajak", n_results=2)
        
        if results and 'documents' in results and results['documents']:
            print(f"✅ Search successful! Found {len(results['documents'][0])} results")
            print(f"📋 Sample result: {results['documents'][0][0][:100]}...")
        else:
            print("⚠️ Search returned no results")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Full error details:")
        return False

def main():
    """Run single document test."""
    
    print("🚀 VECTOR DATABASE - SINGLE DOCUMENT TEST")
    print("="*60)
    print("Purpose: Minimal computation test with just one PDF")
    print("Expected: Working pipeline with 5 chunks maximum")
    print("="*60)
    
    success = test_single_document()
    
    print("\n" + "="*60)
    print("🎯 SINGLE DOCUMENT TEST RESULT")
    print("="*60)
    
    if success:
        print("✅ SUCCESS: Vector database pipeline is working!")
        print("💡 Ready for Multi-Agent system testing")
    else:
        print("❌ FAILED: Pipeline needs fixes")
        
    return success

if __name__ == "__main__":
    main()
