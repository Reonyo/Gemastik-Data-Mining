"""
Test Vector Database Indexing with Real PDF Processing

This test processes a few PDFs and indexes them in the vector database.
Uses the real pipeline: PDF → Processed Docs → Vector Index
"""

import logging
import sys
from pathlib import Path
import json
import tempfile

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


def process_pdfs_to_jsonl(pdf_files, output_file):
    """Process PDF files and create JSONL output."""
    
    print(f"📄 Processing {len(pdf_files)} PDF files...")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize parser
        parser = LegalDocumentParser(
            input_dir=Path(pdf_files[0]).parent,  # Use parent directory
            output_file=output_file
        )
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            print(f"  🔄 Processing: {Path(pdf_file).name}")
            
            # Extract text from PDF
            text = parser.extract_text_from_pdf(Path(pdf_file))
            
            if text:
                # Parse the document into chunks
                chunks = parser.process_document(Path(pdf_file))
                all_chunks.extend(chunks)
                print(f"    ✅ Extracted {len(chunks)} chunks")
            else:
                print(f"    ❌ Failed to extract text")
        
        # Save chunks to JSONL
        print(f"\n💾 Saving {len(all_chunks)} chunks to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        return len(all_chunks)
        
    except Exception as e:
        print(f"❌ Error processing PDFs: {e}")
        return 0


def test_limited_pdf_indexing():
    """Test the complete pipeline: PDF → Processing → Indexing → Search."""
    
    print("🚀 COMPLETE PIPELINE TEST - LIMITED PDFs")
    print("="*60)
    
    # Select 3-4 PDF files for testing
    test_pdfs = [
        "data/raw_legal_docs/kp1831998.pdf",
        "data/raw_legal_docs/1_PMK.05_2021.pdf", 
        "data/raw_legal_docs/Undang-Undang_No.8_Tahun_1995.pdf"
    ]
    
    # Check which files exist
    existing_pdfs = []
    for pdf_path in test_pdfs:
        if Path(pdf_path).exists():
            existing_pdfs.append(pdf_path)
            print(f"  ✅ Found: {Path(pdf_path).name}")
        else:
            print(f"  ❌ Missing: {Path(pdf_path).name}")
    
    if not existing_pdfs:
        print("❌ No PDF files found for testing")
        return False
    
    # Step 1: Process PDFs to JSONL
    print(f"\n📋 STEP 1: Processing PDFs to structured data")
    print("-" * 40)
    
    temp_jsonl = "data/test_processed_docs.jsonl"
    chunk_count = process_pdfs_to_jsonl(existing_pdfs, temp_jsonl)
    
    if chunk_count == 0:
        print("❌ No chunks extracted from PDFs")
        return False
    
    print(f"✅ Processed {chunk_count} chunks from {len(existing_pdfs)} PDFs")
    
    # Step 2: Initialize vector database and indexer
    print(f"\n🔍 STEP 2: Indexing in vector database")
    print("-" * 40)
    
    try:
        # Initialize indexer
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/test_vector_db",
            collection_name="test_legal_docs"
        )
        
        # Initialize components
        print("  🔄 Initializing embedding model...")
        indexer.initialize_embedding_model()
        
        print("  🔄 Initializing vector database...")
        indexer.initialize_vector_database()
        
        # Load processed documents
        print("  🔄 Loading processed documents...")
        documents = indexer.load_knowledge_base(Path(temp_jsonl))
        print(f"  ✅ Loaded {len(documents)} documents")
        
        # Index documents
        print("  🔄 Indexing documents...")
        indexer.index_documents(documents, batch_size=50)
        
        # Check collection stats
        stats = indexer.get_collection_stats()
        print(f"  ✅ Vector database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_functionality():
    """Test search on the indexed documents."""
    
    print(f"\n🔍 STEP 3: Testing search functionality")
    print("-" * 40)
    
    try:
        # Initialize indexer
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/test_vector_db", 
            collection_name="test_legal_docs"
        )
        
        # Initialize components (should load existing database)
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        # Test various search queries
        test_queries = [
            "asas legalitas hukum pidana",
            "peraturan menteri keuangan",
            "tarif layanan keuangan",
            "pengenaan sanksi administratif"
        ]
        
        total_results = 0
        
        for query in test_queries:
            print(f"🔍 Searching: '{query}'")
            
            results = indexer.search_similar_documents(query, n_results=3)
            
            if results and 'documents' in results:
                result_count = len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0
                total_results += result_count
                
                print(f"  📊 Found {result_count} results")
                
                # Show first result if available
                if result_count > 0 and results['metadatas'] and results['metadatas'][0]:
                    first_meta = results['metadatas'][0][0]
                    source = first_meta.get('source', 'Unknown')
                    print(f"  📋 Top result from: {source}")
            else:
                print(f"  ❌ No results found")
        
        print(f"\n📊 Total results across all queries: {total_results}")
        return total_results > 0
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete limited PDF indexing test."""
    
    print("🎯 REAL PDF PROCESSING & VECTOR INDEXING TEST")
    print("="*70)
    print("Testing the complete pipeline with real legal documents")
    print("Expected: PDF processing → Vector indexing → Semantic search")
    print("="*70)
    
    # Test pipeline
    indexing_success = test_limited_pdf_indexing()
    
    if indexing_success:
        search_success = test_search_functionality()
    else:
        search_success = False
    
    # Summary
    print("\n" + "="*70)
    print("🎯 COMPLETE PIPELINE TEST SUMMARY")
    print("="*70)
    print(f"✅ PDF Processing & Indexing: {'PASSED' if indexing_success else 'FAILED'}")
    print(f"✅ Search Functionality: {'PASSED' if search_success else 'FAILED'}")
    
    if indexing_success and search_success:
        print("\n🎉 Complete pipeline is WORKING!")
        print("💡 Vector database now contains indexed legal documents")
        print("🚀 Ready to test Multi-Agent system with real data")
    else:
        print("\n⚠️ Pipeline needs debugging")
    
    return indexing_success and search_success

if __name__ == "__main__":
    main()
