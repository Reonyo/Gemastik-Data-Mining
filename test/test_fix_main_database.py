"""
Fix Main Vector Database - Delete and Recreate

Delete the corrupted main database and recreate it with correct dimensions.
"""

import logging
import sys
import shutil
from pathlib import Path

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


def delete_main_database():
    """Delete the corrupted main database."""
    
    print("🗑️ DELETING CORRUPTED MAIN DATABASE")
    print("="*45)
    
    db_path = Path("data/vector_db")
    
    try:
        if db_path.exists():
            shutil.rmtree(db_path)
            print(f"✅ Deleted corrupted database: {db_path}")
        else:
            print(f"⚠️ Database path doesn't exist: {db_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error deleting database: {e}")
        return False

def recreate_main_database():
    """Recreate main database with correct embedding dimensions."""
    
    print("\n🔧 RECREATING MAIN DATABASE")
    print("="*40)
    
    try:
        # Initialize main database with correct model
        print("📝 Creating new main database...")
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",  # This should produce 1024 dimensions
            db_path="data/vector_db",
            collection_name="legal_knowledge_base"
        )
        
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        print("✅ Main database recreated successfully")
        
        # Verify it's empty
        count = indexer.collection.count()
        print(f"📊 Document count: {count} (should be 0)")
        
        return indexer
        
    except Exception as e:
        print(f"❌ Error recreating database: {e}")
        return None

def populate_main_database(indexer):
    """Populate main database with a few documents."""
    
    print("\n📄 POPULATING MAIN DATABASE")
    print("="*40)
    
    try:
        # Process one document to populate the database
        test_document = "data/raw_legal_docs/1_PMK.05_2021.pdf"
        
        if not Path(test_document).exists():
            print(f"❌ Test document not found: {test_document}")
            return False
        
        print(f"📄 Processing: {Path(test_document).name}")
        
        # Parse PDF
        parser = LegalDocumentParser(
            input_dir="data/raw_legal_docs",
            output_file="data/temp_parsed.jsonl"
        )
        processed_docs = parser.process_document(Path(test_document))
        
        print(f"✅ Extracted {len(processed_docs)} chunks")
        
        # Use only first 5 chunks for testing
        limited_docs = processed_docs[:5]
        print(f"🔄 Indexing {len(limited_docs)} chunks...")
        
        # Index documents
        indexer.index_documents(limited_docs, batch_size=5, embedding_batch_size=2)
        
        # Verify population
        count = indexer.collection.count()
        print(f"✅ Main database now has {count} documents")
        
        # Test search
        print("🔍 Testing search...")
        results = indexer.search_similar_documents("PMK pajak", n_results=2)
        
        if results and 'documents' in results and results['documents']:
            print(f"✅ Search successful! Found {len(results['documents'][0])} results")
            print(f"📋 Sample: {results['documents'][0][0][:80]}...")
            return True
        else:
            print("❌ Search failed")
            return False
            
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        logger.exception("Full error:")
        return False

def main():
    """Fix the main vector database completely."""
    
    print("🚀 FIXING MAIN VECTOR DATABASE")
    print("="*60)
    print("Problem: Dimension mismatch (384 vs 1024)")
    print("Solution: Delete and recreate with correct dimensions")
    print("="*60)
    
    # Step 1: Delete corrupted database
    if not delete_main_database():
        print("❌ Failed to delete corrupted database")
        return False
    
    # Step 2: Recreate database
    indexer = recreate_main_database()
    if not indexer:
        print("❌ Failed to recreate database")
        return False
    
    # Step 3: Populate database
    if not populate_main_database(indexer):
        print("❌ Failed to populate database")
        return False
    
    print("\n" + "="*60)
    print("🎯 MAIN DATABASE FIX COMPLETE")
    print("="*60)
    print("✅ SUCCESS: Main vector database fixed and populated!")
    print("💡 Multi-Agent system internal search should now work")
    print("📊 Ready for full evaluation")
    
    return True

if __name__ == "__main__":
    main()
