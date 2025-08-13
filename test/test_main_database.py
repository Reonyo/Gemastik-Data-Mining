"""
Test and Populate Main Vector Database

Check the main vector database used by Multi-Agent system and populate it if empty.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.vector_indexer import LegalKnowledgeBaseIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_main_database():
    """Check the main vector database used by Multi-Agent system."""
    
    print("🔍 CHECKING MAIN VECTOR DATABASE")
    print("="*50)
    
    try:
        # Initialize indexer with same settings as Multi-Agent system
        indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/vector_db",  # Main database path
            collection_name="legal_knowledge_base"  # Main collection name
        )
        
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        # Check document count
        count = indexer.collection.count()
        print(f"📊 Main database document count: {count}")
        
        if count == 0:
            print("❌ Main database is EMPTY - no documents indexed")
            return False
        else:
            print(f"✅ Main database has {count} documents")
            
            # Test search
            print("🔍 Testing search on main database...")
            results = indexer.search_similar_documents("hukum Indonesia", n_results=2)
            
            if results and 'documents' in results and results['documents']:
                print(f"✅ Search successful! Found {len(results['documents'][0])} results")
                return True
            else:
                print("❌ Search failed - no results returned")
                return False
                
    except Exception as e:
        print(f"❌ Error checking main database: {e}")
        return False

def copy_test_to_main():
    """Copy documents from test database to main database."""
    
    print("\n🔄 COPYING TEST DATA TO MAIN DATABASE")
    print("="*50)
    
    try:
        # Initialize test database (source)
        test_indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/test_vector_db_single",
            collection_name="test_single_doc"
        )
        test_indexer.initialize_embedding_model()
        test_indexer.initialize_vector_database()
        
        # Initialize main database (destination)
        main_indexer = LegalKnowledgeBaseIndexer(
            model_name="BAAI/bge-m3",
            db_path="data/vector_db",
            collection_name="legal_knowledge_base"
        )
        main_indexer.initialize_embedding_model()
        main_indexer.initialize_vector_database()
        
        # Get documents from test database
        test_count = test_indexer.collection.count()
        print(f"📋 Test database has {test_count} documents")
        
        if test_count == 0:
            print("❌ Test database is empty - nothing to copy")
            return False
        
        # Get all documents from test database
        test_results = test_indexer.collection.get()
        
        if not test_results['documents']:
            print("❌ Failed to retrieve documents from test database")
            return False
        
        print(f"✅ Retrieved {len(test_results['documents'])} documents from test")
        
        # Add documents to main database
        main_indexer.collection.add(
            documents=test_results['documents'],
            metadatas=test_results['metadatas'],
            ids=test_results['ids'],
            embeddings=test_results['embeddings']
        )
        
        # Verify copy
        main_count = main_indexer.collection.count()
        print(f"✅ Main database now has {main_count} documents")
        
        # Test search on main database
        print("🔍 Testing search on populated main database...")
        results = main_indexer.search_similar_documents("PMK pajak", n_results=2)
        
        if results and 'documents' in results and results['documents']:
            print(f"✅ Search successful! Found {len(results['documents'][0])} results")
            print(f"📋 Sample result: {results['documents'][0][0][:100]}...")
            return True
        else:
            print("❌ Search failed even after copying")
            return False
            
    except Exception as e:
        print(f"❌ Error copying to main database: {e}")
        logger.exception("Full error details:")
        return False

def main():
    """Check and fix main vector database."""
    
    print("🚀 MAIN VECTOR DATABASE DIAGNOSIS & FIX")
    print("="*60)
    print("Purpose: Ensure Multi-Agent system has populated vector database")
    print("="*60)
    
    # Step 1: Check main database
    main_working = check_main_database()
    
    if not main_working:
        print("\n⚠️ Main database is empty or broken")
        
        # Step 2: Copy from test database
        copy_success = copy_test_to_main()
        
        if copy_success:
            print("\n✅ FIXED: Main database populated successfully")
            
            # Step 3: Final verification
            main_working = check_main_database()
        else:
            print("\n❌ FAILED: Could not populate main database")
    
    print("\n" + "="*60)
    print("🎯 MAIN DATABASE STATUS")
    print("="*60)
    
    if main_working:
        print("✅ SUCCESS: Main vector database is ready for Multi-Agent system!")
        print("💡 Internal search should now work properly")
    else:
        print("❌ FAILED: Main vector database still needs fixes")
        print("💡 Multi-Agent system will rely on external search only")
        
    return main_working

if __name__ == "__main__":
    main()
