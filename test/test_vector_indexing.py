"""
Test Vector Database Indexing - LIMITED DOCUMENTS

Standard test to index only a few documents for testing.
This avoids computation overhead while testing the system.
"""

import logging
import sys
from pathlib import Path
import json
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.vector_indexer import LegalKnowledgeBaseIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(source_file: Path, sample_file: Path, sample_size: int = 50):
    """
    Create a small sample of the knowledge base for testing.
    
    Args:
        source_file: Path to the full knowledge base JSONL file
        sample_file: Path where sample will be saved
        sample_size: Number of documents to include in sample
    """
    logger.info(f"Creating sample dataset with {sample_size} documents")
    
    sample_docs = []
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                
                try:
                    doc = json.loads(line.strip())
                    sample_docs.append(doc)
                except json.JSONDecodeError:
                    continue
        
        # Save sample
        with open(sample_file, 'w', encoding='utf-8') as f:
            for doc in sample_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Created sample file with {len(sample_docs)} documents: {sample_file}")
        return len(sample_docs)
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")
        raise


def test_embedding_model():
    """Test that the embedding model loads and works correctly."""
    logger.info("🧪 Testing embedding model initialization...")
    
    indexer = LegalKnowledgeBaseIndexer(
        model_name=Config.EMBEDDING_MODEL,
        db_path="data/test_vector_db",
        collection_name="test_collection"
    )
    
    try:
        # Initialize model
        indexer.initialize_embedding_model()
        
        # Test embedding generation
        test_texts = [
            "Pasal 1 tentang ketentuan umum dalam undang-undang ini",
            "Tarif layanan keuangan yang ditetapkan oleh menteri",
            "Badan layanan umum pada kementerian keuangan"
        ]
        
        embeddings = indexer.embed_text_batch(test_texts, batch_size=2)
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimension: {len(embeddings[0])}")
        
        # Verify embedding properties
        assert len(embeddings) == len(test_texts), "Number of embeddings doesn't match input texts"
        assert all(len(emb) == Config.EMBEDDING_DIMENSION for emb in embeddings), "Incorrect embedding dimension"
        
        logger.info("✅ Embedding model test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding model test failed: {e}")
        return False


def test_vector_database():
    """Test ChromaDB initialization and basic operations."""
    logger.info("🧪 Testing vector database operations...")
    
    indexer = LegalKnowledgeBaseIndexer(
        model_name=Config.EMBEDDING_MODEL,
        db_path="data/test_vector_db",
        collection_name="test_collection"
    )
    
    try:
        # Initialize components
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        # Test data
        test_docs = [
            {
                "content": "Pasal 1 ayat (1) tentang definisi dalam undang-undang advokat",
                "metadata": {
                    "source_document": "test_doc.pdf",
                    "pasal": "1",
                    "ayat": "1",
                    "is_penjelasan": False,
                    "chunk_type": "ayat"
                }
            },
            {
                "content": "Pasal 2 tentang kewajiban advokat dalam menjalankan profesi",
                "metadata": {
                    "source_document": "test_doc.pdf",
                    "pasal": "2",
                    "ayat": None,
                    "is_penjelasan": False,
                    "chunk_type": "pasal"
                }
            }
        ]
        
        # Index test documents
        indexer.index_documents(test_docs, batch_size=2, embedding_batch_size=2)
        
        # Test search
        results = indexer.search_similar_documents("kewajiban advokat", n_results=2)
        
        logger.info(f"✅ Search returned {len(results.get('documents', [[]])[0])} results")
        
        # Verify results structure
        assert 'documents' in results, "Missing documents in search results"
        assert 'metadatas' in results, "Missing metadata in search results"
        assert 'distances' in results, "Missing distances in search results"
        
        logger.info("✅ Vector database test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector database test failed: {e}")
        return False


def test_full_pipeline_with_sample():
    """Test the complete pipeline with a small sample of real data."""
    logger.info("🧪 Testing full pipeline with sample data...")
    
    # Configuration for test
    test_db_path = "data/test_vector_db_full"
    sample_file = Path("data/processed/sample_knowledge_base.jsonl")
    
    try:
        # Step 1: Create sample data
        Config.ensure_directories()
        Config.validate_files()
        
        sample_size = create_sample_data(
            Config.KNOWLEDGE_BASE_FILE,
            sample_file,
            sample_size=50  # Small sample for testing
        )
        
        if sample_size == 0:
            logger.error("No sample data created")
            return False
        
        # Step 2: Run indexing pipeline
        indexer = LegalKnowledgeBaseIndexer(
            model_name=Config.EMBEDDING_MODEL,
            db_path=test_db_path,
            collection_name="test_legal_kb"
        )
        
        # Initialize components
        indexer.initialize_embedding_model()
        indexer.initialize_vector_database()
        
        # Load and index sample data
        documents = indexer.load_knowledge_base(sample_file)
        indexer.index_documents(documents, batch_size=10, embedding_batch_size=5)
        
        # Get statistics
        stats = indexer.get_collection_stats()
        logger.info(f"Indexed documents: {stats.get('total_documents', 0)}")
        
        # Test various searches
        test_queries = [
            "tarif layanan",
            "undang-undang",
            "kementerian keuangan",
            "pasal"
        ]
        
        search_results = {}
        for query in test_queries:
            results = indexer.search_similar_documents(query, n_results=3)
            search_results[query] = len(results.get('documents', [[]])[0])
            logger.info(f"Query '{query}': {search_results[query]} results")
        
        # Verify we got results
        total_results = sum(search_results.values())
        assert total_results > 0, "No search results returned"
        
        logger.info("✅ Full pipeline test passed!")
        
        # Cleanup sample file
        if sample_file.exists():
            sample_file.unlink()
            logger.info("🧹 Cleaned up sample file")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        return False
    finally:
        # Cleanup sample file if it exists
        if 'sample_file' in locals() and sample_file.exists():
            sample_file.unlink()


def main():
    """Run all tests."""
    logger.info("🚀 Starting Legal Knowledge Base Indexing Tests")
    
    tests = [
        ("Embedding Model", test_embedding_model),
        ("Vector Database", test_vector_database),
        ("Full Pipeline", test_full_pipeline_with_sample)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info('='*60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info('='*60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Ready to process full dataset.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
