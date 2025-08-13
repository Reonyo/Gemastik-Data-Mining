"""
Configuration file for Legal Knowledge Base Vector Indexing
"""

from pathlib import Path

class Config:
    """Configuration settings for the legal knowledge base indexer."""
    
    # Paths
    DATA_DIR = Path("data")
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    
    # Input files
    KNOWLEDGE_BASE_FILE = PROCESSED_DATA_DIR / "legal_knowledge_base.jsonl"
    UPA_DATASET_FILE = PROCESSED_DATA_DIR / "upa_280_mcq_dataset.json"
    
    # Embedding model configuration
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DIMENSION = 1024  # BGE-M3 produces 1024-dimensional embeddings
    
    # Database configuration
    COLLECTION_NAME = "legal_knowledge_base"
    UPA_COLLECTION_NAME = "upa_mcq_questions"
    
    # Processing parameters
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_EMBEDDING_BATCH_SIZE = 32
    TEST_SAMPLE_SIZE = 1000  # For testing with limited data
    
    # Search parameters
    DEFAULT_SEARCH_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for relevant results
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_files(cls):
        """Validate that required input files exist."""
        missing_files = []
        
        if not cls.KNOWLEDGE_BASE_FILE.exists():
            missing_files.append(str(cls.KNOWLEDGE_BASE_FILE))
        
        if missing_files:
            raise FileNotFoundError(
                f"Required input files not found: {', '.join(missing_files)}"
            )
        
        return True
