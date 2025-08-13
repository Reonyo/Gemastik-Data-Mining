"""
Legal Knowledge Base Vector Indexing Pipeline

This module creates vector embeddings for legal documents using BAAI/bge-m3 model
and stores them in ChromaDB for efficient semantic search and retrieval.

Author: Assistant
Date: August 12, 2025
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalKnowledgeBaseIndexer:
    """
    Handles embedding generation and vector database indexing for legal documents.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        db_path: str = "data/vector_db",
        collection_name: str = "legal_knowledge_base"
    ):
        """
        Initialize the indexer with embedding model and database configuration.
        
        Args:
            model_name: HuggingFace model identifier for embeddings
            db_path: Path to persist ChromaDB database
            collection_name: Name of the collection in ChromaDB
        """
        self.model_name = model_name
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Legal Knowledge Base Indexer")
        logger.info(f"Model: {model_name}")
        logger.info(f"Database path: {db_path}")
        logger.info(f"Collection: {collection_name}")
    
    def initialize_embedding_model(self) -> None:
        """Initialize the BAAI/bge-m3 embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load the sentence transformer model
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=device,
                trust_remote_code=True
            )
            
            # Set model to evaluation mode for inference
            self.embedding_model.eval()
            
            logger.info("✅ Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def initialize_vector_database(self) -> None:
        """Initialize ChromaDB client and collection."""
        logger.info("Initializing ChromaDB vector database")
        
        try:
            # Initialize ChromaDB client with persistence
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"📖 Found existing collection: {self.collection_name}")
                
                # Get collection stats
                count = self.collection.count()
                logger.info(f"Collection contains {count} documents")
                
            except Exception:
                # Create new collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Legal documents knowledge base with BAAI/bge-m3 embeddings"}
                )
                logger.info(f"🆕 Created new collection: {self.collection_name}")
            
            logger.info("✅ Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def generate_document_id(self, metadata: Dict[str, Any], content: str) -> str:
        """
        Generate a unique document ID based on metadata and content.
        
        Args:
            metadata: Document metadata (already cleaned of None values)
            content: Document content
            
        Returns:
            Unique document ID
        """
        # Create a unique identifier from source, pasal, ayat, and content hash
        source = metadata.get("source_document", "unknown")
        pasal = metadata.get("pasal", "")
        ayat = metadata.get("ayat", "")
        chunk_type = metadata.get("chunk_type", "")
        
        # Ensure all values are strings and not None
        source = str(source) if source else "unknown"
        pasal = str(pasal) if pasal else ""
        ayat = str(ayat) if ayat else ""
        chunk_type = str(chunk_type) if chunk_type else ""
        
        # Create content hash to ensure uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Combine into unique ID, removing any double underscores
        doc_id = f"{source}_{pasal}_{ayat}_{chunk_type}_{content_hash}"
        # Clean up consecutive underscores
        doc_id = "_".join(part for part in doc_id.split("_") if part)
        
        return doc_id
    
    def embed_text_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized. Call initialize_embedding_model() first.")
        
        embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for the batch
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    normalize_embeddings=True,  # Normalize for cosine similarity
                    show_progress_bar=False
                )
                
                # Convert to list format for ChromaDB
                batch_embeddings_list = batch_embeddings.tolist()
                embeddings.extend(batch_embeddings_list)
                
            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                raise
        
        return embeddings
    
    def load_knowledge_base(self, jsonl_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load legal knowledge base from JSONL file.
        
        Args:
            jsonl_path: Path to the knowledge base JSONL file
            limit: Optional limit on number of documents to load (for testing)
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Loading knowledge base from: {jsonl_path}")
        
        documents = []
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if limit and len(documents) >= limit:
                        break
                        
                    try:
                        doc = json.loads(line.strip())
                        documents.append(doc)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"✅ Loaded {len(documents)} documents")
            
            if limit:
                logger.info(f"📝 Limited to {limit} documents for testing")
                
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge base: {e}")
            raise
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        embedding_batch_size: int = 32
    ) -> None:
        """
        Index documents in the vector database.
        
        Args:
            documents: List of document dictionaries from JSONL
            batch_size: Number of documents to process per batch for database insertion
            embedding_batch_size: Number of texts to embed at once
        """
        logger.info(f"Indexing {len(documents)} documents in ChromaDB")
        
        if not self.collection:
            raise ValueError("Vector database not initialized. Call initialize_vector_database() first.")
        
        # Process documents in batches
        total_processed = 0
        
        with tqdm(total=len(documents), desc="Indexing documents") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                batch_texts = []
                batch_metadatas = []
                batch_ids = []
                
                for doc in batch:
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    
                    # Skip empty content
                    if not content.strip():
                        continue
                    
                    # Clean metadata - remove None values and convert all to strings
                    clean_metadata = {}
                    for key, value in metadata.items():
                        if value is not None:
                            # Convert to string to ensure compatibility with ChromaDB
                            clean_metadata[key] = str(value)
                    
                    # Generate unique document ID
                    doc_id = self.generate_document_id(clean_metadata, content)
                    
                    batch_texts.append(content)
                    batch_metadatas.append(clean_metadata)
                    batch_ids.append(doc_id)
                
                if not batch_texts:
                    continue
                
                try:
                    # Generate embeddings for the batch
                    embeddings = self.embed_text_batch(batch_texts, embedding_batch_size)
                    
                    # Add to ChromaDB collection
                    self.collection.add(
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids,
                        embeddings=embeddings
                    )
                    
                    total_processed += len(batch_texts)
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
        
        logger.info(f"✅ Successfully indexed {total_processed} documents")
        
        # Verify collection count
        final_count = self.collection.count()
        logger.info(f"📊 Final collection count: {final_count}")
    
    def search_similar_documents(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the knowledge base.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Search results with documents, metadata, and distances
        """
        if not self.collection:
            raise ValueError("Vector database not initialized.")
        
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand data distribution
            sample_results = self.collection.get(limit=100)
            
            # Analyze source documents
            sources = set()
            pasal_count = 0
            ayat_count = 0
            penjelasan_count = 0
            
            for metadata in sample_results.get("metadatas", []):
                if metadata:
                    sources.add(metadata.get("source_document", "unknown"))
                    if metadata.get("chunk_type") == "pasal":
                        pasal_count += 1
                    elif metadata.get("chunk_type") == "ayat":
                        ayat_count += 1
                    elif metadata.get("is_penjelasan"):
                        penjelasan_count += 1
            
            stats = {
                "total_documents": count,
                "unique_sources": len(sources),
                "sample_analysis": {
                    "pasal_chunks": pasal_count,
                    "ayat_chunks": ayat_count,
                    "penjelasan_chunks": penjelasan_count
                },
                "sample_sources": list(sources)[:10]  # First 10 sources
            }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function for testing the indexing pipeline."""
    
    # Configuration
    KNOWLEDGE_BASE_PATH = Path("data/processed/legal_knowledge_base.jsonl")
    VECTOR_DB_PATH = "data/vector_db"
    TEST_SAMPLE_SIZE = 1000  # For testing, process only first 1000 documents
    
    # Initialize indexer
    indexer = LegalKnowledgeBaseIndexer(
        model_name="BAAI/bge-m3",
        db_path=VECTOR_DB_PATH,
        collection_name="legal_knowledge_base"
    )
    
    try:
        # Step 1: Initialize embedding model
        logger.info("🚀 Step 1: Initializing embedding model...")
        indexer.initialize_embedding_model()
        
        # Step 2: Initialize vector database
        logger.info("🚀 Step 2: Initializing vector database...")
        indexer.initialize_vector_database()
        
        # Step 3: Load knowledge base (with limit for testing)
        logger.info("🚀 Step 3: Loading knowledge base...")
        documents = indexer.load_knowledge_base(
            KNOWLEDGE_BASE_PATH,
            limit=TEST_SAMPLE_SIZE
        )
        
        if not documents:
            logger.error("No documents loaded. Check the knowledge base file.")
            return
        
        # Step 4: Index documents
        logger.info("🚀 Step 4: Indexing documents...")
        start_time = time.time()
        
        indexer.index_documents(
            documents,
            batch_size=50,  # Smaller batches for testing
            embedding_batch_size=16
        )
        
        end_time = time.time()
        logger.info(f"⏱️ Indexing completed in {end_time - start_time:.2f} seconds")
        
        # Step 5: Get collection statistics
        logger.info("🚀 Step 5: Collection statistics...")
        stats = indexer.get_collection_stats()
        
        print("\n" + "="*80)
        print("LEGAL KNOWLEDGE BASE INDEXING RESULTS")
        print("="*80)
        print(f"Total documents indexed: {stats.get('total_documents', 'N/A')}")
        print(f"Unique source documents: {stats.get('unique_sources', 'N/A')}")
        
        sample_analysis = stats.get('sample_analysis', {})
        print(f"Sample analysis (first 100 docs):")
        print(f"  - Pasal chunks: {sample_analysis.get('pasal_chunks', 0)}")
        print(f"  - Ayat chunks: {sample_analysis.get('ayat_chunks', 0)}")
        print(f"  - Penjelasan chunks: {sample_analysis.get('penjelasan_chunks', 0)}")
        
        # Step 6: Test search functionality
        logger.info("🚀 Step 6: Testing search functionality...")
        
        test_queries = [
            "tarif layanan keuangan",
            "peraturan pemerintah",
            "badan layanan umum",
            "kementerian keuangan",
            "undang-undang advokat"
        ]
        
        print("\n" + "="*80)
        print("SEARCH TEST RESULTS")
        print("="*80)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = indexer.search_similar_documents(query, n_results=3)
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results.get('documents', [[]])[0],
                results.get('metadatas', [[]])[0],
                results.get('distances', [[]])[0]
            )):
                print(f"  Result {i+1} (similarity: {1-distance:.3f}):")
                print(f"    Source: {metadata.get('source_document', 'N/A')}")
                print(f"    Pasal: {metadata.get('pasal', 'N/A')}")
                print(f"    Content: {doc[:100]}...")
        
        logger.info("✅ Legal Knowledge Base indexing and testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
