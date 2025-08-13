"""
Legal Researcher Agent

Specialized agent for hybrid retrieval using ChromaDB and Tavily API.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import ToolUsingAgent
from graph.agent_state import AgentState, RetrievedDocument
from config.legal_config import LegalAgentConfig
from data_processing.vector_indexer import LegalKnowledgeBaseIndexer
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class LegalResearcherAgent(ToolUsingAgent):
    """
    Legal Researcher Agent for hybrid information retrieval.
    
    Role: Perform hybrid retrieval using both internal knowledge base and external web search.
    Tools: ChromaDB vector database, Tavily API
    Constraints: Only retrieve and organize information, no analysis.
    """
    
    def __init__(self):
        config = LegalAgentConfig.get_agent_config("legal_researcher")
        super().__init__(
            agent_name="legal_researcher",
            model_name=config["model"],
            system_prompt=config["system_prompt"]
        )
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize the research tools."""
        try:
            # Initialize ChromaDB indexer
            self.vector_indexer = LegalKnowledgeBaseIndexer(
                model_name="BAAI/bge-m3",
                db_path=LegalAgentConfig.VECTOR_DB_PATH,
                collection_name=LegalAgentConfig.COLLECTION_NAME
            )
            self.vector_indexer.initialize_embedding_model()
            self.vector_indexer.initialize_vector_database()
            
            # Initialize Tavily client
            self.tavily_client = TavilyClient(api_key=LegalAgentConfig.TAVILY_API_KEY)
            
            # Register tools
            self.register_tool("internal_search", self._search_internal)
            self.register_tool("external_search", self._search_external)
            
            logger.info("Legal researcher tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize researcher tools: {e}")
            raise
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the legal researcher's main functionality.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with retrieved documents
        """
        try:
            # Check if we have structured facts to research
            if not state.structured_facts:
                state.add_error("No structured facts available for research", self.agent_name)
                return state
            
            # Create search queries from structured facts
            search_queries = self._generate_search_queries(state.structured_facts)
            
            # Perform internal search
            internal_docs = []
            for query in search_queries:
                docs = self.use_tool("internal_search", state, query=query)
                internal_docs.extend(docs)
            
            # Perform external search
            external_docs = []
            for query in search_queries[:3]:  # Limit external searches
                docs = self.use_tool("external_search", state, query=query)
                external_docs.extend(docs)
            
            # Add documents to state
            for doc in internal_docs:
                state.add_retrieved_document(doc)
            
            for doc in external_docs:
                state.add_retrieved_document(doc)
            
            # Log research results
            self.log_action(state, "research_complete", 
                          internal_docs=len(internal_docs),
                          external_docs=len(external_docs),
                          total_queries=len(search_queries))
            
            # Set next agent to senior lawyer
            state.set_next_agent("senior_lawyer")
            
            logger.info(f"Retrieved {len(internal_docs)} internal and {len(external_docs)} external documents")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e, "conducting legal research")
    
    def _generate_search_queries(self, structured_facts: List[str]) -> List[str]:
        """
        Generate search queries from structured facts.
        
        Args:
            structured_facts: List of structured facts
            
        Returns:
            List of search queries
        """
        queries = []
        
        for fact in structured_facts:
            # Extract key terms from each fact
            if "[Facts]" in fact:
                # Extract factual elements
                query = fact.replace("[Facts]", "").strip()
                queries.append(query)
            elif "[Areas]" in fact:
                # Extract legal areas
                query = fact.replace("[Areas]", "").strip()
                queries.append(f"law {query}")
            elif "[Questions]" in fact:
                # Extract specific questions
                query = fact.replace("[Questions]", "").strip()
                queries.append(query)
        
        # Remove duplicates and empty queries
        queries = list(set([q for q in queries if q]))
        
        # Limit to reasonable number
        return queries[:5]
    
    def _search_internal(self, query: str) -> List[RetrievedDocument]:
        """
        Search the internal ChromaDB knowledge base.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents
        """
        try:
            results = self.vector_indexer.search_similar_documents(
                query, 
                n_results=LegalAgentConfig.MAX_DOCUMENTS_PER_SEARCH
            )
            
            documents = []
            if 'documents' in results and results['documents']:
                docs = results['documents'][0]  # Assuming single query
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc_content in enumerate(docs):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else None
                    
                    doc = RetrievedDocument(
                        content=doc_content,
                        source=metadata.get('source_document', 'internal_db'),
                        metadata=metadata,
                        relevance_score=1 - distance if distance else None,
                        search_type="internal"
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Internal search failed for query '{query}': {e}")
            return []
    
    def _search_external(self, query: str) -> List[RetrievedDocument]:
        """
        Search external sources using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents
        """
        try:
            results = self.tavily_client.search(
                query=query,
                search_depth=LegalAgentConfig.TAVILY_CONFIG["search_depth"],
                include_domains=LegalAgentConfig.TAVILY_CONFIG["include_domains"],
                exclude_domains=LegalAgentConfig.TAVILY_CONFIG["exclude_domains"],
                max_results=LegalAgentConfig.TAVILY_CONFIG["max_results"]
            )
            
            documents = []
            for result in results.get('results', []):
                doc = RetrievedDocument(
                    content=result.get('content', ''),
                    source=result.get('url', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'published_date': result.get('published_date', ''),
                        'score': result.get('score', 0)
                    },
                    relevance_score=result.get('score', 0),
                    search_type="external"
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"External search failed for query '{query}': {e}")
            return []