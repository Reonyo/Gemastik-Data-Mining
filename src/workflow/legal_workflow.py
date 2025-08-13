"""
Legal Multi-Agent Workflow Graph

Main workflow implementation using LangGraph for orchestrating the legal analysis system.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from graph.agent_state import AgentState
from agents.assistant import LegalAssistantAgent
from agents.researcher import LegalResearcherAgent
from agents.lawyer import SeniorLawyerAgent
from agents.editor import LegalEditorAgent
from agents.supervisor import SupervisorAgent

logger = logging.getLogger(__name__)


class LegalWorkflowGraph:
    """
    Main workflow graph for the legal multi-agent system.
    
    Implements a directed cyclic graph using LangGraph with:
    - 5 specialized agents
    - Conditional routing
    - Iterative loops
    - Maximum iteration limits
    - Error handling and fallbacks
    """
    
    def __init__(self):
        """Initialize the workflow graph with all agents."""
        # Initialize agents
        self.legal_assistant = LegalAssistantAgent()
        self.legal_researcher = LegalResearcherAgent()
        self.senior_lawyer = SeniorLawyerAgent()
        self.legal_editor = LegalEditorAgent()
        self.supervisor = SupervisorAgent()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Compile the graph
        self.app = self.graph.compile()
        
        logger.info("Legal workflow graph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(dict)
        
        # Add agent nodes
        workflow.add_node("legal_assistant", self._execute_legal_assistant)
        workflow.add_node("legal_researcher", self._execute_legal_researcher)
        workflow.add_node("senior_lawyer", self._execute_senior_lawyer)
        workflow.add_node("legal_editor", self._execute_legal_editor)
        workflow.add_node("supervisor", self._execute_supervisor)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "legal_assistant": "legal_assistant",
                "legal_researcher": "legal_researcher", 
                "senior_lawyer": "senior_lawyer",
                "legal_editor": "legal_editor",
                "END": END
            }
        )
        
        # Add edges back to supervisor from each agent
        workflow.add_edge("legal_assistant", "supervisor")
        workflow.add_edge("legal_researcher", "supervisor")
        workflow.add_edge("senior_lawyer", "supervisor")
        workflow.add_edge("legal_editor", "supervisor")
        
        return workflow
    
    def _dict_to_agent_state(self, state: dict) -> AgentState:
        """Convert dictionary state to AgentState object."""
        agent_state = AgentState(user_query=state.get('user_query', ''))
        
        # Copy dictionary values to AgentState attributes with proper conversions
        for key, value in state.items():
            if hasattr(agent_state, key):
                if key == 'retrieved_documents' and value:
                    # Convert dict documents back to RetrievedDocument objects
                    from graph.agent_state import RetrievedDocument
                    documents = []
                    for doc_dict in value:
                        if isinstance(doc_dict, dict):
                            doc = RetrievedDocument(
                                content=doc_dict.get('content', ''),
                                source=doc_dict.get('source', ''),
                                metadata=doc_dict.get('metadata', {}),
                                relevance_score=doc_dict.get('relevance_score', 0.0),
                                search_type=doc_dict.get('search_type', 'internal')
                            )
                            documents.append(doc)
                        else:
                            # Already a RetrievedDocument object
                            documents.append(doc_dict)
                    setattr(agent_state, key, documents)
                elif key == 'legal_analysis' and value:
                    # Convert dict analysis steps back to AnalysisStep objects
                    from graph.agent_state import AnalysisStep
                    steps = []
                    for step_dict in value:
                        if isinstance(step_dict, dict):
                            step = AnalysisStep(
                                step_number=step_dict.get('step_number', 0),
                                reasoning=step_dict.get('reasoning', ''),
                                legal_basis=step_dict.get('legal_basis', ''),
                                conclusion=step_dict.get('conclusion', ''),
                                confidence=step_dict.get('confidence', 0.0)
                            )
                            steps.append(step)
                        else:
                            # Already an AnalysisStep object
                            steps.append(step_dict)
                    setattr(agent_state, key, steps)
                else:
                    # For all other attributes, set directly
                    setattr(agent_state, key, value)
        
        return agent_state
    
    def _execute_legal_assistant(self, state: dict) -> dict:
        """Execute the Legal Assistant agent."""
        try:
            logger.info("Executing Legal Assistant agent")
            # Convert dict to AgentState
            agent_state = self._dict_to_agent_state(state)
            # Execute agent
            result_state = self.legal_assistant.execute(agent_state)
            # Convert back to dict
            return result_state.to_dict()
        except Exception as e:
            logger.error(f"Error in Legal Assistant: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Legal Assistant error: {str(e)}")
            return state
    
    def _execute_legal_researcher(self, state: dict) -> dict:
        """Execute the Legal Researcher agent."""
        try:
            logger.info("Executing Legal Researcher agent")
            # Convert dict to AgentState
            agent_state = self._dict_to_agent_state(state)
            # Execute agent
            result_state = self.legal_researcher.execute(agent_state)
            # Convert back to dict
            return result_state.to_dict()
        except Exception as e:
            logger.error(f"Error in Legal Researcher: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Legal Researcher error: {str(e)}")
            return state
    
    def _execute_senior_lawyer(self, state: dict) -> dict:
        """Execute the Senior Lawyer agent."""
        try:
            logger.info("Executing Senior Lawyer agent")
            # Convert dict to AgentState
            agent_state = self._dict_to_agent_state(state)
            # Execute agent
            result_state = self.senior_lawyer.execute(agent_state)
            # Convert back to dict
            return result_state.to_dict()
        except Exception as e:
            logger.error(f"Error in Senior Lawyer: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Senior Lawyer error: {str(e)}")
            return state
    
    def _execute_legal_editor(self, state: dict) -> dict:
        """Execute the Legal Editor agent."""
        try:
            logger.info("Executing Legal Editor agent")
            # Convert dict to AgentState
            agent_state = self._dict_to_agent_state(state)
            # Execute agent
            result_state = self.legal_editor.execute(agent_state)
            # Convert back to dict
            return result_state.to_dict()
        except Exception as e:
            logger.error(f"Error in Legal Editor: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Legal Editor error: {str(e)}")
            return state
    
    def _execute_supervisor(self, state: dict) -> dict:
        """Execute the Supervisor agent."""
        try:
            logger.info("Executing Supervisor agent")
            # Convert dict to AgentState
            agent_state = self._dict_to_agent_state(state)
            # Execute agent
            result_state = self.supervisor.execute(agent_state)
            # Convert back to dict
            return result_state.to_dict()
        except Exception as e:
            logger.error(f"Error in Supervisor: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Supervisor error: {str(e)}")
            # Fallback routing
            state['next_agent'] = "END"
            return state
    
    def _route_from_supervisor(self, state: dict) -> Literal["legal_assistant", "legal_researcher", "senior_lawyer", "legal_editor", "END"]:
        """
        Route from supervisor to the next agent.
        
        Args:
            state: Current agent state dict
            
        Returns:
            Next agent to execute
        """
        # Check if supervisor set the next agent
        if 'next_agent' in state and state['next_agent']:
            next_agent = state['next_agent']
            logger.info(f"Supervisor routing to: {next_agent}")
            return next_agent
        
        # Fallback routing logic
        if not state.get('structured_facts', []):
            return "legal_assistant"
        elif not state.get('retrieved_documents', []):
            return "legal_researcher"
        elif not state.get('legal_analysis', []):
            return "senior_lawyer"
        elif not state.get('final_document', ''):
            return "legal_editor"
        else:
            return "END"
    
    def run_workflow(self, user_query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Run the complete legal analysis workflow.
        
        Args:
            user_query: The legal question or issue to analyze
            max_iterations: Maximum number of iterations allowed
            
        Returns:
            Complete workflow results
        """
        try:
            # Initialize state as dictionary
            initial_state = {
                "user_query": user_query,
                "max_iterations": max_iterations,
                "structured_facts": [],
                "retrieved_documents": [],
                "legal_analysis": [],
                "final_document": "",
                "current_agent": "",
                "next_agent": "",
                "iteration_count": 0,
                "is_complete": False,
                "errors": []
            }
            
            logger.info(f"Starting legal workflow for query: {user_query[:100]}...")
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            # Prepare results
            results = self._prepare_results(final_state)
            
            logger.info("Legal workflow completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_query": user_query,
                "final_document": None,
                "structured_facts": [],
                "legal_analysis": [],
                "document_sections": {}
            }
    
    def _prepare_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare the final results from the workflow state.
        
        Args:
            state: Final workflow state
            
        Returns:
            Structured results dictionary
        """
        return {
            "success": state.get('is_complete', False) and not state.get('errors', []),
            "user_query": state.get('user_query', ''),
            "final_document": state.get('final_document', ''),
            "document_sections": state.get('document_sections', {}),
            "structured_facts": state.get('structured_facts', []),
            "legal_analysis": state.get('legal_analysis', []),
            "retrieved_documents": state.get('retrieved_documents', []),
            "reasoning_chain": state.get('reasoning_chain', ''),
            "workflow_metadata": {
                "iteration_count": state.get('iteration_count', 0),
                "max_iterations": state.get('max_iterations', 3),
                "agent_history": state.get('agent_history', []),
                "errors": state.get('errors', []),
                "is_complete": state.get('is_complete', False),
                "execution_time": state.get('execution_time', None)
            }
        }
    
    def run_workflow_streaming(self, user_query: str, max_iterations: int = 3):
        """
        Run the workflow with streaming updates.
        
        Args:
            user_query: The legal question or issue to analyze
            max_iterations: Maximum number of iterations allowed
            
        Yields:
            Workflow state updates
        """
        try:
            # Initialize state
            initial_state = AgentState(
                user_query=user_query,
                max_iterations=max_iterations
            )
            
            logger.info(f"Starting streaming legal workflow for query: {user_query[:100]}...")
            
            # Stream the workflow
            for state_update in self.app.stream(initial_state):
                yield {
                    "type": "state_update",
                    "state": state_update,
                    "timestamp": state_update.get("timestamp", None)
                }
            
            logger.info("Streaming legal workflow completed")
            
        except Exception as e:
            logger.error(f"Streaming workflow error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "user_query": user_query
            }


def create_legal_workflow() -> LegalWorkflowGraph:
    """
    Factory function to create a legal workflow graph.
    
    Returns:
        Initialized LegalWorkflowGraph
    """
    return LegalWorkflowGraph()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create workflow
    workflow = create_legal_workflow()
    
    # Example query
    test_query = """
    Saya ingin mendirikan perusahaan teknologi di Indonesia. 
    Apa saja persyaratan hukum yang harus dipenuhi dan prosedur yang harus diikuti?
    """
    
    # Run workflow
    results = workflow.run_workflow(test_query, max_iterations=3)
    
    # Print results
    print("=" * 80)
    print("LEGAL ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Query: {results['user_query']}")
    print(f"Success: {results['success']}")
    print(f"Iterations: {results['workflow_metadata']['iteration_count']}")
    
    if results['final_document']:
        print("\nFINAL DOCUMENT:")
        print("-" * 40)
        print(results['final_document'][:1000] + "..." if len(results['final_document']) > 1000 else results['final_document'])
    
    if results['structured_facts']:
        print(f"\nSTRUCTURED FACTS ({len(results['structured_facts'])}):")
        print("-" * 40)
        for i, fact in enumerate(results['structured_facts'][:5], 1):
            print(f"{i}. {fact}")
    
    if results['legal_analysis']:
        print(f"\nLEGAL ANALYSIS STEPS ({len(results['legal_analysis'])}):")
        print("-" * 40)
        for step in results['legal_analysis'][:3]:
            print(f"Step {step['step_number']}: {step['reasoning'][:200]}...")
