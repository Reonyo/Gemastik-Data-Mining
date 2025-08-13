"""
Shared AgentState for Legal Multi-Agent System

This module defines the central state object that serves as shared memory
passed between all agents in the legal analysis workflow.
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from knowledge base or web search."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: Optional[float] = None
    search_type: Literal["internal", "external"] = "internal"


@dataclass
class AnalysisStep:
    """Represents a step in the legal reasoning chain."""
    step_number: int
    reasoning: str
    legal_basis: str
    conclusion: str
    confidence: Optional[float] = None


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    agent_name: str
    action_type: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class AgentState:
    """
    Central state object that holds all information shared between agents.
    
    This object is passed between agents and maintains the complete workflow history,
    user query, retrieved documents, analysis steps, and current status.
    Enhanced with buffer memory and detailed workflow tracing for evaluation.
    """
    
    def __init__(self, user_query: str = "", max_iterations: int = 3):
        # User Input
        self.user_query: str = user_query
        self.structured_facts: List[str] = []
        
        # Retrieved Information
        self.retrieved_documents: List[RetrievedDocument] = []
        self.internal_documents: List[RetrievedDocument] = []
        self.external_documents: List[RetrievedDocument] = []
        
        # Analysis
        self.legal_analysis: List[AnalysisStep] = []
        self.reasoning_chain: str = ""
        self.legal_conclusions: List[str] = []
        
        # Final Output
        self.final_document: str = ""
        self.document_sections: Dict[str, str] = {}
        
        # Workflow Control
        self.current_agent: str = ""
        self.next_agent: str = ""
        self.iteration_count: int = 0
        self.max_iterations: int = max_iterations
        self.is_complete: bool = False
        self.needs_more_research: bool = False
        
        # Enhanced Workflow History for Evaluation
        self.agent_actions: List[AgentAction] = []
        self.workflow_log: List[str] = []
        self.agent_history: List[str] = []  # Track agent execution order
        
        # Buffer Memory System
        self.agent_memory: Dict[str, Dict[str, Any]] = {
            "legal_assistant": {"processed_queries": [], "fact_extractions": []},
            "legal_researcher": {"search_queries": [], "retrieved_chunks": []},
            "senior_lawyer": {"analysis_steps": [], "reasoning_chains": []},
            "legal_editor": {"draft_versions": [], "section_improvements": []},
            "supervisor": {"routing_decisions": [], "iteration_tracking": []}
        }
        
        # Detailed Debug Information for Evaluation
        self.debug_info: Dict[str, Any] = {
            "workflow_progress": [],
            "agent_internal_states": {},
            "decision_points": [],
            "performance_metrics": {}
        }
        
        # Quality Control
        self.confidence_scores: Dict[str, float] = {}
        self.completeness_check: Dict[str, bool] = {}
        
        # Error Handling
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_agent_action(self, agent_name: str, action_type: str, **details):
        """Record an action taken by an agent."""
        action = AgentAction(
            agent_name=agent_name,
            action_type=action_type,
            timestamp=datetime.now(),
            details=details
        )
        self.agent_actions.append(action)
        self.workflow_log.append(f"[{action.timestamp}] {agent_name}: {action_type}")
    
    def add_retrieved_document(self, document: RetrievedDocument):
        """Add a retrieved document to the appropriate collection."""
        self.retrieved_documents.append(document)
        if document.search_type == "internal":
            self.internal_documents.append(document)
        else:
            self.external_documents.append(document)
    
    def add_analysis_step(self, step: AnalysisStep):
        """Add a step to the legal analysis chain."""
        self.legal_analysis.append(step)
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iteration_count += 1
        self.add_agent_action("system", "iteration_increment", count=self.iteration_count)
    
    def is_max_iterations_reached(self) -> bool:
        """Check if maximum iterations have been reached."""
        return self.iteration_count >= self.max_iterations
    
    def set_next_agent(self, agent_name: str):
        """Set the next agent to be called."""
        self.next_agent = agent_name
        self.agent_history.append(agent_name)
        self.add_agent_action("supervisor", "route_to_agent", target_agent=agent_name)
    
    def mark_complete(self):
        """Mark the workflow as complete."""
        self.is_complete = True
        self.add_agent_action("system", "workflow_complete", 
                            total_iterations=self.iteration_count,
                            total_documents=len(self.retrieved_documents))
    
    # Buffer Memory Methods
    def add_to_agent_memory(self, agent_name: str, key: str, value: Any):
        """Add information to agent's buffer memory."""
        if agent_name in self.agent_memory:
            if key not in self.agent_memory[agent_name]:
                self.agent_memory[agent_name][key] = []
            self.agent_memory[agent_name][key].append(value)
    
    def get_agent_memory(self, agent_name: str, key: str = None) -> Any:
        """Retrieve information from agent's buffer memory."""
        if agent_name not in self.agent_memory:
            return {} if key is None else []
        
        if key is None:
            return self.agent_memory[agent_name]
        
        return self.agent_memory[agent_name].get(key, [])
    
    def update_debug_info(self, category: str, info: Dict[str, Any]):
        """Update debug information for evaluation tracking."""
        if category not in self.debug_info:
            self.debug_info[category] = []
        
        info["timestamp"] = datetime.now().isoformat()
        self.debug_info[category].append(info)
    
    def log_workflow_progress(self, agent_name: str, action: str, details: Dict[str, Any] = None):
        """Log detailed workflow progress for evaluation."""
        progress_entry = {
            "agent": agent_name,
            "action": action,
            "iteration": self.iteration_count,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.debug_info["workflow_progress"].append(progress_entry)
    
    def record_agent_internal_state(self, agent_name: str, internal_state: Dict[str, Any]):
        """Record agent's internal state for debugging."""
        self.debug_info["agent_internal_states"][agent_name] = {
            "timestamp": datetime.now().isoformat(),
            "state": internal_state,
            "iteration": self.iteration_count
        }
    
    def add_decision_point(self, agent_name: str, decision: str, reasoning: str, alternatives: List[str] = None):
        """Record decision points for qualitative analysis."""
        decision_entry = {
            "agent": agent_name,
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count
        }
        
        self.debug_info["decision_points"].append(decision_entry)
    
    def add_error(self, error_message: str, agent_name: str = "unknown"):
        """Add an error message."""
        self.errors.append(f"[{agent_name}] {error_message}")
        self.add_agent_action(agent_name, "error", message=error_message)
    
    def add_warning(self, warning_message: str, agent_name: str = "unknown"):
        """Add a warning message."""
        self.warnings.append(f"[{agent_name}] {warning_message}")
        self.add_agent_action(agent_name, "warning", message=warning_message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "user_query": self.user_query,
            "current_agent": self.current_agent,
            "next_agent": self.next_agent,
            "iteration_count": self.iteration_count,
            "is_complete": self.is_complete,
            "documents_retrieved": len(self.retrieved_documents),
            "internal_docs": len(self.internal_documents),
            "external_docs": len(self.external_documents),
            "analysis_steps": len(self.legal_analysis),
            "has_final_document": bool(self.final_document),
            "errors": len(self.errors),
            "warnings": len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "user_query": self.user_query,
            "structured_facts": self.structured_facts,
            "retrieved_documents": [
                {
                    "content": doc.content,
                    "source": doc.source,
                    "metadata": doc.metadata,
                    "relevance_score": doc.relevance_score,
                    "search_type": doc.search_type
                }
                for doc in self.retrieved_documents
            ],
            "legal_analysis": [
                {
                    "step_number": step.step_number,
                    "reasoning": step.reasoning,
                    "legal_basis": step.legal_basis,
                    "conclusion": step.conclusion,
                    "confidence": step.confidence
                }
                for step in self.legal_analysis
            ],
            "reasoning_chain": self.reasoning_chain,
            "legal_conclusions": self.legal_conclusions,
            "final_document": self.final_document,
            "document_sections": self.document_sections,
            "current_agent": self.current_agent,
            "next_agent": self.next_agent,
            "iteration_count": self.iteration_count,
            "is_complete": self.is_complete,
            "needs_more_research": self.needs_more_research,
            "workflow_log": self.workflow_log,
            "confidence_scores": self.confidence_scores,
            "completeness_check": self.completeness_check,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.get_summary()
        }
