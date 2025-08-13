"""
Supervisor Agent

Orchestrates the multi-agent workflow and makes routing decisions.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Literal

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from graph.agent_state import AgentState
from config.legal_config import LegalAgentConfig

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent for workflow orchestration and routing.
    
    Role: Direct the workflow between agents and make routing decisions.
    """
    
    def __init__(self):
        config = LegalAgentConfig.get_agent_config("supervisor")
        super().__init__(
            agent_name="supervisor",
            model_name=config["model"],
            system_prompt=config["system_prompt"]
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute supervisor routing logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with routing decision
        """
        try:
            # Determine the next agent based on workflow state
            next_agent = self.route_to_next_agent(state)
            
            # Update state with routing decision
            state.next_agent = next_agent
            
            # Log the routing decision
            self.log_action(state, "route_decision", 
                          next_agent=next_agent,
                          iteration=state.iteration_count,
                          workflow_state=self._get_workflow_state(state))
            
            logger.info(f"Supervisor routing to: {next_agent} (iteration {state.iteration_count})")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e, "routing workflow")
    
    def route_to_next_agent(self, state: AgentState) -> str:
        """
        Determine the next agent in the workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next agent to execute
        """
        # Check if workflow is complete
        if state.is_complete:
            return "END"
        
        # Check for maximum iterations
        if state.iteration_count >= state.max_iterations:
            logger.warning(f"Maximum iterations ({state.max_iterations}) reached")
            # Force completion with available data
            if state.legal_analysis:
                return "legal_editor"
            else:
                state.mark_complete()
                return "END"
        
        # Initial routing - start with assistant
        if not state.structured_facts:
            return "legal_assistant"
        
        # If we have facts but no research, go to researcher
        if state.structured_facts and not state.retrieved_documents:
            return "legal_researcher"
        
        # If we have research but no analysis, go to lawyer
        if state.retrieved_documents and not state.legal_analysis:
            return "senior_lawyer"
        
        # If lawyer requested more research
        if hasattr(state, 'needs_more_research') and state.needs_more_research:
            state.needs_more_research = False  # Reset flag
            state.iteration_count += 1  # Increment iteration
            return "legal_researcher"
        
        # If we have analysis but no final document, go to editor
        if state.legal_analysis and not state.final_document:
            return "legal_editor"
        
        # If we have everything, we're done
        if state.final_document:
            state.mark_complete()
            return "END"
        
        # Fallback - shouldn't reach here
        logger.warning("Supervisor fallback routing to legal_assistant")
        return "legal_assistant"
    
    def should_continue(self, state: AgentState) -> bool:
        """
        Determine if the workflow should continue.
        
        Args:
            state: Current agent state
            
        Returns:
            True if workflow should continue, False otherwise
        """
        # Check completion flag
        if state.is_complete:
            return False
        
        # Check maximum iterations
        if state.iteration_count >= state.max_iterations:
            return False
        
        # Check if we have final document
        if state.final_document:
            return False
        
        # Check for terminal errors
        if len(state.errors) > 5:  # Too many errors
            logger.error("Too many errors encountered, terminating workflow")
            return False
        
        return True
    
    def make_routing_decision(self, state: AgentState) -> Dict[str, Any]:
        """
        Make intelligent routing decision with LLM assistance.
        
        Args:
            state: Current agent state
            
        Returns:
            Routing decision with reasoning
        """
        try:
            # Build routing prompt
            routing_prompt = self._build_routing_prompt(state)
            
            # Get LLM decision
            response = self.call_llm(routing_prompt, temperature=0.0, max_tokens=500)
            
            # Parse routing decision
            decision = self._parse_routing_decision(response)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in routing decision: {e}")
            # Fallback to rule-based routing
            return {
                "next_agent": self.route_to_next_agent(state),
                "reasoning": "Fallback routing due to LLM error",
                "confidence": 0.5
            }
    
    def _build_routing_prompt(self, state: AgentState) -> str:
        """
        Build prompt for routing decision.
        
        Args:
            state: Current agent state
            
        Returns:
            Routing prompt for LLM
        """
        prompt = f"""
You are the Supervisor agent in a legal multi-agent system. Analyze the current workflow state and decide the next agent to route to.

CURRENT STATE:
- User Query: {state.user_query}
- Iteration: {state.iteration_count}/{state.max_iterations}
- Structured Facts: {'✓' if state.structured_facts else '✗'}
- Retrieved Documents: {'✓' if state.retrieved_documents else '✗'}
- Legal Analysis: {'✓' if state.legal_analysis else '✗'}
- Final Document: {'✓' if state.final_document else '✗'}
- Completion Status: {'✓' if state.is_complete else '✗'}

AVAILABLE AGENTS:
1. legal_assistant - Query clarification and fact structuring
2. legal_researcher - Document retrieval and research
3. senior_lawyer - Legal analysis and reasoning
4. legal_editor - Final document drafting
5. END - Complete the workflow

ROUTING RULES:
- Start with legal_assistant if no structured facts
- Move to legal_researcher if facts exist but no documents
- Move to senior_lawyer if documents exist but no analysis
- Move to legal_editor if analysis exists but no final document
- Use END if final document exists or max iterations reached

Respond with:
NEXT_AGENT: [agent_name]
REASONING: [brief explanation]
CONFIDENCE: [0.0-1.0]
"""
        
        return prompt
    
    def _parse_routing_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse routing decision from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed routing decision
        """
        decision = {
            "next_agent": "legal_assistant",  # Default
            "reasoning": "Default routing",
            "confidence": 0.5
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('NEXT_AGENT:'):
                agent = line.replace('NEXT_AGENT:', '').strip()
                decision["next_agent"] = agent
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
                decision["reasoning"] = reasoning
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    decision["confidence"] = max(0.0, min(1.0, confidence))
                except:
                    pass
        
        return decision
    
    def _get_workflow_state(self, state: AgentState) -> str:
        """
        Get a string representation of the current workflow state.
        
        Args:
            state: Current agent state
            
        Returns:
            Workflow state description
        """
        components = []
        
        if state.structured_facts:
            components.append("facts")
        if state.retrieved_documents:
            components.append("research")
        if state.legal_analysis:
            components.append("analysis")
        if state.final_document:
            components.append("document")
        
        if not components:
            return "initial"
        
        return "_".join(components)
    
    def log_workflow_summary(self, state: AgentState) -> None:
        """
        Log a summary of the workflow execution.
        
        Args:
            state: Final agent state
        """
        summary = {
            "total_iterations": state.iteration_count,
            "agents_executed": len(state.agent_history),
            "documents_retrieved": len(state.retrieved_documents) if state.retrieved_documents else 0,
            "analysis_steps": len(state.legal_analysis) if state.legal_analysis else 0,
            "completion_status": "completed" if state.is_complete else "incomplete",
            "errors_encountered": len(state.errors),
            "final_document_length": len(state.final_document) if state.final_document else 0
        }
        
        logger.info(f"Workflow Summary: {summary}")
        
        # Log to state for debugging
        self.log_action(state, "workflow_summary", **summary)
