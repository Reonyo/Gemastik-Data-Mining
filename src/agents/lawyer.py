"""
Senior Lawyer Agent

Specialized agent for legal analysis and reasoning chain building.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import ToolUsingAgent
from graph.agent_state import AgentState, AnalysisStep
from config.legal_config import LegalAgentConfig

logger = logging.getLogger(__name__)


class SeniorLawyerAgent(ToolUsingAgent):
    """
    Senior Lawyer Agent for legal analysis and reasoning.
    
    Role: Analyze and synthesize all facts and retrieved documents to build 
          a comprehensive chain of legal reasoning.
    Tools: Python interpreter for complex calculations
    Authority: Can request additional research or provide final analysis
    """
    
    def __init__(self):
        config = LegalAgentConfig.get_agent_config("senior_lawyer")
        super().__init__(
            agent_name="senior_lawyer",
            model_name=config["model"],
            system_prompt=config["system_prompt"]
        )
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize the analysis tools."""
        try:
            # Register Python interpreter tool
            self.register_tool("python_interpreter", self._execute_python)
            
            logger.info("Senior lawyer tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize lawyer tools: {e}")
            raise
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the senior lawyer's main functionality.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with legal analysis
        """
        try:
            # Check if we have sufficient information
            if not state.structured_facts:
                state.add_error("No structured facts available for analysis", self.agent_name)
                return state
            
            if not state.retrieved_documents:
                state.add_warning("No retrieved documents available, proceeding with limited analysis", self.agent_name)
            
            # Build comprehensive analysis prompt
            analysis_prompt = self._build_analysis_prompt(state)
            
            # Call LLM for analysis
            response = self.call_llm(analysis_prompt, temperature=0.1, max_tokens=3000)
            
            # Parse the analysis response
            analysis_steps = self._parse_analysis_response(response)
            
            # Add analysis steps to state
            for step in analysis_steps:
                state.add_analysis_step(step)
            
            # Build reasoning chain
            state.reasoning_chain = self._build_reasoning_chain(analysis_steps)
            
            # Determine if more research is needed
            needs_more_research = self._assess_research_needs(response, state)
            
            if needs_more_research and not state.is_max_iterations_reached():
                # Request more research
                state.needs_more_research = True
                state.increment_iteration()
                state.set_next_agent("legal_researcher")
                
                self.log_action(state, "request_more_research", 
                              iteration=state.iteration_count)
                
                logger.info(f"Requesting additional research (iteration {state.iteration_count})")
            else:
                # Analysis complete, proceed to editor
                state.needs_more_research = False
                state.set_next_agent("legal_editor")
                
                self.log_action(state, "analysis_complete", 
                              steps_count=len(analysis_steps),
                              max_iterations_reached=state.is_max_iterations_reached())
                
                logger.info(f"Legal analysis complete with {len(analysis_steps)} steps")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e, "conducting legal analysis")
    
    def _build_analysis_prompt(self, state: AgentState) -> str:
        """
        Build comprehensive analysis prompt from state information.
        
        Args:
            state: Current agent state
            
        Returns:
            Analysis prompt for the LLM
        """
        prompt = f"""
Please conduct a comprehensive legal analysis based on the following information:

STRUCTURED FACTS:
{chr(10).join(f"- {fact}" for fact in state.structured_facts)}

RETRIEVED DOCUMENTS:
"""
        
        # Add internal documents
        if state.internal_documents:
            prompt += "\nINTERNAL LEGAL SOURCES:\n"
            for i, doc in enumerate(state.internal_documents[:5], 1):
                prompt += f"{i}. Source: {doc.source}\n"
                prompt += f"   Content: {doc.content[:500]}...\n"
                prompt += f"   Relevance: {doc.relevance_score:.2f}\n\n"
        
        # Add external documents
        if state.external_documents:
            prompt += "\nEXTERNAL LEGAL SOURCES:\n"
            for i, doc in enumerate(state.external_documents[:3], 1):
                prompt += f"{i}. Source: {doc.source}\n"
                prompt += f"   Content: {doc.content[:500]}...\n"
                prompt += f"   Relevance: {doc.relevance_score:.2f}\n\n"
        
        prompt += """
Please provide your analysis in the following structured format:

LEGAL ISSUES IDENTIFIED:
1. [Issue 1]
2. [Issue 2]
...

APPLICABLE LAWS AND PRECEDENTS:
1. [Law/Precedent 1 and how it applies]
2. [Law/Precedent 2 and how it applies]
...

LEGAL REASONING CHAIN:
Step 1: [Analysis step with reasoning]
Step 2: [Analysis step with reasoning]
...

POTENTIAL OUTCOMES:
1. [Outcome 1 with probability/confidence]
2. [Outcome 2 with probability/confidence]
...

CONFIDENCE LEVEL: [High/Medium/Low] - [Explanation]

ADDITIONAL RESEARCH NEEDS:
[Specify if more information is needed or if analysis is complete]

Base your analysis strictly on the provided facts and retrieved documents. 
Clearly distinguish between strong and weak legal arguments.
"""
        
        return prompt
    
    def _parse_analysis_response(self, response: str) -> List[AnalysisStep]:
        """
        Parse the analysis response into structured steps.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            List of analysis steps
        """
        steps = []
        lines = response.split('\n')
        
        current_section = None
        step_number = 1
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify sections
            if 'LEGAL REASONING CHAIN:' in line.upper():
                current_section = 'reasoning'
                continue
            elif any(header in line.upper() for header in 
                   ['LEGAL ISSUES', 'APPLICABLE LAWS', 'POTENTIAL OUTCOMES', 'CONFIDENCE LEVEL']):
                current_section = 'other'
                continue
            
            # Extract reasoning steps
            if current_section == 'reasoning' and (line.startswith('Step') or 
                                                  line.startswith(f'{step_number}.')):
                if current_content:
                    # Save previous step
                    step = AnalysisStep(
                        step_number=step_number - 1,
                        reasoning=' '.join(current_content),
                        legal_basis="Based on provided documents",
                        conclusion="See reasoning",
                        confidence=0.8
                    )
                    steps.append(step)
                    current_content = []
                
                current_content.append(line)
                step_number += 1
            elif current_section == 'reasoning':
                current_content.append(line)
        
        # Add final step if exists
        if current_content:
            step = AnalysisStep(
                step_number=step_number - 1,
                reasoning=' '.join(current_content),
                legal_basis="Based on provided documents",
                conclusion="See reasoning",
                confidence=0.8
            )
            steps.append(step)
        
        # If no structured steps found, create one from entire response
        if not steps:
            step = AnalysisStep(
                step_number=1,
                reasoning=response,
                legal_basis="Comprehensive analysis",
                conclusion="See full analysis",
                confidence=0.7
            )
            steps.append(step)
        
        return steps
    
    def _build_reasoning_chain(self, analysis_steps: List[AnalysisStep]) -> str:
        """
        Build a coherent reasoning chain from analysis steps.
        
        Args:
            analysis_steps: List of analysis steps
            
        Returns:
            Coherent reasoning chain text
        """
        if not analysis_steps:
            return ""
        
        chain_parts = []
        for step in analysis_steps:
            chain_parts.append(f"Step {step.step_number}: {step.reasoning}")
        
        return "\n\n".join(chain_parts)
    
    def _assess_research_needs(self, response: str, state: AgentState) -> bool:
        """
        Assess if additional research is needed based on the analysis.
        
        Args:
            response: Analysis response from LLM
            state: Current agent state
            
        Returns:
            True if more research is needed, False otherwise
        """
        # Check for explicit research requests in response
        research_indicators = [
            "more information needed",
            "additional research required",
            "insufficient information",
            "need more details",
            "require additional sources"
        ]
        
        response_lower = response.lower()
        for indicator in research_indicators:
            if indicator in response_lower:
                return True
        
        # Check if we have very few documents
        if len(state.retrieved_documents) < 3:
            return True
        
        # Check confidence level
        if "confidence level: low" in response_lower:
            return True
        
        return False
    
    def _execute_python(self, code: str) -> str:
        """
        Execute Python code for complex calculations.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result
        """
        try:
            # Basic Python execution (restricted for security)
            # In production, use a proper sandboxed environment
            allowed_modules = ['math', 'datetime', 'statistics']
            
            # Simple execution for basic calculations
            import math
            import datetime
            import statistics
            
            # Create restricted globals
            restricted_globals = {
                'math': math,
                'datetime': datetime,
                'statistics': statistics,
                '__builtins__': {}
            }
            
            exec(code, restricted_globals)
            return "Code executed successfully"
            
        except Exception as e:
            return f"Execution error: {str(e)}"