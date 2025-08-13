"""
Legal Assistant Agent

Specialized agent for clarifying and structuring user queries into actionable facts.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from graph.agent_state import AgentState
from config.legal_config import LegalAgentConfig

logger = logging.getLogger(__name__)


class LegalAssistantAgent(BaseAgent):
    """
    Legal Assistant Agent for query clarification and fact structuring.
    
    Role: Clarify and structure the initial user query into clear, actionable facts.
    Constraints: Must not provide any legal opinions or advice.
    """
    
    def __init__(self):
        config = LegalAgentConfig.get_agent_config("legal_assistant")
        super().__init__(
            agent_name="legal_assistant",
            model_name=config["model"],
            system_prompt=config["system_prompt"]
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the legal assistant's main functionality.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with structured facts
        """
        try:
            # Check if we have a user query
            if not state.user_query:
                state.add_error("No user query provided", self.agent_name)
                return state
            
            # Create prompt for structuring the query
            user_prompt = f"""
Please analyze the following legal query and structure it into clear, actionable facts:

USER QUERY:
{state.user_query}

Please provide your analysis in the following format:

KEY FACTS:
1. [Factual element 1]
2. [Factual element 2]
...

PARTIES INVOLVED:
- [Party 1 and their role]
- [Party 2 and their role]
...

LEGAL AREAS:
- [Area of law 1]
- [Area of law 2]
...

SPECIFIC QUESTIONS:
1. [Specific legal question 1]
2. [Specific legal question 2]
...

MISSING INFORMATION:
- [What additional facts might be needed]
- [What clarifications would be helpful]

Remember: Only structure and clarify the factual information. Do not provide legal opinions or analysis.
"""
            
            # Call LLM to structure the query
            response = self.call_llm(user_prompt, temperature=0.1)
            
            # Parse the response to extract structured facts
            structured_facts = self._parse_assistant_response(response)
            
            # Update state with structured facts
            state.structured_facts = structured_facts
            
            # Log the action
            self.log_action(state, "query_structured", 
                          facts_count=len(structured_facts),
                          response_length=len(response))
            
            # Set next agent to researcher
            state.set_next_agent("legal_researcher")
            
            logger.info(f"Structured {len(structured_facts)} facts from user query")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e, "structuring user query")
    
    def _parse_assistant_response(self, response: str) -> list[str]:
        """
        Parse the assistant's response to extract structured facts.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            List of structured facts
        """
        facts = []
        lines = response.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if 'KEY FACTS:' in line.upper():
                current_section = 'facts'
                continue
            elif 'PARTIES INVOLVED:' in line.upper():
                current_section = 'parties'
                continue
            elif 'LEGAL AREAS:' in line.upper():
                current_section = 'areas'
                continue
            elif 'SPECIFIC QUESTIONS:' in line.upper():
                current_section = 'questions'
                continue
            elif 'MISSING INFORMATION:' in line.upper():
                current_section = 'missing'
                continue
            
            # Extract facts from each section
            if current_section and (line.startswith('-') or line.startswith('1.') or 
                                  line.startswith('2.') or line.startswith('3.') or
                                  line.startswith('4.') or line.startswith('5.')):
                # Clean up the fact
                fact = line.lstrip('- 123456789.').strip()
                if fact:
                    facts.append(f"[{current_section.title()}] {fact}")
        
        # If no structured facts found, use the entire response
        if not facts:
            facts = [response]
        
        return facts