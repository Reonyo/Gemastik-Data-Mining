"""
Legal Editor Agent

Specialized agent for drafting final coherent legal documents.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from graph.agent_state import AgentState
from config.legal_config import LegalAgentConfig

logger = logging.getLogger(__name__)


class LegalEditorAgent(BaseAgent):
    """
    Legal Editor Agent for document drafting and formatting.
    
    Role: Draft the final, coherent legal document based on the Senior Lawyer's analysis.
    Constraints: Must not add any new analysis or interpretation.
    """
    
    def __init__(self):
        config = LegalAgentConfig.get_agent_config("legal_editor")
        super().__init__(
            agent_name="legal_editor",
            model_name=config["model"],
            system_prompt=config["system_prompt"]
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the legal editor's main functionality.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with final document
        """
        try:
            # Check if we have analysis to work with
            if not state.legal_analysis and not state.reasoning_chain:
                state.add_error("No legal analysis available for document drafting", self.agent_name)
                return state
            
            # Build document drafting prompt
            drafting_prompt = self._build_drafting_prompt(state)
            
            # Call LLM to draft the document
            response = self.call_llm(drafting_prompt, temperature=0.1, max_tokens=4000)
            
            # Process the response into structured document
            final_document = self._process_document_response(response)
            
            # Parse document sections
            document_sections = self._parse_document_sections(final_document)
            
            # Update state with final document
            state.final_document = final_document
            state.document_sections = document_sections
            state.mark_complete()
            
            # Log completion
            self.log_action(state, "document_complete", 
                          document_length=len(final_document),
                          sections_count=len(document_sections))
            
            logger.info(f"Final legal document drafted with {len(document_sections)} sections")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e, "drafting final document")
    
    def _build_drafting_prompt(self, state: AgentState) -> str:
        """
        Build document drafting prompt from analysis results.
        
        Args:
            state: Current agent state
            
        Returns:
            Drafting prompt for the LLM
        """
        prompt = f"""
Please draft a comprehensive legal document based on the following analysis:

ORIGINAL QUERY:
{state.user_query}

STRUCTURED FACTS:
{chr(10).join(f"- {fact}" for fact in state.structured_facts)}

LEGAL ANALYSIS STEPS:
"""
        
        # Add analysis steps
        for step in state.legal_analysis:
            prompt += f"\nStep {step.step_number}: {step.reasoning}\n"
            prompt += f"Legal Basis: {step.legal_basis}\n"
            prompt += f"Conclusion: {step.conclusion}\n"
        
        # Add reasoning chain if available
        if state.reasoning_chain:
            prompt += f"\nREASONING CHAIN:\n{state.reasoning_chain}\n"
        
        # Add document structure requirements
        prompt += """

Please draft a professional legal document with the following structure:

EXECUTIVE SUMMARY:
[Brief overview of the legal issue and key conclusions]

FACTS AND BACKGROUND:
[Organized presentation of the relevant facts]

LEGAL ISSUES:
[Clear identification of the legal questions to be addressed]

ANALYSIS AND REASONING:
[Detailed legal analysis based on the provided reasoning]

CONCLUSIONS AND RECOMMENDATIONS:
[Summary of findings and recommended actions]

ADDITIONAL CONSIDERATIONS:
[Any limitations, uncertainties, or additional factors to consider]

FORMATTING REQUIREMENTS:
- Use clear, professional legal language
- Maintain logical flow between sections
- Include proper legal terminology
- Present information objectively
- Do not add any new analysis or interpretation
- Only reorganize and present the provided analysis

Ensure the document is comprehensive, well-structured, and maintains the quality expected of professional legal work.
"""
        
        return prompt
    
    def _process_document_response(self, response: str) -> str:
        """
        Process and clean the document response.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Cleaned and formatted document
        """
        # Basic cleaning and formatting
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive whitespace
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Join lines with proper spacing
        document = '\n\n'.join(cleaned_lines)
        
        # Ensure proper section headers
        document = self._format_section_headers(document)
        
        return document
    
    def _format_section_headers(self, document: str) -> str:
        """
        Format section headers consistently.
        
        Args:
            document: Document text
            
        Returns:
            Document with formatted headers
        """
        headers = [
            "EXECUTIVE SUMMARY",
            "FACTS AND BACKGROUND", 
            "LEGAL ISSUES",
            "ANALYSIS AND REASONING",
            "CONCLUSIONS AND RECOMMENDATIONS",
            "ADDITIONAL CONSIDERATIONS"
        ]
        
        for header in headers:
            # Ensure headers are properly formatted
            document = document.replace(header + ":", f"\n{header}:\n")
            document = document.replace(header.lower() + ":", f"\n{header}:\n")
            document = document.replace(header.title() + ":", f"\n{header}:\n")
        
        return document
    
    def _parse_document_sections(self, document: str) -> Dict[str, str]:
        """
        Parse the document into sections.
        
        Args:
            document: Complete document text
            
        Returns:
            Dictionary of section name to content
        """
        sections = {}
        current_section = None
        current_content = []
        
        lines = document.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            if line.endswith(':') and line.upper() in [
                'EXECUTIVE SUMMARY:',
                'FACTS AND BACKGROUND:',
                'LEGAL ISSUES:',
                'ANALYSIS AND REASONING:',
                'CONCLUSIONS AND RECOMMENDATIONS:',
                'ADDITIONAL CONSIDERATIONS:'
            ]:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace(':', '').strip()
                current_content = []
            else:
                # Add content to current section
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no sections found, create a single section
        if not sections:
            sections['COMPLETE DOCUMENT'] = document
        
        return sections