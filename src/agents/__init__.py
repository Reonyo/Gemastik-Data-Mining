"""
Agent Package

Contains all specialized agent implementations for the legal multi-agent system.
"""

from .assistant import LegalAssistantAgent
from .researcher import LegalResearcherAgent
from .lawyer import SeniorLawyerAgent
from .editor import LegalEditorAgent
from .supervisor import SupervisorAgent
from .base_agent import BaseAgent, ToolUsingAgent

__all__ = [
    "LegalAssistantAgent",
    "LegalResearcherAgent", 
    "SeniorLawyerAgent",
    "LegalEditorAgent",
    "SupervisorAgent",
    "BaseAgent",
    "ToolUsingAgent"
]
