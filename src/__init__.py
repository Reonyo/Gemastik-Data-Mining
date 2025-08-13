"""
Legal Multi-Agent System - Source Package

This package contains the complete implementation of a generative multi-agent system
for legal analysis using LangChain and LangGraph.
"""

from src.workflow.legal_workflow import create_legal_workflow, LegalWorkflowGraph
from src.graph.agent_state import AgentState
from src.config.legal_config import LegalAgentConfig

__version__ = "1.0.0"
__author__ = "Legal AI Team"

__all__ = [
    "create_legal_workflow",
    "LegalWorkflowGraph", 
    "AgentState",
    "LegalAgentConfig"
]
