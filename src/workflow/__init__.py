"""
Workflow Package

Main workflow implementation using LangGraph.
"""

from .legal_workflow import LegalWorkflowGraph, create_legal_workflow

__all__ = [
    "LegalWorkflowGraph",
    "create_legal_workflow"
]
