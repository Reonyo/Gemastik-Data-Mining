"""
Base Agent Class for Legal Multi-Agent System

This module provides the base agent class that all specialized agents inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from groq import Groq
from graph.agent_state import AgentState
from config.legal_config import LegalAgentConfig

# Set up logging
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the legal multi-agent system.
    
    Provides common functionality for LLM interaction, error handling,
    and state management.
    """
    
    def __init__(self, agent_name: str, model_name: str, system_prompt: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Name of the agent
            model_name: Name of the LLM model to use
            system_prompt: System prompt defining the agent's role
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        
        # Initialize Groq client
        try:
            self.client = Groq(api_key=LegalAgentConfig.GROQ_API_KEY)
            logger.info(f"Initialized {agent_name} with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client for {agent_name}: {e}")
            raise
    
    def call_llm(self, user_prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            user_prompt: The user prompt to send to the LLM
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            The LLM's response
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {self.agent_name}: {e}")
            raise
    
    def update_state(self, state: AgentState, **updates) -> AgentState:
        """
        Update the agent state with new information.
        
        Args:
            state: Current agent state
            **updates: Key-value pairs to update in the state
            
        Returns:
            Updated agent state
        """
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning(f"Attempted to set unknown state attribute: {key}")
        
        state.current_agent = self.agent_name
        return state
    
    def log_action(self, state: AgentState, action_type: str, **details):
        """
        Log an action taken by this agent.
        
        Args:
            state: Current agent state
            action_type: Type of action taken
            **details: Additional details about the action
        """
        state.add_agent_action(self.agent_name, action_type, **details)
        logger.info(f"{self.agent_name} performed action: {action_type}")
    
    def handle_error(self, state: AgentState, error: Exception, context: str = "") -> AgentState:
        """
        Handle an error that occurred during agent execution.
        
        Args:
            state: Current agent state
            error: The exception that occurred
            context: Additional context about when the error occurred
            
        Returns:
            Updated agent state with error information
        """
        error_message = f"Error in {context}: {str(error)}" if context else str(error)
        state.add_error(error_message, self.agent_name)
        logger.error(f"{self.agent_name}: {error_message}")
        return state
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the agent's main functionality.
        
        This method must be implemented by each specialized agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Make the agent callable. This is the main entry point.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            logger.info(f"Executing {self.agent_name}")
            self.log_action(state, "agent_start")
            
            # Execute the agent's main functionality
            updated_state = self.execute(state)
            
            self.log_action(updated_state, "agent_complete")
            logger.info(f"Completed {self.agent_name}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Failed to execute {self.agent_name}: {e}")
            return self.handle_error(state, e, "agent execution")


class ToolUsingAgent(BaseAgent):
    """
    Extended base class for agents that use external tools.
    
    Provides additional functionality for tool management and execution.
    """
    
    def __init__(self, agent_name: str, model_name: str, system_prompt: str):
        super().__init__(agent_name, model_name, system_prompt)
        self.tools = {}
    
    def register_tool(self, tool_name: str, tool_function):
        """
        Register a tool that this agent can use.
        
        Args:
            tool_name: Name of the tool
            tool_function: Function to call when using this tool
        """
        self.tools[tool_name] = tool_function
        logger.info(f"Registered tool '{tool_name}' for {self.agent_name}")
    
    def use_tool(self, tool_name: str, state: AgentState, **kwargs) -> Any:
        """
        Use a registered tool.
        
        Args:
            tool_name: Name of the tool to use
            state: Current agent state
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result from the tool
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered for {self.agent_name}")
        
        try:
            self.log_action(state, "tool_use", tool_name=tool_name)
            result = self.tools[tool_name](**kwargs)
            self.log_action(state, "tool_complete", tool_name=tool_name)
            return result
            
        except Exception as e:
            self.handle_error(state, e, f"using tool {tool_name}")
            raise
