"""
Test suite for the Legal Multi-Agent System.

Comprehensive tests for all components of the legal analysis workflow.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graph.agent_state import AgentState, RetrievedDocument, AnalysisStep
from config.legal_config import LegalAgentConfig
from agents.assistant import LegalAssistantAgent
from agents.researcher import LegalResearcherAgent
from agents.lawyer import SeniorLawyerAgent
from agents.editor import LegalEditorAgent
from agents.supervisor import SupervisorAgent


class TestAgentState(unittest.TestCase):
    """Test the AgentState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = AgentState(
            user_query="Test legal question",
            max_iterations=3
        )
    
    def test_initialization(self):
        """Test AgentState initialization."""
        self.assertEqual(self.state.user_query, "Test legal question")
        self.assertEqual(self.state.max_iterations, 3)
        self.assertEqual(self.state.iteration_count, 0)
        self.assertFalse(self.state.is_complete)
        self.assertEqual(len(self.state.errors), 0)
    
    def test_add_error(self):
        """Test error addition."""
        self.state.add_error("Test error", "test_agent")
        self.assertEqual(len(self.state.errors), 1)
        self.assertIn("Test error", self.state.errors[0])
        self.assertIn("test_agent", self.state.errors[0])
    
    def test_mark_complete(self):
        """Test completion marking."""
        self.assertFalse(self.state.is_complete)
        self.state.mark_complete()
        self.assertTrue(self.state.is_complete)
    
    def test_add_document(self):
        """Test document addition."""
        doc = RetrievedDocument(
            title="Test Document",
            content="Test content",
            source="test_source",
            relevance_score=0.8
        )
        
        self.state.retrieved_documents = [doc]
        self.assertEqual(len(self.state.retrieved_documents), 1)
        self.assertEqual(self.state.retrieved_documents[0].title, "Test Document")
    
    def test_add_analysis_step(self):
        """Test analysis step addition."""
        step = AnalysisStep(
            step_number=1,
            reasoning="Test reasoning",
            legal_basis="Test legal basis",
            conclusion="Test conclusion"
        )
        
        self.state.legal_analysis = [step]
        self.assertEqual(len(self.state.legal_analysis), 1)
        self.assertEqual(self.state.legal_analysis[0].step_number, 1)


class TestLegalConfig(unittest.TestCase):
    """Test the LegalAgentConfig class."""
    
    def test_get_agent_config(self):
        """Test agent configuration retrieval."""
        config = LegalAgentConfig.get_agent_config("legal_assistant")
        self.assertIn("model", config)
        self.assertIn("system_prompt", config)
        self.assertEqual(config["model"], "gemma2-9b-it")
    
    def test_get_all_models(self):
        """Test all models retrieval."""
        models = LegalAgentConfig.get_all_models()
        self.assertIn("gemma2-9b-it", models)
        self.assertIn("meta-llama/llama-3-8b-instruct", models)
        self.assertIn("moonshot-v1-8k", models)
        self.assertIn("llama-3.1-8b-instant", models)
    
    def test_invalid_agent_config(self):
        """Test invalid agent configuration."""
        with self.assertRaises(KeyError):
            LegalAgentConfig.get_agent_config("invalid_agent")


class TestLegalAssistantAgent(unittest.TestCase):
    """Test the Legal Assistant Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = LegalAssistantAgent()
        self.state = AgentState(
            user_query="Saya ingin mendirikan perusahaan teknologi di Indonesia. Apa persyaratan hukumnya?",
            max_iterations=3
        )
    
    @patch('agents.base_agent.BaseAgent.call_llm')
    def test_execute_success(self, mock_llm):
        """Test successful execution."""
        mock_llm.return_value = """
        STRUCTURED_FACTS:
        - Pendirian perusahaan teknologi di Indonesia
        - Persyaratan hukum yang diperlukan
        - Prosedur pendaftaran perusahaan
        
        QUERY_TYPE: company_formation
        COMPLETENESS: sufficient
        """
        
        result_state = self.agent.execute(self.state)
        
        self.assertIsNotNone(result_state.structured_facts)
        self.assertTrue(len(result_state.structured_facts) > 0)
        mock_llm.assert_called_once()
    
    @patch('agents.base_agent.BaseAgent.call_llm')
    def test_execute_error(self, mock_llm):
        """Test execution with error."""
        mock_llm.side_effect = Exception("LLM error")
        
        result_state = self.agent.execute(self.state)
        
        self.assertTrue(len(result_state.errors) > 0)
        self.assertIn("LLM error", result_state.errors[0])


class TestSupervisorAgent(unittest.TestCase):
    """Test the Supervisor Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.supervisor = SupervisorAgent()
        self.state = AgentState(
            user_query="Test query",
            max_iterations=3
        )
    
    def test_route_initial_state(self):
        """Test routing from initial state."""
        next_agent = self.supervisor.route_to_next_agent(self.state)
        self.assertEqual(next_agent, "legal_assistant")
    
    def test_route_with_facts(self):
        """Test routing with structured facts."""
        self.state.structured_facts = ["Fact 1", "Fact 2"]
        next_agent = self.supervisor.route_to_next_agent(self.state)
        self.assertEqual(next_agent, "legal_researcher")
    
    def test_route_with_documents(self):
        """Test routing with retrieved documents."""
        self.state.structured_facts = ["Fact 1"]
        self.state.retrieved_documents = [Mock()]
        next_agent = self.supervisor.route_to_next_agent(self.state)
        self.assertEqual(next_agent, "senior_lawyer")
    
    def test_route_with_analysis(self):
        """Test routing with legal analysis."""
        self.state.structured_facts = ["Fact 1"]
        self.state.retrieved_documents = [Mock()]
        self.state.legal_analysis = [Mock()]
        next_agent = self.supervisor.route_to_next_agent(self.state)
        self.assertEqual(next_agent, "legal_editor")
    
    def test_route_complete(self):
        """Test routing when complete."""
        self.state.structured_facts = ["Fact 1"]
        self.state.retrieved_documents = [Mock()]
        self.state.legal_analysis = [Mock()]
        self.state.final_document = "Final document"
        next_agent = self.supervisor.route_to_next_agent(self.state)
        self.assertEqual(next_agent, "END")
    
    def test_max_iterations(self):
        """Test maximum iterations handling."""
        self.state.iteration_count = 3
        next_agent = self.supervisor.route_to_next_agent(self.state)
        # Should force completion or END
        self.assertIn(next_agent, ["legal_editor", "END"])
    
    def test_should_continue(self):
        """Test continue decision logic."""
        # Should continue initially
        self.assertTrue(self.supervisor.should_continue(self.state))
        
        # Should not continue when complete
        self.state.mark_complete()
        self.assertFalse(self.supervisor.should_continue(self.state))
        
        # Should not continue at max iterations
        self.state.is_complete = False
        self.state.iteration_count = 3
        self.assertFalse(self.supervisor.should_continue(self.state))


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables for testing
        os.environ['GROQ_API_KEY'] = 'test_groq_key'
        os.environ['TAVILY_API_KEY'] = 'test_tavily_key'
    
    @patch('workflow.legal_workflow.LegalWorkflowGraph')
    def test_workflow_creation(self, mock_workflow):
        """Test workflow creation."""
        from workflow.legal_workflow import create_legal_workflow
        
        workflow = create_legal_workflow()
        mock_workflow.assert_called_once()
    
    @patch('main.LegalAnalysisSystem._validate_environment')
    @patch('workflow.legal_workflow.create_legal_workflow')
    def test_system_initialization(self, mock_create_workflow, mock_validate):
        """Test system initialization."""
        from main import LegalAnalysisSystem
        
        mock_validate.return_value = None
        mock_workflow = Mock()
        mock_create_workflow.return_value = mock_workflow
        
        system = LegalAnalysisSystem()
        
        self.assertIsNotNone(system.workflow)
        mock_validate.assert_called_once()
        mock_create_workflow.assert_called_once()
    
    def test_environment_validation(self):
        """Test environment variable validation."""
        from main import LegalAnalysisSystem
        
        # Test with valid environment
        system = LegalAnalysisSystem()
        self.assertTrue(system._check_environment())
        
        # Test with missing environment variable
        del os.environ['GROQ_API_KEY']
        with self.assertRaises(ValueError):
            system._validate_environment()


class TestErrorHandling(unittest.TestCase):
    """Test error handling across the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = AgentState(
            user_query="Test query",
            max_iterations=3
        )
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        agent = LegalAssistantAgent()
        
        with patch.object(agent, 'call_llm', side_effect=Exception("Test error")):
            result_state = agent.execute(self.state)
            
            self.assertTrue(len(result_state.errors) > 0)
            self.assertIn("Test error", result_state.errors[0])
    
    def test_state_error_accumulation(self):
        """Test error accumulation in state."""
        self.state.add_error("Error 1", "agent1")
        self.state.add_error("Error 2", "agent2")
        self.state.add_error("Error 3", "agent3")
        
        self.assertEqual(len(self.state.errors), 3)
        self.assertIn("Error 1", self.state.errors[0])
        self.assertIn("Error 3", self.state.errors[2])


class TestDocumentProcessing(unittest.TestCase):
    """Test document processing functionality."""
    
    def test_document_creation(self):
        """Test document creation."""
        doc = RetrievedDocument(
            title="Test Document",
            content="This is test content for legal analysis.",
            source="test_database",
            relevance_score=0.95
        )
        
        self.assertEqual(doc.title, "Test Document")
        self.assertEqual(doc.relevance_score, 0.95)
        self.assertIn("legal analysis", doc.content)
    
    def test_analysis_step_creation(self):
        """Test analysis step creation."""
        step = AnalysisStep(
            step_number=1,
            reasoning="This is the reasoning for step 1",
            legal_basis="Article 15 of Company Law",
            conclusion="Therefore, companies must register with authorities"
        )
        
        self.assertEqual(step.step_number, 1)
        self.assertIn("reasoning", step.reasoning)
        self.assertIn("Article 15", step.legal_basis)
        self.assertIn("register", step.conclusion)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAgentState,
        TestLegalConfig,
        TestLegalAssistantAgent,
        TestSupervisorAgent,
        TestWorkflowIntegration,
        TestErrorHandling,
        TestDocumentProcessing
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
