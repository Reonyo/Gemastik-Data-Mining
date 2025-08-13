"""
Baseline Systems for Evaluation

This module implements the baseline systems:
1. Simple RAG (kimi-k2-instruct + standard RAG pipeline)
2. Single LLM (kimi-k2-instruct only)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from groq import Groq
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config.legal_config import LegalAgentConfig

logger = logging.getLogger(__name__)


class SingleLLMBaseline:
    """Single LLM baseline using moonshotai/kimi-k2-instruct."""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "moonshotai/kimi-k2-instruct"
        self.system_prompt = """You are an expert Indonesian legal advisor. 
        Provide accurate, comprehensive legal analysis based on Indonesian law.
        For multiple choice questions, provide the letter of the correct answer.
        For case studies, provide detailed legal reasoning and conclusions."""
    
    def generate_response(self, question: str, context: str = "") -> str:
        """Generate response using single LLM only."""
        try:
            prompt = f"{context}\n\nQuestion: {question}\n\nProvide your legal analysis:"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error in Single LLM: {e}")
            return f"Error: {str(e)}"
    
    def answer_multiple_choice(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Answer multiple choice question."""
        question = question_data["question"]
        options = question_data["options"]
        
        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        full_question = f"{question}\n\n{options_text}\n\nAnswer (provide only the letter):"
        
        response = self.generate_response(full_question)
        
        # Extract answer letter
        answer_letter = None
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response.upper():
                answer_letter = letter
                break
        
        return {
            "question_id": question_data["id"],
            "predicted_answer": answer_letter,
            "confidence": 0.5,  # Default confidence
            "reasoning": response,
            "system": "single_llm"
        }
    
    def analyze_case_study(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze case study."""
        scenario = case_data["scenario"]
        questions = case_data.get("questions", [])
        
        case_prompt = f"""Legal Case Analysis:

Scenario: {scenario}

Questions to address:
{chr(10).join(f"- {q}" for q in questions)}

Provide comprehensive legal analysis including:
1. Legal issues identification
2. Applicable legal principles
3. Analysis and reasoning
4. Conclusions and recommendations"""
        
        response = self.generate_response(case_prompt)
        
        return {
            "case_id": case_data["id"],
            "legal_analysis": response,
            "system": "single_llm",
            "timestamp": datetime.now().isoformat()
        }


class SimpleRAGBaseline:
    """Simple RAG baseline using kimi-k2-instruct + ChromaDB."""
    
    def __init__(self, vector_db_path: str = "data/vector_db"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "moonshotai/kimi-k2-instruct"
        self.vector_db_path = vector_db_path
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.chroma_client.get_collection("legal_documents")
        except:
            logger.warning("No existing vector database found. Please run indexing first.")
            self.collection = None
        
        self.system_prompt = """You are an expert Indonesian legal advisor.
        Use the provided legal documents to answer questions accurately.
        Base your analysis on the given legal context and Indonesian law.
        For multiple choice questions, provide the letter of the correct answer.
        For case studies, provide detailed legal reasoning with citations."""
    
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })
            
            return documents
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_rag_response(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using RAG approach."""
        # Prepare context from retrieved documents
        context = ""
        if context_docs:
            context = "Relevant Legal Documents:\n\n"
            for i, doc in enumerate(context_docs[:3], 1):
                content = doc["content"][:1000]  # Limit content length
                context += f"Document {i}:\n{content}\n\n"
        
        prompt = f"""{context}

Question: {question}

Based on the provided legal documents and Indonesian law, provide your analysis:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            return f"Error: {str(e)}"
    
    def generate_response(self, question: str, context: str = "") -> str:
        """Generate response using RAG approach - compatibility method."""
        context_docs = self.retrieve_documents(question)
        return self.generate_rag_response(question, context_docs)
    
    def answer_multiple_choice(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Answer multiple choice question using RAG."""
        question = question_data["question"]
        options = question_data["options"]
        
        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        full_question = f"{question}\n\n{options_text}"
        
        # Retrieve relevant documents
        context_docs = self.retrieve_documents(question)
        
        # Generate RAG response
        full_prompt = f"{full_question}\n\nAnswer (provide only the letter):"
        response = self.generate_rag_response(full_prompt, context_docs)
        
        # Extract answer letter
        answer_letter = None
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response.upper():
                answer_letter = letter
                break
        
        return {
            "question_id": question_data["id"],
            "predicted_answer": answer_letter,
            "confidence": 0.7,  # Higher confidence with RAG
            "reasoning": response,
            "retrieved_docs": len(context_docs),
            "system": "simple_rag"
        }
    
    def analyze_case_study(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze case study using RAG."""
        scenario = case_data["scenario"]
        questions = case_data.get("questions", [])
        
        # Retrieve relevant documents for the case
        context_docs = self.retrieve_documents(scenario, n_results=8)
        
        case_prompt = f"""Legal Case Analysis:

Scenario: {scenario}

Questions to address:
{chr(10).join(f"- {q}" for q in questions)}

Provide comprehensive legal analysis including:
1. Legal issues identification
2. Applicable legal principles
3. Analysis and reasoning
4. Conclusions and recommendations"""
        
        response = self.generate_rag_response(case_prompt, context_docs)
        
        return {
            "case_id": case_data["id"],
            "legal_analysis": response,
            "retrieved_docs": len(context_docs),
            "system": "simple_rag",
            "timestamp": datetime.now().isoformat()
        }


def test_baselines():
    """Test both baseline systems."""
    print("Testing Baseline Systems...")
    
    # Test data
    test_mc = {
        "id": "test_mc_1",
        "type": "multiple_choice",
        "question": "Apa yang dimaksud dengan Perseroan Terbatas (PT)?",
        "options": {
            "A": "Badan usaha perorangan",
            "B": "Badan hukum yang modalnya terbagi dalam saham",
            "C": "Persekutuan komanditer",
            "D": "Koperasi"
        }
    }
    
    test_case = {
        "id": "test_case_1",
        "type": "case_study",
        "scenario": "PT ABC didirikan dengan modal Rp 100 juta. Setelah 2 tahun beroperasi, perusahaan mengalami kerugian besar dan tidak mampu membayar utang kepada kreditor.",
        "questions": ["Bagaimana tanggung jawab pemegang saham?", "Apa langkah hukum yang dapat diambil kreditor?"]
    }
    
    # Test Single LLM
    print("\n1. Testing Single LLM Baseline...")
    single_llm = SingleLLMBaseline()
    
    result_mc = single_llm.answer_multiple_choice(test_mc)
    print(f"MC Answer: {result_mc['predicted_answer']}")
    
    result_case = single_llm.analyze_case_study(test_case)
    print(f"Case Analysis Length: {len(result_case['legal_analysis'])} characters")
    
    # Test Simple RAG
    print("\n2. Testing Simple RAG Baseline...")
    simple_rag = SimpleRAGBaseline()
    
    result_mc_rag = simple_rag.answer_multiple_choice(test_mc)
    print(f"RAG MC Answer: {result_mc_rag['predicted_answer']}")
    print(f"Retrieved Docs: {result_mc_rag['retrieved_docs']}")
    
    result_case_rag = simple_rag.analyze_case_study(test_case)
    print(f"RAG Case Analysis Length: {len(result_case_rag['legal_analysis'])} characters")
    print(f"Retrieved Docs: {result_case_rag['retrieved_docs']}")
    
    print("\nBaseline systems tested successfully!")


if __name__ == "__main__":
    test_baselines()
