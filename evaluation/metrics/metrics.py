"""
Evaluation Metrics for Multi-Agent System vs Baselines

This module implements evaluation metrics:
1. Answer Accuracy for Multiple Choice Questions
2. Semantic Reasoning Score for Case Studies
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import re
from groq import Groq
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    system_name: str
    accuracy: Optional[float] = None
    semantic_score: Optional[float] = None
    total_questions: int = 0
    correct_answers: int = 0
    case_scores: List[float] = None
    detailed_results: Dict[str, Any] = None


class AccuracyEvaluator:
    """Evaluates multiple choice question accuracy."""
    
    def __init__(self):
        self.results = {}
    
    def load_answer_key(self, answer_key_path: str) -> Dict[str, str]:
        """Load answer key from file."""
        try:
            with open(answer_key_path, 'r', encoding='utf-8') as f:
                answer_key = json.load(f)
            return answer_key
        except Exception as e:
            logger.error(f"Error loading answer key: {e}")
            return {}
    
    def evaluate_mc_answers(
        self, 
        predictions: List[Dict[str, Any]], 
        answer_key: Dict[str, str]
    ) -> EvaluationResult:
        """Evaluate multiple choice predictions against answer key."""
        
        correct_count = 0
        total_count = 0
        detailed_results = []
        
        for prediction in predictions:
            question_id = prediction["question_id"]
            predicted_answer = prediction.get("predicted_answer")
            correct_answer = answer_key.get(question_id)
            
            if correct_answer:
                total_count += 1
                is_correct = predicted_answer == correct_answer
                if is_correct:
                    correct_count += 1
                
                detailed_results.append({
                    "question_id": question_id,
                    "predicted": predicted_answer,
                    "correct": correct_answer,
                    "is_correct": is_correct,
                    "confidence": prediction.get("confidence", 0.5)
                })
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        return EvaluationResult(
            system_name=predictions[0].get("system", "unknown") if predictions else "unknown",
            accuracy=accuracy,
            total_questions=total_count,
            correct_answers=correct_count,
            detailed_results={"mc_results": detailed_results}
        )


class SemanticReasoningEvaluator:
    """Evaluates case study semantic reasoning using LLM-as-a-judge."""
    
    def __init__(self, judge_model: str = "gemma2-9b-it"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.judge_model = judge_model
        self.evaluation_prompt = """You are an expert legal evaluation system. Your task is to evaluate the quality of legal reasoning in case study responses.

Please evaluate the following legal analysis on a scale of 0.0 to 1.0 based on these criteria:
1. Legal Accuracy (0.3): Correct application of Indonesian law
2. Reasoning Quality (0.3): Logical structure and coherence
3. Completeness (0.2): Addresses all relevant legal issues
4. Practical Utility (0.2): Provides actionable legal advice

Reference Answer:
{reference_answer}

System Response:
{system_response}

Provide your evaluation as a JSON object with:
- "score": float between 0.0 and 1.0
- "reasoning": string explaining your evaluation
- "criteria_scores": object with scores for each criterion

Example format:
{
  "score": 0.85,
  "reasoning": "The response demonstrates strong legal knowledge...",
  "criteria_scores": {
    "legal_accuracy": 0.9,
    "reasoning_quality": 0.8,
    "completeness": 0.8,
    "practical_utility": 0.9
  }
}"""
    
    def evaluate_case_response(
        self, 
        system_response: str, 
        reference_answer: str
    ) -> Dict[str, Any]:
        """Evaluate a single case study response."""
        
        try:
            prompt = self.evaluation_prompt.format(
                reference_answer=reference_answer,
                system_response=system_response
            )
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert legal evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    "score": result.get("score", 0.5),
                    "reasoning": result.get("reasoning", ""),
                    "criteria_scores": result.get("criteria_scores", {}),
                    "raw_response": result_text
                }
            except json.JSONDecodeError:
                # Fallback: extract score using regex
                score_match = re.search(r'"score":\s*([0-9.]+)', result_text)
                score = float(score_match.group(1)) if score_match else 0.5
                
                return {
                    "score": score,
                    "reasoning": "Failed to parse full evaluation",
                    "criteria_scores": {},
                    "raw_response": result_text
                }
        
        except Exception as e:
            logger.error(f"Error in semantic evaluation: {e}")
            return {
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "criteria_scores": {},
                "raw_response": ""
            }
    
    def evaluate_case_studies(
        self, 
        predictions: List[Dict[str, Any]], 
        reference_answers: Dict[str, str]
    ) -> EvaluationResult:
        """Evaluate all case study predictions."""
        
        case_scores = []
        detailed_results = []
        
        for prediction in predictions:
            case_id = prediction["case_id"]
            system_response = prediction["legal_analysis"]
            reference_answer = reference_answers.get(case_id, "")
            
            if reference_answer:
                evaluation = self.evaluate_case_response(system_response, reference_answer)
                score = evaluation["score"]
                case_scores.append(score)
                
                detailed_results.append({
                    "case_id": case_id,
                    "score": score,
                    "evaluation": evaluation,
                    "system": prediction.get("system", "unknown")
                })
        
        avg_semantic_score = sum(case_scores) / len(case_scores) if case_scores else 0
        
        return EvaluationResult(
            system_name=predictions[0].get("system", "unknown") if predictions else "unknown",
            semantic_score=avg_semantic_score,
            case_scores=case_scores,
            detailed_results={"case_results": detailed_results}
        )


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""
    
    def __init__(self):
        self.accuracy_evaluator = AccuracyEvaluator()
        self.semantic_evaluator = SemanticReasoningEvaluator()
        self.results_dir = Path("../results")
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_system(
        self,
        system_name: str,
        mc_predictions: List[Dict[str, Any]],
        case_predictions: List[Dict[str, Any]], 
        mc_answer_key: Dict[str, str],
        case_reference_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate a complete system (MC + Case Studies)."""
        
        # Evaluate multiple choice
        mc_result = self.accuracy_evaluator.evaluate_mc_answers(mc_predictions, mc_answer_key)
        
        # Evaluate case studies
        case_result = self.semantic_evaluator.evaluate_case_studies(case_predictions, case_reference_answers)
        
        # Combine results
        combined_result = {
            "system_name": system_name,
            "multiple_choice": {
                "accuracy": mc_result.accuracy,
                "correct_answers": mc_result.correct_answers,
                "total_questions": mc_result.total_questions,
                "detailed_results": mc_result.detailed_results
            },
            "case_studies": {
                "semantic_score": case_result.semantic_score,
                "individual_scores": case_result.case_scores,
                "detailed_results": case_result.detailed_results
            },
            "overall_performance": {
                "mc_accuracy": mc_result.accuracy,
                "semantic_reasoning": case_result.semantic_score,
                "combined_score": (mc_result.accuracy + case_result.semantic_score) / 2
            }
        }
        
        return combined_result
    
    def create_comparison_table(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison table similar to Table II in paper."""
        
        comparison = {
            "systems": [],
            "metrics": {
                "mc_accuracy": [],
                "semantic_reasoning": [],
                "combined_score": []
            },
            "summary": {}
        }
        
        for result in all_results:
            system_name = result["system_name"]
            comparison["systems"].append(system_name)
            
            mc_acc = result["overall_performance"]["mc_accuracy"]
            sem_score = result["overall_performance"]["semantic_reasoning"] 
            combined = result["overall_performance"]["combined_score"]
            
            comparison["metrics"]["mc_accuracy"].append(mc_acc)
            comparison["metrics"]["semantic_reasoning"].append(sem_score)
            comparison["metrics"]["combined_score"].append(combined)
        
        # Calculate summary statistics
        comparison["summary"] = {
            "best_mc_accuracy": max(comparison["metrics"]["mc_accuracy"]),
            "best_semantic_reasoning": max(comparison["metrics"]["semantic_reasoning"]),
            "best_combined": max(comparison["metrics"]["combined_score"]),
            "system_rankings": {
                "mc_accuracy": sorted(zip(comparison["systems"], comparison["metrics"]["mc_accuracy"]), 
                                    key=lambda x: x[1], reverse=True),
                "semantic_reasoning": sorted(zip(comparison["systems"], comparison["metrics"]["semantic_reasoning"]), 
                                           key=lambda x: x[1], reverse=True),
                "combined_score": sorted(zip(comparison["systems"], comparison["metrics"]["combined_score"]), 
                                       key=lambda x: x[1], reverse=True)
            }
        }
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to file."""
        output_path = self.results_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def create_sample_answer_keys():
    """Create sample answer keys for testing."""
    
    # Sample MC answer key
    mc_answer_key = {
        "mc_1": "B",
        "mc_2": "A", 
        "mc_3": "C",
        "mc_4": "B",
        "mc_5": "D"
    }
    
    # Sample case study reference answers
    case_reference_answers = {
        "case_1": """Dalam kasus PT ABC, berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas:

1. Tanggung jawab pemegang saham terbatas pada modal yang disetor
2. Kreditor dapat mengajukan pailit jika syarat terpenuhi
3. Direksi bertanggung jawab atas pengurusan perusahaan
4. Likuidasi dapat dilakukan melalui pengadilan

Analisis hukum menunjukkan bahwa prinsip limited liability berlaku, namun ada pengecualian dalam kondisi tertentu.""",
        
        "case_2": """Berdasarkan analisis yuridis terhadap kasus kontrak kerja:

1. Pemutusan hubungan kerja harus sesuai UU Ketenagakerjaan
2. Pesangon dan kompensasi wajib dibayarkan
3. Prosedur pemutusan harus mengikuti tahapan yang ditetapkan
4. Pekerja memiliki hak untuk mengajukan keberatan

Kesimpulan: Pemutusan sepihak tanpa prosedur yang tepat dapat dikategorikan sebagai pemutusan tidak sah."""
    }
    
    # Save to files
    output_dir = Path("evaluation/datasets")
    
    with open(output_dir / "mc_answer_key.json", 'w', encoding='utf-8') as f:
        json.dump(mc_answer_key, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "case_reference_answers.json", 'w', encoding='utf-8') as f:
        json.dump(case_reference_answers, f, ensure_ascii=False, indent=2)
    
    print("Sample answer keys created!")


if __name__ == "__main__":
    create_sample_answer_keys()
