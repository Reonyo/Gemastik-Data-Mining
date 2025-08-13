"""
Legal AI System Evaluator

Implements three evaluation metrics for legal AI systems:
1. MCQ Accuracy - Binary correctness for multiple choice questions
2. MCQ Semantic Score - LLM-judge evaluation for reasoning quality  
3. Essay Semantic Score - LLM-judge evaluation for case analysis quality

Uses Gemma 2 as the LLM judge for semantic evaluation.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
from groq import Groq
import numpy as np
from datetime import datetime

class LegalEvaluator:
    """Evaluator for legal AI systems."""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.judge_model = "gemma2-9b-it"  # LLM judge for semantic reasoning
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the UPA legal examination dataset."""
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def load_answer_keys(self, mcq_key_path: str, essay_key_path: str) -> Tuple[Dict, Dict]:
        """Load answer keys for both MC and essays."""
        with open(mcq_key_path, 'r', encoding='utf-8') as f:
            mcq_keys = json.load(f)
        
        with open(essay_key_path, 'r', encoding='utf-8') as f:
            essay_keys = json.load(f)
            
        return mcq_keys, essay_keys
    
    def evaluate_answer_accuracy_mc(self, predictions: List[Dict], mcq_keys: Dict) -> Dict:
        """
        Metric 1: Answer Accuracy (MC)
        Binary correctness evaluation for multiple choice questions.
        """
        results = {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "details": []
        }
        
        for pred in predictions:
            if pred["type"] == "multiple_choice":
                question_id = pred["id"]
                predicted_answer = pred.get("predicted_answer", "").strip().upper()
                correct_answer = mcq_keys.get(question_id, {}).get("correct_answer", "").strip().upper()
                
                is_correct = predicted_answer == correct_answer
                if is_correct:
                    results["correct"] += 1
                
                results["total"] += 1
                results["details"].append({
                    "question_id": question_id,
                    "predicted": predicted_answer,
                    "correct": correct_answer,
                    "is_correct": is_correct,
                    "topic": mcq_keys.get(question_id, {}).get("topic", "Unknown")
                })
        
        if results["total"] > 0:
            results["accuracy"] = (results["correct"] / results["total"]) * 100
        
        return results
    
    async def evaluate_semantic_reasoning_mc(self, predictions: List[Dict], mcq_keys: Dict) -> Dict:
        """
        Metric 2: Semantic Reasoning Score (MC)
        LLM-as-judge evaluation of MC justification quality.
        """
        results = {
            "scores": [],
            "average_score": 0.0,
            "details": []
        }
        
        for pred in predictions:
            if pred["type"] == "multiple_choice":
                question_id = pred["id"]
                predicted_justification = pred.get("justification", "No justification provided")
                reference_justification = mcq_keys.get(question_id, {}).get("justification", "")
                topic = mcq_keys.get(question_id, {}).get("topic", "Unknown")
                
                # Create evaluation prompt for LLM judge
                evaluation_prompt = f"""You are an expert legal educator evaluating the quality of legal reasoning for multiple choice answers.

**Question Topic:** {topic}

**Reference Answer Justification (Gold Standard):**
{reference_justification}

**Student's Justification to Evaluate:**
{predicted_justification}

**Evaluation Criteria:**
1. **Legal Accuracy**: Does the justification cite correct legal sources and principles?
2. **Completeness**: Does it address the key legal elements required?
3. **Clarity**: Is the legal reasoning clear and well-structured?
4. **Relevance**: Is the justification directly relevant to the legal question?

**Instructions:**
Rate the student's justification on a scale of 0.0 to 1.0, where:
- 1.0 = Perfect legal reasoning (matches or exceeds reference quality)
- 0.8-0.9 = Excellent reasoning with minor omissions
- 0.6-0.7 = Good reasoning but missing some key elements
- 0.4-0.5 = Basic reasoning with significant gaps
- 0.2-0.3 = Poor reasoning with major errors
- 0.0-0.1 = Completely incorrect or irrelevant

Respond with only a single number between 0.0 and 1.0 (e.g., "0.75")."""

                try:
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    score_text = response.choices[0].message.content.strip()
                    # Extract numerical score
                    try:
                        score = float(score_text)
                        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    except:
                        score = 0.0  # Default if parsing fails
                    
                    results["scores"].append(score)
                    results["details"].append({
                        "question_id": question_id,
                        "score": score,
                        "topic": topic,
                        "predicted_justification": predicted_justification[:200] + "..." if len(predicted_justification) > 200 else predicted_justification,
                        "reference_justification": reference_justification[:200] + "..." if len(reference_justification) > 200 else reference_justification
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {question_id}: {e}")
                    results["scores"].append(0.0)
                    results["details"].append({
                        "question_id": question_id,
                        "score": 0.0,
                        "error": str(e),
                        "topic": topic
                    })
        
        if results["scores"]:
            results["average_score"] = np.mean(results["scores"])
        
        return results
    
    async def evaluate_semantic_reasoning_essays(self, predictions: List[Dict], essay_keys: Dict) -> Dict:
        """
        Metric 3: Semantic Reasoning Score (Essays)  
        LLM-as-judge evaluation of essay analysis quality.
        """
        results = {
            "scores": [],
            "average_score": 0.0,
            "details": []
        }
        
        for pred in predictions:
            if pred["type"] == "case_study":
                question_id = pred["id"]
                predicted_analysis = pred.get("analysis", "No analysis provided")
                reference_answer = essay_keys.get(question_id, {}).get("official_answer", "")
                key_points = essay_keys.get(question_id, {}).get("key_points", [])
                topic = essay_keys.get(question_id, {}).get("topic", "Unknown")
                
                # Create comprehensive evaluation prompt
                evaluation_prompt = f"""You are a senior law professor evaluating legal case analysis quality.

**Case Topic:** {topic}

**Reference Analysis (Gold Standard):**
{reference_answer}

**Key Points That Should Be Addressed:**
{chr(10).join([f"• {point}" for point in key_points])}

**Student's Analysis to Evaluate:**
{predicted_analysis}

**Evaluation Criteria:**
1. **Legal Accuracy (25%)**: Correct application of relevant laws, regulations, and legal principles
2. **Completeness (25%)**: Addresses all major legal issues and required elements  
3. **Reasoning Quality (25%)**: Logical structure, coherent argumentation, and sound legal analysis
4. **Practical Application (25%)**: Provides actionable legal advice and realistic solutions

**Instructions:**
Evaluate the student's analysis comprehensively across all four criteria. Rate on a scale of 0.0 to 1.0:
- 1.0 = Exceptional analysis (law firm partner level)
- 0.8-0.9 = Excellent analysis (senior associate level)
- 0.6-0.7 = Good analysis (junior associate level)
- 0.4-0.5 = Basic analysis (law student level)
- 0.2-0.3 = Poor analysis (major deficiencies)
- 0.0-0.1 = Inadequate analysis (fundamental errors)

Consider both the depth of legal knowledge and practical applicability.
Respond with only a single number between 0.0 and 1.0 (e.g., "0.82")."""

                try:
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    score_text = response.choices[0].message.content.strip()
                    try:
                        score = float(score_text)
                        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    except:
                        score = 0.0
                    
                    results["scores"].append(score)
                    results["details"].append({
                        "question_id": question_id,
                        "score": score,
                        "topic": topic,
                        "analysis_length": len(predicted_analysis),
                        "key_points_covered": len(key_points),
                        "predicted_analysis": predicted_analysis[:300] + "..." if len(predicted_analysis) > 300 else predicted_analysis
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {question_id}: {e}")
                    results["scores"].append(0.0)
                    results["details"].append({
                        "question_id": question_id,
                        "score": 0.0,
                        "error": str(e),
                        "topic": topic
                    })
        
        if results["scores"]:
            results["average_score"] = np.mean(results["scores"])
        
        return results
    
    async def comprehensive_evaluation(self, predictions: List[Dict], 
                                     mcq_key_path: str, essay_key_path: str) -> Dict:
        """
        Run comprehensive Table II evaluation across all three metrics.
        """
        print("🚀 Starting Table II Comprehensive Evaluation...")
        
        # Load answer keys
        mcq_keys, essay_keys = self.load_answer_keys(mcq_key_path, essay_key_path)
        
        # Metric 1: Answer Accuracy (MC)
        print("📊 Evaluating Metric 1: Answer Accuracy (MC)...")
        accuracy_results = self.evaluate_answer_accuracy_mc(predictions, mcq_keys)
        
        # Metric 2: Semantic Reasoning Score (MC)
        print("🧠 Evaluating Metric 2: Semantic Reasoning Score (MC)...")
        mc_semantic_results = await self.evaluate_semantic_reasoning_mc(predictions, mcq_keys)
        
        # Metric 3: Semantic Reasoning Score (Essays)  
        print("📝 Evaluating Metric 3: Semantic Reasoning Score (Essays)...")
        essay_semantic_results = await self.evaluate_semantic_reasoning_essays(predictions, essay_keys)
        
        # Compile comprehensive results
        comprehensive_results = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(predictions),
                "mc_questions": len([p for p in predictions if p["type"] == "multiple_choice"]),
                "essay_questions": len([p for p in predictions if p["type"] == "case_study"]),
                "judge_model": self.judge_model
            },
            "table_ii_metrics": {
                "metric_1_answer_accuracy_mc": {
                    "score": accuracy_results["accuracy"],
                    "unit": "percentage",
                    "description": "Percentage of correctly answered multiple choice questions"
                },
                "metric_2_semantic_reasoning_mc": {
                    "score": mc_semantic_results["average_score"],
                    "unit": "0-1 scale", 
                    "description": "LLM-judge evaluation of MC justification quality"
                },
                "metric_3_semantic_reasoning_essays": {
                    "score": essay_semantic_results["average_score"],
                    "unit": "0-1 scale",
                    "description": "LLM-judge evaluation of essay analysis quality"
                }
            },
            "detailed_results": {
                "answer_accuracy": accuracy_results,
                "mc_semantic_reasoning": mc_semantic_results,
                "essay_semantic_reasoning": essay_semantic_results
            }
        }
        
        return comprehensive_results

def create_sample_predictions():
    """Create sample predictions for testing the evaluation framework."""
    sample_predictions = [
        {
            "id": "mcq_1",
            "type": "multiple_choice",
            "predicted_answer": "C",
            "justification": "Berdasarkan Pasal 32 ayat (3) Undang-Undang Nomor 18 Tahun 2003 tentang Advokat, profesi advokat yang terhimpun dalam PERADI terdiri dari 8 organisasi advokat yaitu: IKADIN, AAI, IPHI, HAPI, SPI, AKHI, HKHPM, dan APSI."
        },
        {
            "id": "mcq_2",
            "type": "multiple_choice", 
            "predicted_answer": "D",
            "justification": "Pasal 32 ayat (3) UU Advokat menyebutkan 8 organisasi yang terhimpun dalam PERADI. Peradin tidak termasuk dalam daftar organisasi tersebut."
        },
        {
            "id": "essay_1",
            "type": "case_study",
            "analysis": "Dalam kasus PT Bank Central Anggrek, tanggung jawab hukum Direktur Utama bank diatur dalam beberapa ketentuan: (1) Berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas, direksi memiliki tanggung jawab fiduciary duty dan harus menjalankan tugasnya dengan itikad baik serta penuh tanggung jawab. (2) UU No. 10 Tahun 1998 tentang Perbankan mengatur prinsip kehati-hatian yang harus dijalankan direksi bank. (3) Bank Indonesia memiliki kewenangan pengawasan makroprudensial dan mikroprudensial berdasarkan UU No. 3 Tahun 2004. (4) Pertanggungjawaban pidana direksi diatur dalam Pasal 49-52 UU Perbankan untuk pelanggaran prinsip kehati-hatian."
        }
    ]
    return sample_predictions

async def main():
    """Main function to demonstrate Table II evaluation."""
    import os
    
    # Setup (replace with actual API key)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        return
    
    # Initialize evaluator
    evaluator = LegalEvaluator(groq_api_key)
    
    # Create sample predictions for testing
    sample_predictions = create_sample_predictions()
    
    # Run comprehensive evaluation
    results = await evaluator.comprehensive_evaluation(
        sample_predictions,
        "real_mcq_answer_key.json",
        "real_essay_reference_answers.json"
    )
    
    # Save results
    with open("table_ii_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display summary
    print("\n" + "="*60)
    print("📊 TABLE II EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    metrics = results["table_ii_metrics"]
    print(f"🎯 Metric 1 - Answer Accuracy (MC): {metrics['metric_1_answer_accuracy_mc']['score']:.1f}%")
    print(f"🧠 Metric 2 - Semantic Reasoning (MC): {metrics['metric_2_semantic_reasoning_mc']['score']:.3f}/1.0")
    print(f"📝 Metric 3 - Semantic Reasoning (Essays): {metrics['metric_3_semantic_reasoning_essays']['score']:.3f}/1.0")
    
    print(f"\n📁 Detailed results saved to: table_ii_evaluation_results.json")
    print("✅ Table II Evaluation Complete!")

if __name__ == "__main__":
    asyncio.run(main())
