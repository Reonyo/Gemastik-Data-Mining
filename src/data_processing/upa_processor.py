"""
UPA Evaluation Dataset Processor

This module processes the UPA exam dataset and creates structured evaluation data 
for comparing Multi-Agent System vs baselines.
"""

import json
import re
import PyPDF2
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UPADatasetProcessor:
    """Processes UPA exam data into structured evaluation format."""
    
    def __init__(self, data_dir: str = "../../data/raw_evaluation_dataset"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(".")  # Current directory (evaluation/datasets)
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_pdf_content(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return ""
    
    def parse_multiple_choice_questions(self, text: str) -> List[Dict[str, Any]]:
        """Parse multiple choice questions from text."""
        questions = []
        
        # Pattern to match questions with options A, B, C, D
        question_pattern = r'(\d+)\.\s*(.*?)(?=\n[A-D]\.|\n\d+\.|$)'
        option_pattern = r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|\n\d+\.|$)'
        
        question_matches = re.finditer(question_pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in question_matches:
            question_num = int(match.group(1))
            question_text = match.group(2).strip()
            
            # Find options for this question
            question_end = match.end()
            next_question_start = text.find(f"\n{question_num + 1}.", question_end)
            if next_question_start == -1:
                question_section = text[question_end:]
            else:
                question_section = text[question_end:next_question_start]
            
            options = {}
            option_matches = re.finditer(option_pattern, question_section, re.DOTALL)
            for opt_match in option_matches:
                option_letter = opt_match.group(1)
                option_text = opt_match.group(2).strip()
                options[option_letter] = option_text
            
            if len(options) >= 3:  # At least 3 options
                questions.append({
                    "id": f"mc_{question_num}",
                    "type": "multiple_choice",
                    "question": question_text,
                    "options": options,
                    "answer": None  # To be filled manually or from answer key
                })
        
        return questions
    
    def parse_case_studies(self, text: str) -> List[Dict[str, Any]]:
        """Parse case study questions from text."""
        case_studies = []
        
        # Pattern to identify case studies (usually longer text blocks)
        case_pattern = r'(?:KASUS|Case Study|Studi Kasus)\s*(\d+)[:.]?\s*(.*?)(?=(?:KASUS|Case Study|Studi Kasus)\s*\d+|$)'
        
        case_matches = re.finditer(case_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(case_matches, 1):
            case_num = match.group(1) if match.group(1) else str(i)
            case_text = match.group(2).strip()
            
            # Split into scenario and questions
            lines = case_text.split('\n')
            scenario_lines = []
            questions = []
            current_question = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line starts with question indicator
                if re.match(r'^\d+\.|\?|Pertanyaan:', line):
                    if current_question:
                        questions.append(current_question.strip())
                    current_question = line
                elif current_question:
                    current_question += " " + line
                else:
                    scenario_lines.append(line)
            
            if current_question:
                questions.append(current_question.strip())
            
            scenario = " ".join(scenario_lines)
            
            case_studies.append({
                "id": f"case_{case_num}",
                "type": "case_study",
                "scenario": scenario,
                "questions": questions,
                "expected_reasoning": None  # To be filled from reference
            })
        
        return case_studies
    
    def process_upa_exam(self) -> Dict[str, List[Dict[str, Any]]]:
        """Process the main UPA exam file."""
        pdf_path = self.data_dir / "SOAL UJIAN ADVOKAT.pdf"
        
        if not pdf_path.exists():
            logger.error(f"UPA exam file not found: {pdf_path}")
            return {"multiple_choice": [], "case_studies": []}
        
        text = self.extract_pdf_content(pdf_path)
        
        # Parse different question types
        mc_questions = self.parse_multiple_choice_questions(text)
        case_studies = self.parse_case_studies(text)
        
        logger.info(f"Extracted {len(mc_questions)} multiple choice questions")
        logger.info(f"Extracted {len(case_studies)} case studies")
        
        return {
            "multiple_choice": mc_questions,
            "case_studies": case_studies
        }
    
    def create_evaluation_dataset(self) -> None:
        """Create structured evaluation dataset files."""
        # Process UPA exam
        upa_data = self.process_upa_exam()
        
        # Create multiple choice dataset
        mc_dataset = {
            "metadata": {
                "name": "UPA Multiple Choice Questions",
                "description": "280 multiple choice questions from UPA exam",
                "question_count": len(upa_data["multiple_choice"]),
                "evaluation_metric": "accuracy"
            },
            "questions": upa_data["multiple_choice"]
        }
        
        # Create case study dataset
        case_dataset = {
            "metadata": {
                "name": "UPA Case Studies", 
                "description": "6 case studies from UPA exam",
                "case_count": len(upa_data["case_studies"]),
                "evaluation_metric": "semantic_reasoning_score"
            },
            "cases": upa_data["case_studies"]
        }
        
        # Save datasets
        with open(self.output_dir / "upa_multiple_choice.json", 'w', encoding='utf-8') as f:
            json.dump(mc_dataset, f, ensure_ascii=False, indent=2)
        
        with open(self.output_dir / "upa_case_studies.json", 'w', encoding='utf-8') as f:
            json.dump(case_dataset, f, ensure_ascii=False, indent=2)
        
        # Create combined JSONL format for easy processing
        with open(self.output_dir / "upa_eval_set.jsonl", 'w', encoding='utf-8') as f:
            for question in upa_data["multiple_choice"]:
                f.write(json.dumps(question, ensure_ascii=False) + '\n')
            
            for case in upa_data["case_studies"]:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        logger.info("Evaluation datasets created successfully")
        logger.info(f"Files saved to: {self.output_dir}")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the created dataset."""
        stats = {
            "multiple_choice_questions": 0,
            "case_studies": 0,
            "total_items": 0,
            "missing_answers": 0,
            "validation_errors": []
        }
        
        try:
            # Check JSONL file
            jsonl_path = self.output_dir / "upa_eval_set.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        stats["total_items"] += 1
                        
                        if item["type"] == "multiple_choice":
                            stats["multiple_choice_questions"] += 1
                            if not item.get("answer"):
                                stats["missing_answers"] += 1
                        elif item["type"] == "case_study":
                            stats["case_studies"] += 1
                            if not item.get("expected_reasoning"):
                                stats["missing_answers"] += 1
            
        except Exception as e:
            stats["validation_errors"].append(str(e))
        
        return stats


def main():
    """Main function to process UPA dataset."""
    logging.basicConfig(level=logging.INFO)
    
    processor = UPADatasetProcessor()
    
    print("Processing UPA Exam Dataset...")
    processor.create_evaluation_dataset()
    
    print("\nValidating dataset...")
    stats = processor.validate_dataset()
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset files created in: evaluation/datasets/")


if __name__ == "__main__":
    main()
