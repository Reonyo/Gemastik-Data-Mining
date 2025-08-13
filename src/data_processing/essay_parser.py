"""
Essay Dataset Parser for UPA Legal Cases

Parses essay questions and answers from raw text files into structured JSON format.
Creates proper dataset structure with instruction and answer pairs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

class EssayDatasetParser:
    """Parser for UPA essay questions and answers"""
    
    def __init__(self, raw_data_path: str, output_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.output_data_path = Path(output_data_path)
        self.output_data_path.mkdir(exist_ok=True)
        
    def parse_essay_file(self, question_file: str, answer_file: str) -> List[Dict]:
        """Parse a single essay question-answer pair file"""
        
        # Read question file
        question_path = self.raw_data_path / question_file
        with open(question_path, 'r', encoding='utf-8') as f:
            question_content = f.read().strip()
            
        # Read answer file  
        answer_path = self.raw_data_path / answer_file
        with open(answer_path, 'r', encoding='utf-8') as f:
            answer_content = f.read().strip()
        
        # Extract case description and questions
        if "Pertanyaan" in question_content:
            parts = question_content.split("Pertanyaan")
            case_description = parts[0].strip()
            questions_text = parts[1].strip()
        else:
            case_description = question_content
            questions_text = ""
        
        # Parse individual questions from the questions section
        questions = self._extract_questions(questions_text)
        
        # Parse answers
        answers = self._extract_answers(answer_content)
        
        # Create structured data
        essay_data = []
        for i, question in enumerate(questions, 1):
            # Find corresponding answer
            answer = self._find_answer_for_question(i, answers)
            
            essay_data.append({
                "id": f"essay_{question_file.replace('.txt', '')}_{i}",
                "case_description": case_description,
                "instruction": question,
                "answer": answer,
                "type": "essay",
                "source_files": {
                    "question": question_file,
                    "answer": answer_file
                }
            })
            
        return essay_data
    
    def _extract_questions(self, questions_text: str) -> List[str]:
        """Extract individual questions from questions section"""
        questions = []
        
        # Split by question numbers
        lines = questions_text.split('\n')
        current_question = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (new question)
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_question:
                    questions.append(current_question.strip())
                current_question = line
            else:
                current_question += " " + line
                
        if current_question:
            questions.append(current_question.strip())
            
        return questions
    
    def _extract_answers(self, answer_content: str) -> Dict[int, str]:
        """Extract answers mapped by question number"""
        answers = {}
        
        # Remove "Jawaban" header if present
        if answer_content.startswith("Jawaban"):
            answer_content = answer_content[7:].strip()
        
        # Split by question numbers
        sections = []
        current_section = ""
        
        for line in answer_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (new answer)
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += "\n" + line
                
        if current_section:
            sections.append(current_section.strip())
        
        # Parse each section
        for section in sections:
            lines = section.split('\n')
            first_line = lines[0].strip()
            
            if first_line.startswith(('1.', '2.', '3.', '4.', '5.')):
                question_num = int(first_line[0])
                answer_text = '\n'.join(lines).strip()
                answers[question_num] = answer_text
                
        return answers
    
    def _find_answer_for_question(self, question_num: int, answers: Dict[int, str]) -> str:
        """Find the answer for a specific question number"""
        return answers.get(question_num, "Answer not found")
    
    def parse_all_essays(self) -> List[Dict]:
        """Parse all essay files in the dataset"""
        
        # Define the essay file pairs based on the actual structure
        essay_pairs = [
            ("soal 1-2.txt", "jawaban 1-2.txt"),
            ("soal 3.txt", "jawaban 3.txt"), 
            ("soal 4-5.txt", "jawaban 4-5.txt"),
            ("soal 6-7.txt", "jawaban 6-7.txt"),
            ("soal 8-9.txt", "jawaban 8-9.txt")
        ]
        
        all_essays = []
        
        for question_file, answer_file in essay_pairs:
            try:
                essays = self.parse_essay_file(question_file, answer_file)
                all_essays.extend(essays)
                print(f"✅ Parsed {len(essays)} essays from {question_file}")
            except Exception as e:
                print(f"❌ Error parsing {question_file}: {e}")
                
        return all_essays
    
    def save_dataset(self, essays: List[Dict], filename: str = "essay_dataset.json"):
        """Save parsed essays to JSON file"""
        
        output_file = self.output_data_path / filename
        
        # Create the structured dataset
        dataset = {
            "metadata": {
                "description": "UPA Legal Essay Dataset",
                "total_essays": len(essays),
                "question_types": ["legal_document_drafting", "legal_analysis"],
                "created_by": "essay_parser.py"
            },
            "essays": essays
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Saved {len(essays)} essays to {output_file}")
        return output_file

def main():
    """Main function to parse essays"""
    
    # Define paths according to proper project structure
    raw_data_path = "data/raw_evaluation_dataset"
    output_data_path = "data/processed"
    
    # Initialize parser
    parser = EssayDatasetParser(raw_data_path, output_data_path)
    
    # Parse all essays
    print("🔍 Parsing UPA Essay Dataset...")
    essays = parser.parse_all_essays()
    
    # Save dataset
    output_file = parser.save_dataset(essays, "upa_essay_dataset.json")
    
    # Print summary
    print("\n📊 Essay Dataset Summary:")
    print(f"  - Total Essays: {len(essays)}")
    
    # Count questions per file pair
    file_counts = {}
    for essay in essays:
        source = essay['source_files']['question']
        if source not in file_counts:
            file_counts[source] = 0
        file_counts[source] += 1
    
    for file, count in file_counts.items():
        print(f"  - {file}: {count} questions")
    
    print(f"  - Output: {output_file}")
    
    return essays

if __name__ == "__main__":
    main()
