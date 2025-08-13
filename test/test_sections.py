"""
Test script to debug section identification
"""

import re
from pathlib import Path
from pdfminer.high_level import extract_text

def test_sections():
    pdf_path = Path("data/raw_evaluation_dataset/SOAL UJIAN ADVOKAT.pdf")
    text = extract_text(str(pdf_path))
    
    print(f"Text length: {len(text)}")
    
    # Test the question number reset approach
    question_1_pattern = r'\b1\.\s+'
    question_1_matches = list(re.finditer(question_1_pattern, text))
    
    print(f"\nFound {len(question_1_matches)} instances of '1.'")
    
    # Show context around each "1."
    for i, match in enumerate(question_1_matches[:10]):  # Show first 10
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 100)
        context = text[start:end].replace('\n', ' ')
        print(f"Match {i+1}: ...{context}...")
    
    # Count questions 1-40 patterns
    print(f"\nQuestion number patterns:")
    for num in [1, 2, 3, 5, 10, 20, 30, 40]:
        pattern = rf'\b{num}\.\s+'
        count = len(re.findall(pattern, text))
        print(f"  {num}.: {count} times")
    
    # Count Jawaban patterns
    jawaban_count = len(re.findall(r'Jawaban\s+[A-D]', text))
    print(f"\nJawaban patterns: {jawaban_count}")
    
    # Look for potential section dividers
    print(f"\nPotential section indicators:")
    patterns = [
        (r'MATERI', 'MATERI'),
        (r'UNDANG[‐-]UNDANG', 'UNDANG-UNDANG'),
        (r'KODE\s+ETIK', 'KODE ETIK'),
        (r'HUKUM\s+ACARA', 'HUKUM ACARA'),
    ]
    
    for pattern, desc in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"  {desc}: {len(matches)} times")

if __name__ == "__main__":
    test_sections()
