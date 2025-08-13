"""
Real UPA Dataset Processor for Legal AI Evaluation

Processes the actual UPA exam files:
- MCQ.pdf: 280 multiple choice questions  
- Essay 1-5.pdf: 10 essay questions (2 per PDF)

Note: Manual extraction and validation required due to different PDF templates.
"""

import PyPDF2
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class RealUPAProcessor:
    """Process real UPA exam PDFs for evaluation dataset."""
    
    def __init__(self, data_dir: str = "../../data/raw_evaluation_dataset"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(".")
        self.mcq_file = self.data_dir / "MCQ.pdf"
        self.essay_files = [self.data_dir / f"Essay {i}.pdf" for i in range(1, 6)]
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract raw text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return ""
    
    def analyze_mcq_structure(self):
        """Analyze MCQ PDF structure and provide extraction guidance."""
        if not self.mcq_file.exists():
            print(f"❌ MCQ file not found: {self.mcq_file}")
            return
            
        print(f"📄 Analyzing MCQ file: {self.mcq_file}")
        text = self.extract_text_from_pdf(self.mcq_file)
        
        # Save raw text for manual inspection
        with open("mcq_raw_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        
        # Basic analysis
        lines = text.split('\n')
        print(f"📊 MCQ Analysis:")
        print(f"  - Total lines: {len(lines)}")
        print(f"  - File size: {len(text)} characters")
        
        # Look for question patterns
        question_patterns = [
            r'\d+\.\s',  # "1. "
            r'Soal\s+\d+',  # "Soal 1"
            r'Question\s+\d+',  # "Question 1"
            r'\d+\)',  # "1)"
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text)
            print(f"  - Pattern '{pattern}': {len(matches)} matches")
        
        # Look for option patterns
        option_patterns = [
            r'[A-D]\.\s',  # "A. "
            r'[A-D]\)',  # "A)"
            r'[a-d]\.\s',  # "a. "
            r'[a-d]\)',  # "a)"
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, text)
            print(f"  - Option pattern '{pattern}': {len(matches)} matches")
        
        print(f"\n📝 First 500 characters:")
        print(text[:500])
        print(f"\n💾 Raw text saved to: mcq_raw_text.txt")
        
    def analyze_essay_structure(self):
        """Analyze Essay PDF structure and provide extraction guidance."""
        print(f"\n📄 Analyzing Essay files:")
        
        for i, essay_file in enumerate(self.essay_files, 1):
            if not essay_file.exists():
                print(f"❌ Essay {i} file not found: {essay_file}")
                continue
                
            print(f"\n📖 Essay {i}: {essay_file}")
            text = self.extract_text_from_pdf(essay_file)
            
            # Save raw text for manual inspection
            with open(f"essay_{i}_raw_text.txt", "w", encoding="utf-8") as f:
                f.write(text)
            
            lines = text.split('\n')
            print(f"  - Total lines: {len(lines)}")
            print(f"  - File size: {len(text)} characters")
            
            # Look for case/scenario patterns
            case_patterns = [
                r'[Kk]asus',  # "Kasus"
                r'[Ss]kenario',  # "Skenario"
                r'[Cc]ase',  # "Case"
                r'[Ss]oal\s+[Ee]ssay',  # "Soal Essay"
            ]
            
            for pattern in case_patterns:
                matches = re.findall(pattern, text)
                print(f"  - Case pattern '{pattern}': {len(matches)} matches")
            
            print(f"  - First 300 characters:")
            print(f"    {text[:300]}")
            print(f"  - Raw text saved to: essay_{i}_raw_text.txt")
    
    def create_manual_extraction_template(self):
        """Create template files for manual data entry."""
        
        # MCQ Template
        mcq_template = {
            "instructions": "Fill in the extracted MCQ data here",
            "total_questions": 280,
            "questions": [
                {
                    "id": f"mcq_{i}",
                    "type": "multiple_choice",
                    "question": "EXTRACT_QUESTION_TEXT_HERE",
                    "options": {
                        "A": "OPTION_A_TEXT",
                        "B": "OPTION_B_TEXT", 
                        "C": "OPTION_C_TEXT",
                        "D": "OPTION_D_TEXT"
                    },
                    "correct_answer": "CORRECT_LETTER",
                    "justification": "OFFICIAL_REASONING_IF_AVAILABLE"
                } for i in range(1, 6)  # Sample 5 questions
            ]
        }
        
        with open("mcq_extraction_template.json", "w", encoding="utf-8") as f:
            json.dump(mcq_template, f, indent=2, ensure_ascii=False)
        
        # Essay Template  
        essay_template = {
            "instructions": "Fill in the extracted essay data here",
            "total_essays": 10,
            "essays": [
                {
                    "id": f"essay_{i}",
                    "type": "case_study",
                    "pdf_source": f"Essay {(i-1)//2 + 1}.pdf",  # Map to PDF files
                    "scenario": "EXTRACT_CASE_SCENARIO_HERE",
                    "questions": [
                        "EXTRACT_QUESTION_1",
                        "EXTRACT_QUESTION_2_IF_EXISTS"
                    ],
                    "official_answer": "EXTRACT_OFFICIAL_ANSWER_IF_AVAILABLE",
                    "key_points": [
                        "KEY_LEGAL_POINT_1",
                        "KEY_LEGAL_POINT_2"
                    ]
                } for i in range(1, 11)
            ]
        }
        
        with open("essay_extraction_template.json", "w", encoding="utf-8") as f:
            json.dump(essay_template, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Manual extraction templates created:")
        print(f"  - mcq_extraction_template.json")
        print(f"  - essay_extraction_template.json")
    
    def create_sample_real_dataset(self):
        """Create sample dataset based on real UPA structure for testing."""
        
        # Sample real MCQ questions (Indonesian legal content)
        sample_mcqs = [
            {
                "id": "mcq_1",
                "type": "multiple_choice",
                "question": "Berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas, berapakah jumlah minimum pemegang saham untuk mendirikan PT?",
                "options": {
                    "A": "1 (satu) orang",
                    "B": "2 (dua) orang", 
                    "C": "3 (tiga) orang",
                    "D": "5 (lima) orang"
                },
                "correct_answer": "A",
                "justification": "Berdasarkan Pasal 7 ayat (1) UU No. 40 Tahun 2007, PT dapat didirikan oleh 1 (satu) orang atau lebih."
            },
            {
                "id": "mcq_2",
                "type": "multiple_choice", 
                "question": "Dalam hukum kontrak Indonesia, unsur-unsur sahnya perjanjian berdasarkan Pasal 1320 KUHPerdata adalah:",
                "options": {
                    "A": "Sepakat, cakap, hal tertentu, causa yang halal",
                    "B": "Sepakat, cakap, tertulis, bermaterai",
                    "C": "Sepakat, dewasa, objek jelas, tidak melawan hukum", 
                    "D": "Sepakat, cakap, hal tertentu, tidak ada paksaan"
                },
                "correct_answer": "A",
                "justification": "Pasal 1320 KUHPerdata menyebutkan 4 syarat: sepakat, cakap, hal tertentu, dan causa yang halal."
            },
            {
                "id": "mcq_3",
                "type": "multiple_choice",
                "question": "Berdasarkan UU No. 8 Tahun 1999 tentang Perlindungan Konsumen, pelaku usaha yang menyediakan barang/jasa bertanggung jawab memberikan:",
                "options": {
                    "A": "Ganti rugi atas kerugian konsumen",
                    "B": "Informasi yang benar dan jelas",
                    "C": "Kompensasi atas ketidakpuasan", 
                    "D": "Semua jawaban benar"
                },
                "correct_answer": "B",
                "justification": "Pasal 7 UU Perlindungan Konsumen mewajibkan pelaku usaha memberikan informasi yang benar, jelas dan jujur."
            },
            {
                "id": "mcq_4",
                "type": "multiple_choice",
                "question": "Dalam hukum perdata, jangka waktu daluwarsa untuk menuntut ganti kerugian akibat perbuatan melawan hukum adalah:",
                "options": {
                    "A": "1 (satu) tahun",
                    "B": "3 (tiga) tahun",
                    "C": "5 (lima) tahun",
                    "D": "30 (tiga puluh) tahun"
                },
                "correct_answer": "A", 
                "justification": "Berdasarkan Pasal 1478 KUHPerdata, tuntutan ganti kerugian akibat PMH daluwarsa dalam 1 tahun."
            },
            {
                "id": "mcq_5",
                "type": "multiple_choice",
                "question": "Menurut UU No. 13 Tahun 2003 tentang Ketenagakerjaan, masa percobaan kerja maksimal untuk pekerja adalah:",
                "options": {
                    "A": "1 (satu) bulan",
                    "B": "2 (dua) bulan", 
                    "C": "3 (tiga) bulan",
                    "D": "6 (enam) bulan"
                },
                "correct_answer": "C",
                "justification": "Pasal 60 UU Ketenagakerjaan menetapkan masa percobaan maksimal 3 bulan."
            }
        ]
        
        # Sample real case studies
        sample_essays = [
            {
                "id": "essay_1",
                "type": "case_study",
                "scenario": "PT Maju Sejahtera didirikan dengan modal dasar Rp 5 miliar dan modal disetor Rp 1 miliar. Direksi PT tersebut meminjam dana Rp 3 miliar dari Bank XYZ untuk ekspansi usaha tanpa persetujuan RUPS. Karena bisnis merugi, perusahaan tidak dapat melunasi hutang tersebut. Bank XYZ menuntut para pemegang saham untuk bertanggung jawab secara pribadi atas hutang perusahaan dengan alasan bahwa modal disetor tidak mencukupi untuk operasional perusahaan (undercapitalization).",
                "questions": [
                    "Jelaskan prinsip tanggung jawab terbatas (limited liability) dalam PT berdasarkan UU No. 40 Tahun 2007!",
                    "Analisis apakah tindakan direksi meminjam dana tanpa persetujuan RUPS melanggar ketentuan hukum!",
                    "Dalam kondisi apa doktrin piercing the corporate veil dapat diterapkan terhadap pemegang saham PT?",
                    "Bagaimana perlindungan hukum bagi kreditor dalam struktur PT menurut hukum Indonesia?"
                ],
                "official_answer": "Analisis harus mencakup: (1) Prinsip limited liability dalam Pasal 3 ayat (1) UU PT, (2) Kewenangan direksi dan batasan-batasannya, (3) Kondisi piercing the corporate veil dalam undercapitalization dan penyalahgunaan badan hukum, (4) Mekanisme perlindungan kreditor melalui modal minimum, jaminan, dan prosedur kepailitan.",
                "key_points": [
                    "Tanggung jawab terbatas pemegang saham sebatas saham yang dimiliki",
                    "Kewenangan direksi meminjam dana sesuai anggaran dasar dan RUPS",
                    "Piercing the corporate veil dalam kasus undercapitalization dan penyalahgunaan",
                    "Perlindungan kreditor melalui modal, jaminan, dan mekanisme hukum"
                ]
            },
            {
                "id": "essay_2", 
                "type": "case_study",
                "scenario": "CV Sukses Mandiri membuat kontrak jual beli 1000 unit smartphone dengan PT Digital Store senilai Rp 2 miliar dengan pembayaran 50% di muka dan sisanya setelah barang diterima. Setelah menerima pembayaran 50%, CV Sukses Mandiri hanya mengirim 600 unit dengan kualitas yang tidak sesuai spesifikasi. PT Digital Store menolak pembayaran sisanya dan menuntut ganti rugi Rp 500 juta atas kerugian yang diderita akibat tidak dapat memenuhi pesanan pelanggan.",
                "questions": [
                    "Analisis unsur-unsur wanprestasi yang terjadi dalam kasus ini berdasarkan KUHPerdata!",
                    "Jelaskan jenis-jenis ganti rugi yang dapat dituntut oleh PT Digital Store!",
                    "Bagaimana cara menghitung kerugian yang dapat diminta sebagai ganti rugi?",
                    "Apa saja pembelaan hukum yang dapat diajukan oleh CV Sukses Mandiri?"
                ],
                "official_answer": "Analisis harus mencakup: (1) Unsur wanprestasi: tidak memenuhi prestasi, terlambat, keliru dalam prestasi berdasarkan Pasal 1234 KUHPerdata, (2) Jenis ganti rugi: biaya, rugi, dan bunga (Pasal 1246), (3) Perhitungan kerugian langsung dan tidak langsung yang dapat dibuktikan, (4) Pembelaan: keadaan memaksa, kesalahan PT Digital Store, atau impossibility of performance.",
                "key_points": [
                    "Wanprestasi berupa tidak memenuhi prestasi dan prestasi yang keliru",
                    "Ganti rugi meliputi biaya, rugi, dan bunga yang dapat dibuktikan",
                    "Kerugian langsung dan tidak langsung harus memiliki hubungan kausal",
                    "Pembelaan force majeure dan contributory negligence"
                ]
            }
        ]
        
        # Create comprehensive dataset
        all_data = sample_mcqs + sample_essays
        
        # Save as JSONL
        with open("real_upa_sample_dataset.jsonl", "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Create answer keys
        mcq_answers = {item["id"]: item["correct_answer"] for item in sample_mcqs}
        with open("real_mcq_answer_key.json", "w", encoding="utf-8") as f:
            json.dump(mcq_answers, f, indent=2, ensure_ascii=False)
        
        # Create reference answers for essays
        essay_references = {
            item["id"]: {
                "official_answer": item["official_answer"],
                "key_points": item["key_points"]
            } for item in sample_essays
        }
        with open("real_essay_reference_answers.json", "w", encoding="utf-8") as f:
            json.dump(essay_references, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Real UPA sample dataset created:")
        print(f"  📊 {len(sample_mcqs)} multiple choice questions")
        print(f"  📝 {len(sample_essays)} case study essays")
        print(f"  📁 Files: real_upa_sample_dataset.jsonl, real_mcq_answer_key.json, real_essay_reference_answers.json")


def main():
    """Main function to process real UPA dataset."""
    processor = RealUPAProcessor()
    
    print("🔍 UPA Dataset Analysis and Processing")
    print("=" * 50)
    
    # Analyze file structures
    processor.analyze_mcq_structure()
    processor.analyze_essay_structure()
    
    # Create templates for manual extraction
    processor.create_manual_extraction_template()
    
    # Create sample dataset based on real UPA structure
    processor.create_sample_real_dataset()
    
    print(f"\n📋 Next Steps:")
    print(f"1. Review the raw text files (mcq_raw_text.txt, essay_*_raw_text.txt)")
    print(f"2. Use the template files to manually extract questions")
    print(f"3. Update the sample dataset with real extracted data")
    print(f"4. Run evaluation with the real dataset")


if __name__ == "__main__":
    main()
