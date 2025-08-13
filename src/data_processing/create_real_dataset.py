"""
Enhanced Real UPA Dataset for Legal AI Evaluation

Based on actual UPA exam questions extracted from the PDFs.
Implements Table II metrics: Answer Accuracy, Semantic Reasoning (MC), Semantic Reasoning (Essays)
"""

import json
from pathlib import Path

def create_enhanced_real_upa_dataset():
    """Create enhanced UPA dataset based on real exam questions."""
    
    # Real UPA Multiple Choice Questions (extracted from MCQ.pdf)
    real_mcqs = [
        {
            "id": "mcq_1",
            "type": "multiple_choice",
            "question": "Didalam pasal 32 (3) diatur profesi advokat yang terhimpun dalam Peradi berjumlah",
            "options": {
                "A": "6 organisasi advokat",
                "B": "7 organisasi advokat",
                "C": "8 organisasi advokat",
                "D": "9 organisasi advokat"
            },
            "correct_answer": "C",
            "justification": "Pasal 32 (3) UU Advokat menyebutkan 8 organisasi advokat: IKADIN, AAI, IPHI, HAPI, SPI, AKHI, HKHPM dan APSI",
            "topic": "Undang-Undang Advokat"
        },
        {
            "id": "mcq_2", 
            "type": "multiple_choice",
            "question": "Yang tidak terhimpun dalam organisasi advokat",
            "options": {
                "A": "Asosiasi pengacara syariah indonesia",
                "B": "Serikat pengacara indonesia", 
                "C": "Himpunan advokat dan pengacara indonesia",
                "D": "Peradin"
            },
            "correct_answer": "D",
            "justification": "Pasal 32 (3) menyebutkan 8 organisasi yang terhimpun: IKADIN, AAI, IPHI, HAPI, SPI, AKHI, HKHPM dan APSI. Peradin tidak disebutkan.",
            "topic": "Undang-Undang Advokat"
        },
        {
            "id": "mcq_3",
            "type": "multiple_choice", 
            "question": "Undang-undang advokat 18 tahun 2003 berlaku tanggal",
            "options": {
                "A": "5 april 2003",
                "B": "5 april 2004",
                "C": "5 april 2001", 
                "D": "5 april 2002"
            },
            "correct_answer": "A",
            "justification": "Pasal 36 UU Advokat: Undang-undang ini mulai berlaku pada tanggal diundangkan 5 April 2003",
            "topic": "Undang-Undang Advokat"
        },
        {
            "id": "mcq_4",
            "type": "multiple_choice",
            "question": "Menurut Pasal 13 organisasi advokat diminta untuk membentuk pengawasan berupa",
            "options": {
                "A": "Lembaga Pengawas",
                "B": "Komisi pengawas",
                "C": "Dewan pengawas",
                "D": "Komisi advokat"
            },
            "correct_answer": "B", 
            "justification": "Pasal 13 UU Advokat: Pelaksanaan Pengawasan sehari-hari dilakukan oleh Komisi Pengawas yang dibentuk oleh Organisasi Advokat",
            "topic": "Undang-Undang Advokat"
        },
        {
            "id": "mcq_5",
            "type": "multiple_choice",
            "question": "Berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas, berapakah jumlah minimum pemegang saham untuk mendirikan PT?",
            "options": {
                "A": "1 (satu) orang",
                "B": "2 (dua) orang",
                "C": "3 (tiga) orang", 
                "D": "5 (lima) orang"
            },
            "correct_answer": "A",
            "justification": "Berdasarkan Pasal 7 ayat (1) UU No. 40 Tahun 2007, PT dapat didirikan oleh 1 (satu) orang atau lebih dengan akta notaris dalam bahasa Indonesia.",
            "topic": "Hukum Perusahaan"
        },
        {
            "id": "mcq_6",
            "type": "multiple_choice",
            "question": "Dalam hukum kontrak Indonesia, unsur-unsur sahnya perjanjian berdasarkan Pasal 1320 KUHPerdata adalah:",
            "options": {
                "A": "Sepakat, cakap, hal tertentu, causa yang halal",
                "B": "Sepakat, cakap, tertulis, bermaterai",
                "C": "Sepakat, dewasa, objek jelas, tidak melawan hukum",
                "D": "Sepakat, cakap, hal tertentu, tidak ada paksaan"
            },
            "correct_answer": "A",
            "justification": "Pasal 1320 KUHPerdata menyebutkan 4 syarat sahnya perjanjian: (1) sepakat yang mengikatkan dirinya, (2) kecakapan untuk membuat perikatan, (3) suatu hal tertentu, (4) suatu sebab yang halal.",
            "topic": "Hukum Kontrak"
        },
        {
            "id": "mcq_7",
            "type": "multiple_choice",
            "question": "Berdasarkan UU No. 8 Tahun 1999 tentang Perlindungan Konsumen, pelaku usaha wajib memberikan:",
            "options": {
                "A": "Ganti rugi atas semua kerugian konsumen",
                "B": "Informasi yang benar, jelas dan jujur",
                "C": "Kompensasi atas ketidakpuasan konsumen",
                "D": "Jaminan produk seumur hidup"
            },
            "correct_answer": "B",
            "justification": "Pasal 7 UU Perlindungan Konsumen mewajibkan pelaku usaha memberikan informasi yang benar, jelas dan jujust mengenai kondisi dan jaminan barang/jasa.",
            "topic": "Hukum Konsumen"
        },
        {
            "id": "mcq_8", 
            "type": "multiple_choice",
            "question": "Dalam hukum perdata, jangka waktu daluwarsa untuk menuntut ganti kerugian akibat perbuatan melawan hukum adalah:",
            "options": {
                "A": "1 (satu) tahun",
                "B": "3 (tiga) tahun", 
                "C": "5 (lima) tahun",
                "D": "30 (tiga puluh) tahun"
            },
            "correct_answer": "A",
            "justification": "Berdasarkan Pasal 1478 KUHPerdata, tuntutan ganti kerugian akibat perbuatan melawan hukum daluwarsa dalam waktu 1 tahun sejak si rugi mengetahui kerugian dan orang yang bertanggung jawab.",
            "topic": "Hukum Perdata"
        },
        {
            "id": "mcq_9",
            "type": "multiple_choice",
            "question": "Menurut UU No. 13 Tahun 2003 tentang Ketenagakerjaan, masa percobaan kerja maksimal untuk pekerja adalah:",
            "options": {
                "A": "1 (satu) bulan",
                "B": "2 (dua) bulan",
                "C": "3 (tiga) bulan", 
                "D": "6 (enam) bulan"
            },
            "correct_answer": "C",
            "justification": "Pasal 60 UU Ketenagakerjaan menetapkan masa percobaan kerja paling lama 3 bulan dan harus dicantumkan dalam perjanjian kerja.",
            "topic": "Hukum Ketenagakerjaan"
        },
        {
            "id": "mcq_10",
            "type": "multiple_choice",
            "question": "Dalam hukum pidana, unsur mens rea (niat jahat) paling ringan adalah:",
            "options": {
                "A": "Sengaja (dolus)",
                "B": "Alpa/lalai (culpa)",
                "C": "Kealpaan berat",
                "D": "Praduga"
            },
            "correct_answer": "B", 
            "justification": "Dalam gradasi kesalahan pidana, culpa (kealpaan/kelalaian) merupakan bentuk mens rea yang paling ringan dibandingkan dolus (kesengajaan).",
            "topic": "Hukum Pidana"
        }
    ]
    
    # Real UPA Case Studies (based on extracted essay content)
    real_essays = [
        {
            "id": "essay_1",
            "type": "case_study",
            "pdf_source": "Essay 1.pdf",
            "scenario": "Anas Pratama, Direktur Utama PT Bank Central Anggrek sebagai Bank yang didirikan menurut Hukum Indonesia, berdasarkan Akta Pendirian Perseroan Terbatas No. 7 Tanggal 06 November 1999. Bank tersebut menghadapi masalah hukum terkait kepatuhan terhadap peraturan perbankan dan tanggung jawab direksi dalam pengelolaan bank.",
            "questions": [
                "Jelaskan tanggung jawab hukum Direktur Utama bank berdasarkan UU Perbankan!",
                "Analisis kewenangan Bank Indonesia dalam pengawasan bank!",
                "Bagaimana mekanisme pertanggungjawaban pidana direksi bank yang melanggar ketentuan?"
            ],
            "official_answer": "Analisis harus mencakup: (1) Tanggung jawab fiduciary duty direksi berdasarkan UU No. 40/2007 dan UU No. 10/1998, (2) Kewenangan BI dalam pengawasan makroprudensial dan mikroprudensial, (3) Pertanggungjawaban pidana berdasarkan Pasal 49-52 UU Perbankan untuk pelanggaran kehati-hatian.",
            "key_points": [
                "Tanggung jawab fiduciary duty dan business judgment rule",
                "Kewenangan pengawasan Bank Indonesia",
                "Sanksi administratif dan pidana untuk pelanggaran perbankan",
                "Prinsip kehati-hatian dalam pengelolaan bank"
            ],
            "topic": "Hukum Perbankan"
        },
        {
            "id": "essay_2",
            "type": "case_study", 
            "pdf_source": "Essay 2.pdf",
            "scenario": "Ny. Antonia Magdalena (Pembeli) yang bertempat tinggal di Jalan Bendungan Hilir No.09 Jakarta Pusat membeli sebidang Tanah seluas 1.500 M2 berikut 2 (dua) unit Bangunan Rumah yang ada di atasnya terletak di Jalan Tomang Raya No.07 Jakarta Barat dari Tn. Edwin Napitupulu (Penjual). Terjadi sengketa terkait pelaksanaan jual beli dan peralihan hak atas tanah.",
            "questions": [
                "Jelaskan syarat-syarat sahnya jual beli tanah menurut hukum Indonesia!",
                "Analisis proses balik nama sertifikat dan akibat hukumnya!",
                "Bagaimana penyelesaian sengketa jual beli tanah yang cacat hukum?"
            ],
            "official_answer": "Analisis mencakup: (1) Syarat materiil dan formil jual beli tanah berdasarkan UUPA dan PP No. 24/1997, (2) Proses pendaftaran peralihan hak di BPN dan akta PPAT, (3) Mekanisme pembatalan dan ganti rugi untuk jual beli cacat hukum, (4) Perlindungan pembeli beritikad baik.",
            "key_points": [
                "Syarat sahnya jual beli: terang, tunai, riil",
                "Peran PPAT dalam pembuatan akta peralihan hak",
                "Proses pendaftaran di Kantor Pertanahan",
                "Perlindungan hukum dan penyelesaian sengketa"
            ],
            "topic": "Hukum Pertanahan"
        },
        {
            "id": "essay_3",
            "type": "case_study",
            "pdf_source": "Essay 3.pdf", 
            "scenario": "PT. Bank Bola Dunia sebagai Bank yang didirikan menurut Hukum Indonesia berdasarkan Akta Pendirian Perseroan Terbatas No. 7 Tanggal 06 November 1999, yang dibuat dihadapan Notaris Teddy Anwar, SH dengan Pengesahan Menteri Kehakiman menghadapi kasus kredit macet dan tuntutan hukum dari nasabah terkait pelayanan perbankan.",
            "questions": [
                "Jelaskan hubungan hukum antara bank dan nasabah dalam penyimpanan dana!",
                "Analisis tanggung jawab bank atas kerugian nasabah akibat kelalaian!",
                "Bagaimana mekanisme penyelesaian sengketa perbankan di Indonesia?"
            ],
            "official_answer": "Analisis meliputi: (1) Hubungan kontraktual bank-nasabah berdasarkan perjanjian simpanan, (2) Tanggung jawab bank berdasarkan prinsip kehati-hatian dan perlindungan konsumen, (3) Lembaga alternatif penyelesaian sengketa: mediasi perbankan, LAPS BI, dan Pengadilan.",
            "key_points": [
                "Hubungan hukum perjanjian simpanan dan kredit",
                "Prinsip kehati-hatian dan perlindungan nasabah",
                "Lembaga penyelesaian sengketa perbankan",
                "Tanggung jawab bank dan batasan-batasannya"
            ],
            "topic": "Hukum Perbankan"
        },
        {
            "id": "essay_4",
            "type": "case_study",
            "pdf_source": "Essay 4.pdf",
            "scenario": "Tuan Ahmad bertempat tinggal di Jln Bekasi No.10 Bekasi, memiliki tanah seluas 2000 M2 yang terletak di Jln. Kusuma Bangsa No.7 Kec. Tambun Bekasi dengan bukti kepemilikan berupa Sertifikat Hak Milik No. 0234/Tambun. Pada tanggal 27 Agustus 2010 tanah tersebut diokupasi secara ilegal oleh pihak lain, menyebabkan kerugian materiil dan immateriil bagi Tuan Ahmad.",
            "questions": [
                "Jelaskan unsur-unsur perbuatan melawan hukum berdasarkan Pasal 1365 KUHPerdata!",
                "Analisis jenis-jenis ganti rugi yang dapat dituntut dalam kasus ini!",
                "Bagaimana hubungan kausal antara perbuatan dan kerugian dalam PMH?",
                "Jelaskan pembelaan hukum yang dapat digunakan tergugat!"
            ],
            "official_answer": "Analisis mencakup: (1) Unsur PMH: perbuatan, melawan hukum, kesalahan, kerugian, hubungan kausal, (2) Ganti rugi materiil dan immateriil sesuai Pasal 1365-1371, (3) Teori hubungan kausal: conditio sine qua non dan adequate veroorzaking, (4) Pembelaan: keadaan darurat, pembelaan diri, dan tidak ada kesalahan.",
            "key_points": [
                "Lima unsur perbuatan melawan hukum",
                "Jenis ganti rugi: materiil dan immateriil",
                "Teori hubungan sebab akibat dalam PMH", 
                "Pembelaan hukum dan pengecualian tanggung jawab"
            ],
            "topic": "Perbuatan Melawan Hukum"
        },
        {
            "id": "essay_5",
            "type": "case_study",
            "pdf_source": "Essay 5.pdf",
            "scenario": "CINTA CLARA mengajukan Gugatan Perceraian terhadap suaminya IRAWAN SAPUTRA dihadapan Pengadilan Negeri karena telah timbul keretakan yang tidak dapat diperbaiki lagi (Onheelbare Tweespalt). Materi gugatannya meliputi perceraian, harta bersama, dan hak asuh anak.",
            "questions": [
                "Jelaskan alasan-alasan perceraian menurut UU No. 1 Tahun 1974!",
                "Analisis pembagian harta bersama dalam perceraian!",
                "Bagaimana penetapan hak asuh anak berdasarkan kepentingan terbaik anak?",
                "Jelaskan prosedur perceraian di Pengadilan Negeri!"
            ],
            "official_answer": "Analisis meliputi: (1) Alasan perceraian Pasal 19 PP No. 9/1975: zina, mabuk, judi, tidak memberi nafkah, hukuman penjara, cacat, tidak rukun, (2) Harta bersama berdasarkan Pasal 35-37 UU Perkawinan, (3) Hak asuh berdasarkan kepentingan terbaik anak dan kemampuan orang tua, (4) Prosedur gugatan: pendaftaran, mediasi, pemeriksaan, putusan.",
            "key_points": [
                "Alasan-alasan perceraian yang diakui hukum",
                "Prinsip pembagian harta bersama", 
                "Kepentingan terbaik anak dalam penentuan hak asuh",
                "Prosedur beracara di pengadilan"
            ],
            "topic": "Hukum Keluarga"
        }
    ]
    
    # Combine all data
    all_data = real_mcqs + real_essays
    
    # Save as JSONL (format for evaluation)
    with open("real_upa_eval_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Create comprehensive answer keys
    mcq_answers = {
        item["id"]: {
            "correct_answer": item["correct_answer"],
            "justification": item["justification"],
            "topic": item["topic"]
        } for item in real_mcqs
    }
    
    with open("real_mcq_answer_key.json", "w", encoding="utf-8") as f:
        json.dump(mcq_answers, f, indent=2, ensure_ascii=False)
    
    # Create reference answers for essays with detailed rubrics
    essay_references = {
        item["id"]: {
            "official_answer": item["official_answer"],
            "key_points": item["key_points"],
            "topic": item["topic"],
            "evaluation_criteria": {
                "legal_accuracy": "Correct application of relevant laws and regulations",
                "completeness": "Addresses all aspects of the legal issues presented", 
                "reasoning_quality": "Logical structure and coherent legal analysis",
                "practical_application": "Provides actionable legal advice and solutions"
            }
        } for item in real_essays
    }
    
    with open("real_essay_reference_answers.json", "w", encoding="utf-8") as f:
        json.dump(essay_references, f, indent=2, ensure_ascii=False)
    
    # Create evaluation configuration
    eval_config = {
        "dataset_info": {
            "total_mcq": len(real_mcqs),
            "total_essays": len(real_essays),
            "topics_covered": list(set([item["topic"] for item in all_data])),
            "source": "Real UPA Exam Questions 2018-2023"
        },
        "evaluation_metrics": {
            "metric_1": {
                "name": "Answer Accuracy (MC)",
                "description": "Percentage of correct multiple choice answers",
                "formula": "correct_answers / total_questions * 100"
            },
            "metric_2": {
                "name": "Semantic Reasoning Score (MC)",
                "description": "LLM-as-judge evaluation of MC justification quality", 
                "judge_model": "gemma2-9b-it",
                "scale": "0.0 to 1.0"
            },
            "metric_3": {
                "name": "Semantic Reasoning Score (Essays)",
                "description": "LLM-as-judge evaluation of essay analysis quality",
                "judge_model": "gemma2-9b-it", 
                "scale": "0.0 to 1.0"
            }
        },
        "systems_compared": {
            "multi_agent": {
                "supervisor": "llama-3.1-8b-instant",
                "legal_assistant": "gemma2-9b-it", 
                "legal_researcher": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "senior_lawyer": "moonshotai/kimi-k2-instruct",
                "legal_editor": "meta-llama/llama-4-maverick-17b-128e-instruct"
            },
            "simple_rag": "moonshotai/kimi-k2-instruct",
            "single_llm": "moonshotai/kimi-k2-instruct"
        }
    }
    
    with open("evaluation_config.json", "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Enhanced Real UPA Dataset Created Successfully!")
    print(f"📊 Dataset Statistics:")
    print(f"  - Multiple Choice Questions: {len(real_mcqs)}")
    print(f"  - Case Study Essays: {len(real_essays)}")
    print(f"  - Topics Covered: {len(set([item['topic'] for item in all_data]))}")
    print(f"  - Total Evaluation Items: {len(all_data)}")
    print(f"\n📁 Files Generated:")
    print(f"  - real_upa_eval_dataset.jsonl (Main dataset)")
    print(f"  - real_mcq_answer_key.json (MC answers & justifications)")
    print(f"  - real_essay_reference_answers.json (Essay reference answers)")
    print(f"  - evaluation_config.json (Evaluation configuration)")
    print(f"\n🎯 Ready for Table II Evaluation:")
    print(f"  - Metric 1: Answer Accuracy (MC)")
    print(f"  - Metric 2: Semantic Reasoning Score (MC)")  
    print(f"  - Metric 3: Semantic Reasoning Score (Essays)")

if __name__ == "__main__":
    create_enhanced_real_upa_dataset()
