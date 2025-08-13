"""
Create sample evaluation dataset for testing
"""

import json
from pathlib import Path

def create_sample_upa_dataset():
    """Create sample UPA dataset for testing."""
    
    # Sample multiple choice questions
    mc_questions = [
        {
            "id": "mc_1",
            "type": "multiple_choice",
            "question": "Apa yang dimaksud dengan Perseroan Terbatas (PT)?",
            "options": {
                "A": "Badan usaha perorangan",
                "B": "Badan hukum yang modalnya terbagi dalam saham", 
                "C": "Persekutuan komanditer",
                "D": "Koperasi"
            }
        },
        {
            "id": "mc_2", 
            "type": "multiple_choice",
            "question": "Berapa modal minimum untuk mendirikan PT menurut UU No. 40 Tahun 2007?",
            "options": {
                "A": "Rp 25 juta",
                "B": "Rp 50 juta",
                "C": "Rp 100 juta", 
                "D": "Tidak ada minimum"
            }
        },
        {
            "id": "mc_3",
            "type": "multiple_choice",
            "question": "Siapa yang berwenang mengangkat dan memberhentikan direksi PT?",
            "options": {
                "A": "Pemegang saham",
                "B": "Rapat Umum Pemegang Saham (RUPS)",
                "C": "Dewan komisaris",
                "D": "Menteri Hukum dan HAM"
            }
        },
        {
            "id": "mc_4",
            "type": "multiple_choice",
            "question": "Dalam hukum kontrak, apa yang dimaksud dengan 'wanprestasi'?",
            "options": {
                "A": "Pelaksanaan kontrak sesuai kesepakatan",
                "B": "Tidak dipenuhinya kewajiban kontraktual",
                "C": "Pembatalan kontrak secara sepihak",
                "D": "Perubahan isi kontrak"
            }
        },
        {
            "id": "mc_5",
            "type": "multiple_choice", 
            "question": "Berdasarkan Pasal 1365 KUHPerdata, apa unsur utama perbuatan melawan hukum?",
            "options": {
                "A": "Kesengajaan dan kelalaian",
                "B": "Kerugian dan hubungan kausal",
                "C": "Perbuatan, kesalahan, kerugian, hubungan kausal",
                "D": "Niat jahat dan kesengajaan"
            }
        }
    ]
    
    # Sample case studies
    case_studies = [
        {
            "id": "case_1",
            "type": "case_study",
            "scenario": "PT ABC didirikan dengan modal dasar Rp 1 miliar dan modal disetor Rp 250 juta. Setelah 2 tahun beroperasi, perusahaan mengalami kerugian besar akibat investasi yang gagal. Perusahaan tidak mampu membayar utang kepada bank sebesar Rp 800 juta dan kepada supplier sebesar Rp 300 juta. Para kreditor menuntut agar pemegang saham bertanggung jawab secara pribadi atas utang perusahaan.",
            "questions": [
                "Bagaimana prinsip tanggung jawab terbatas pemegang saham dalam PT menurut UU No. 40 Tahun 2007?",
                "Dalam kondisi apa pemegang saham dapat dimintai pertanggungjawaban pribadi (piercing the corporate veil)?",
                "Apa langkah hukum yang dapat diambil kreditor untuk menagih utang dari PT yang mengalami kesulitan keuangan?",
                "Bagaimana perlindungan hukum bagi kreditor dalam struktur PT?"
            ]
        },
        {
            "id": "case_2", 
            "type": "case_study",
            "scenario": "CV Maju Jaya membuat kontrak supply bahan baku dengan PT Industri Besar senilai Rp 500 juta dengan jangka waktu 6 bulan. Setelah 3 bulan berjalan, CV Maju Jaya tiba-tiba menghentikan pasokan tanpa pemberitahuan yang memadai, menyebabkan PT Industri Besar harus mencari supplier lain dengan harga 30% lebih mahal. PT Industri Besar menuntut ganti rugi sebesar Rp 150 juta.",
            "questions": [
                "Apakah tindakan CV Maju Jaya dapat dikategorikan sebagai wanprestasi? Jelaskan dasar hukumnya.",
                "Jenis ganti rugi apa saja yang dapat diminta PT Industri Besar?",
                "Bagaimana cara menghitung kerugian yang dapat diminta sebagai ganti rugi?",
                "Apa pembelaan hukum yang dapat diajukan CV Maju Jaya?"
            ]
        }
    ]
    
    # Combine all data
    all_data = mc_questions + case_studies
    
    # Save as JSONL
    with open("upa_eval_set.jsonl", "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Create answer keys
    mc_answers = {
        "mc_1": "B",  # PT adalah badan hukum dengan modal terbagi dalam saham
        "mc_2": "D",  # Tidak ada modal minimum menurut UU No. 40 Tahun 2007
        "mc_3": "B",  # RUPS berwenang mengangkat dan memberhentikan direksi
        "mc_4": "B",  # Wanprestasi adalah tidak dipenuhinya kewajiban kontraktual
        "mc_5": "C"   # Unsur PMH: perbuatan, kesalahan, kerugian, hubungan kausal
    }
    
    with open("mc_answer_key.json", "w", encoding="utf-8") as f:
        json.dump(mc_answers, f, indent=2, ensure_ascii=False)
    
    # Create reference answers for case studies
    case_references = {
        "case_1": {
            "reference_points": [
                "Prinsip tanggung jawab terbatas: pemegang saham hanya bertanggung jawab sebatas modal yang disetor",
                "Piercing the corporate veil: dapat terjadi jika ada pencampuran harta, penggunaan PT untuk tujuan tidak sah, atau undercapitalization",
                "Langkah kreditor: penagihan, restrukturisasi utang, atau PKPU/kepailitan",
                "Perlindungan kreditor: hak preferen, jaminan, dan mekanisme hukum kepailitan"
            ]
        },
        "case_2": {
            "reference_points": [
                "Wanprestasi terjadi karena CV Maju Jaya tidak memenuhi kewajiban sesuai kontrak",
                "Ganti rugi: kerugian langsung, keuntungan yang hilang, dan biaya tambahan",
                "Perhitungan: selisih harga supplier baru dan kerugian operasional",
                "Pembelaan: force majeure, kesalahan PT Industri, atau impossibility of performance"
            ]
        }
    }
    
    with open("case_reference_answers.json", "w", encoding="utf-8") as f:
        json.dump(case_references, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample UPA dataset created successfully!")
    print(f"  📊 {len(mc_questions)} multiple choice questions")
    print(f"  📝 {len(case_studies)} case studies")
    print(f"  📁 Files: upa_eval_set.jsonl, mc_answer_key.json, case_reference_answers.json")

if __name__ == "__main__":
    create_sample_upa_dataset()
