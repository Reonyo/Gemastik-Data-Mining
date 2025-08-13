"""
Indonesian Legal Document PDF Parser
Extracts and chunks legal documents by Pasal (Article) and ayat (clause)
with special handling for Penjelasan (Elucidation) sections.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
from docx import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalDocumentParser:
    """Parser for Indonesian legal documents with structure-aware chunking."""
    
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.chunks = []
        
        # Regex patterns for Indonesian legal document structure
        self.pasal_pattern = re.compile(r'Pasal\s+(\d+[a-zA-Z]*)', re.IGNORECASE)
        self.ayat_pattern = re.compile(r'\((\d+)\)', re.MULTILINE)
        self.penjelasan_pattern = re.compile(r'PENJELASAN', re.IGNORECASE)
        self.bab_pattern = re.compile(r'BAB\s+([IVX]+)', re.IGNORECASE)
        
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_text_from_doc(self, file_path: Path) -> str:
        """Extract text from DOC file (legacy Word format)."""
        try:
            # For .DOC files, we'll try to read as text (basic approach)
            # In production, you might want to use python-docx2txt or similar
            with open(file_path, 'rb') as file:
                content = file.read()
                # Simple text extraction (this is basic and may need improvement)
                text = content.decode('utf-8', errors='ignore')
                # Clean up binary artifacts
                text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and common artifacts
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Halaman\s+\d+', '', text, flags=re.IGNORECASE)
        # Clean line breaks
        text = text.replace('\n\n\n', '\n\n')
        return text.strip()
    
    def find_penjelasan_section(self, text: str) -> Tuple[str, str]:
        """Separate main content from Penjelasan section."""
        penjelasan_match = self.penjelasan_pattern.search(text)
        
        if penjelasan_match:
            main_content = text[:penjelasan_match.start()].strip()
            penjelasan_content = text[penjelasan_match.start():].strip()
            return main_content, penjelasan_content
        else:
            return text, ""
    
    def extract_pasal_chunks(self, text: str, source_file: str, is_penjelasan: bool = False) -> List[Dict[str, Any]]:
        """Extract chunks based on Pasal (Article) structure."""
        chunks = []
        
        # Find all Pasal matches
        pasal_matches = list(self.pasal_pattern.finditer(text))
        
        if not pasal_matches:
            # If no Pasal found, treat as single chunk
            if text.strip():
                chunk = {
                    "content": text.strip(),
                    "metadata": {
                        "source_document": source_file,
                        "pasal": None,
                        "ayat": None,
                        "is_penjelasan": is_penjelasan,
                        "chunk_type": "document_section"
                    }
                }
                chunks.append(chunk)
            return chunks
        
        # Process each Pasal
        for i, pasal_match in enumerate(pasal_matches):
            pasal_number = pasal_match.group(1)
            start_pos = pasal_match.start()
            
            # Determine end position (start of next Pasal or end of text)
            if i + 1 < len(pasal_matches):
                end_pos = pasal_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            pasal_text = text[start_pos:end_pos].strip()
            
            # Extract ayat within this Pasal
            ayat_chunks = self.extract_ayat_chunks(pasal_text, source_file, pasal_number, is_penjelasan)
            
            if ayat_chunks:
                chunks.extend(ayat_chunks)
            else:
                # If no ayat found, treat entire Pasal as one chunk
                chunk = {
                    "content": pasal_text,
                    "metadata": {
                        "source_document": source_file,
                        "pasal": pasal_number,
                        "ayat": None,
                        "is_penjelasan": is_penjelasan,
                        "chunk_type": "pasal"
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def extract_ayat_chunks(self, pasal_text: str, source_file: str, pasal_number: str, is_penjelasan: bool) -> List[Dict[str, Any]]:
        """Extract ayat (clause) chunks within a Pasal."""
        chunks = []
        ayat_matches = list(self.ayat_pattern.finditer(pasal_text))
        
        if not ayat_matches:
            return []  # No ayat found
        
        # Process each ayat
        for i, ayat_match in enumerate(ayat_matches):
            ayat_number = ayat_match.group(1)
            start_pos = ayat_match.start()
            
            # Determine end position
            if i + 1 < len(ayat_matches):
                end_pos = ayat_matches[i + 1].start()
            else:
                end_pos = len(pasal_text)
            
            ayat_text = pasal_text[start_pos:end_pos].strip()
            
            chunk = {
                "content": ayat_text,
                "metadata": {
                    "source_document": source_file,
                    "pasal": pasal_number,
                    "ayat": ayat_number,
                    "is_penjelasan": is_penjelasan,
                    "chunk_type": "ayat"
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single document and return chunks."""
        logger.info(f"Processing document: {file_path.name}")
        
        # Extract text based on file extension
        if file_path.suffix.lower() == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            raw_text = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() == '.doc':
            raw_text = self.extract_text_from_doc(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return []
        
        if not raw_text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        # Clean the text
        clean_text = self.clean_text(raw_text)
        
        # Separate main content from Penjelasan
        main_content, penjelasan_content = self.find_penjelasan_section(clean_text)
        
        chunks = []
        
        # Process main content
        if main_content:
            main_chunks = self.extract_pasal_chunks(main_content, file_path.name, is_penjelasan=False)
            chunks.extend(main_chunks)
        
        # Process Penjelasan section
        if penjelasan_content:
            penjelasan_chunks = self.extract_pasal_chunks(penjelasan_content, file_path.name, is_penjelasan=True)
            chunks.extend(penjelasan_chunks)
        
        logger.info(f"Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks
    
    def process_all_documents(self) -> None:
        """Process all documents in the input directory."""
        logger.info(f"Starting to process documents from {self.input_dir}")
        
        # Get all PDF and DOC files
        file_patterns = ['*.pdf', '*.PDF', '*.doc', '*.DOC', '*.docx', '*.DOCX']
        files = []
        for pattern in file_patterns:
            files.extend(self.input_dir.glob(pattern))
        
        if not files:
            logger.error(f"No supported documents found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(files)} documents to process")
        
        # Process each document
        total_chunks = 0
        for file_path in files:
            if file_path.name.startswith('.'):  # Skip hidden files
                continue
                
            document_chunks = self.process_document(file_path)
            self.chunks.extend(document_chunks)
            total_chunks += len(document_chunks)
        
        logger.info(f"Total chunks extracted: {total_chunks}")
    
    def save_to_jsonl(self) -> None:
        """Save all chunks to JSONL file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                json_line = json.dumps(chunk, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Saved {len(self.chunks)} chunks to {self.output_file}")
    
    def run(self) -> None:
        """Main processing pipeline."""
        self.process_all_documents()
        self.save_to_jsonl()


def main():
    """Main function to run the legal document parser."""
    input_dir = "data/raw_legal_docs"
    output_file = "data/processed/legal_knowledge_base.jsonl"
    
    parser = LegalDocumentParser(input_dir, output_file)
    parser.run()


if __name__ == "__main__":
    main()