"""
Main Preprocessor Orchestrator
Coordinates the preprocessing of both legal knowledge base and evaluation dataset.
"""

import sys
import json
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.pdf_parser import LegalDocumentParser
from .upa_parser_final import parse_upa_pdf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MainPreprocessor:
    """Main orchestrator for all preprocessing tasks."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Input directories
        self.legal_docs_dir = self.base_dir / "data" / "raw_legal_docs"
        self.evaluation_dir = self.base_dir / "data" / "raw_evaluation_dataset"
        
        # Output files
        self.legal_output = self.base_dir / "data" / "processed" / "legal_knowledge_base.jsonl"
        self.evaluation_output = self.base_dir / "data" / "processed" / "evaluation_dataset.jsonl"
        
        # Ensure output directory exists
        self.legal_output.parent.mkdir(parents=True, exist_ok=True)
    
    def check_dependencies(self) -> bool:
        """Check if required directories and files exist."""
        missing = []
        
        if not self.legal_docs_dir.exists():
            missing.append(f"Legal documents directory: {self.legal_docs_dir}")
        
        if not self.evaluation_dir.exists():
            missing.append(f"Evaluation dataset directory: {self.evaluation_dir}")
        
        if missing:
            logger.error("Missing required directories:")
            for item in missing:
                logger.error(f"  - {item}")
            return False
        
        return True
    
    def preprocess_legal_knowledge_base(self) -> bool:
        """Process the legal knowledge base documents."""
        logger.info("=" * 60)
        logger.info("STARTING LEGAL KNOWLEDGE BASE PREPROCESSING")
        logger.info("=" * 60)
        
        try:
            parser = LegalDocumentParser(
                input_dir=str(self.legal_docs_dir),
                output_file=str(self.legal_output)
            )
            parser.run()
            
            logger.info(f"✅ Legal knowledge base preprocessing completed!")
            logger.info(f"📄 Output saved to: {self.legal_output}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in legal knowledge base preprocessing: {e}")
            return False
    
    def preprocess_evaluation_dataset(self) -> bool:
        """Process the evaluation dataset."""
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION DATASET PREPROCESSING")
        logger.info("=" * 60)
        
        try:
            # Process evaluation files directly using the final parser
            pdf_files = list(self.evaluation_dir.glob("*.pdf"))
            all_questions = []
            
            for pdf_file in pdf_files:
                logger.info(f"Processing {pdf_file.name}")
                questions = parse_upa_pdf(pdf_file)
                logger.info(f"Extracted {len(questions)} questions from {pdf_file.name}")
                all_questions.extend(questions)
            
            # Save results
            self.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
            with open(self.evaluation_output, 'w', encoding='utf-8') as f:
                for question in all_questions:
                    json_line = json.dumps(question, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            logger.info(f"✅ Evaluation dataset preprocessing completed!")
            logger.info(f"📄 Output saved to: {self.evaluation_output}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in evaluation dataset preprocessing: {e}")
            return False
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of the preprocessing results."""
        logger.info("=" * 60)
        logger.info("PREPROCESSING SUMMARY REPORT")
        logger.info("=" * 60)
        
        # Check legal knowledge base output
        if self.legal_output.exists():
            try:
                with open(self.legal_output, 'r', encoding='utf-8') as f:
                    legal_count = sum(1 for _ in f)
                logger.info(f"📚 Legal Knowledge Base: {legal_count} chunks processed")
            except Exception as e:
                logger.error(f"Error reading legal output: {e}")
        else:
            logger.warning("⚠️ Legal knowledge base output file not found")
        
        # Check evaluation dataset output
        if self.evaluation_output.exists():
            try:
                with open(self.evaluation_output, 'r', encoding='utf-8') as f:
                    eval_count = sum(1 for _ in f)
                logger.info(f"📝 Evaluation Dataset: {eval_count} questions processed")
            except Exception as e:
                logger.error(f"Error reading evaluation output: {e}")
        else:
            logger.warning("⚠️ Evaluation dataset output file not found")
        
        logger.info("=" * 60)
        logger.info("📂 Output files location:")
        logger.info(f"  Legal Knowledge Base: {self.legal_output}")
        logger.info(f"  Evaluation Dataset: {self.evaluation_output}")
        logger.info("=" * 60)
    
    def run_all(self) -> None:
        """Run complete preprocessing pipeline."""
        logger.info("🚀 Starting Complete Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed. Aborting preprocessing.")
            return
        
        # Track success
        legal_success = False
        eval_success = False
        
        # Process legal knowledge base
        legal_success = self.preprocess_legal_knowledge_base()
        
        # Process evaluation dataset
        eval_success = self.preprocess_evaluation_dataset()
        
        # Generate summary
        self.generate_summary_report()
        
        # Final status
        if legal_success and eval_success:
            logger.info("🎉 All preprocessing tasks completed successfully!")
        elif legal_success or eval_success:
            logger.warning("⚠️ Some preprocessing tasks completed with issues.")
        else:
            logger.error("❌ Preprocessing pipeline failed.")
    
    def run_legal_only(self) -> None:
        """Run only legal knowledge base preprocessing."""
        logger.info("🚀 Starting Legal Knowledge Base Preprocessing Only")
        
        if not self.legal_docs_dir.exists():
            logger.error(f"❌ Legal documents directory not found: {self.legal_docs_dir}")
            return
        
        success = self.preprocess_legal_knowledge_base()
        
        if success:
            logger.info("🎉 Legal knowledge base preprocessing completed!")
        else:
            logger.error("❌ Legal knowledge base preprocessing failed.")
    
    def run_evaluation_only(self) -> None:
        """Run only evaluation dataset preprocessing."""
        logger.info("🚀 Starting Evaluation Dataset Preprocessing Only")
        
        if not self.evaluation_dir.exists():
            logger.error(f"❌ Evaluation dataset directory not found: {self.evaluation_dir}")
            return
        
        success = self.preprocess_evaluation_dataset()
        
        if success:
            logger.info("🎉 Evaluation dataset preprocessing completed!")
        else:
            logger.error("❌ Evaluation dataset preprocessing failed.")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess legal documents and evaluation dataset")
    parser.add_argument(
        "--mode",
        choices=["all", "legal", "evaluation"],
        default="all",
        help="Processing mode: all (default), legal only, or evaluation only"
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory path (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MainPreprocessor(args.base_dir)
    
    # Run based on mode
    if args.mode == "all":
        preprocessor.run_all()
    elif args.mode == "legal":
        preprocessor.run_legal_only()
    elif args.mode == "evaluation":
        preprocessor.run_evaluation_only()


if __name__ == "__main__":
    main()
