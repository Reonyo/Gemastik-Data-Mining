"""
Multi-Agent Legal AI System - Production Gradio Interface
Integrates real agents with legal document processing capabilities
"""

import gradio as gr
import time
import json
from typing import Dict, Any, List, Optional
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.workflow.legal_workflow import LegalWorkflowGraph
from src.config.legal_config import LegalAgentConfig
from src.graph.agent_state import AgentState

class LegalAIWorkflow:
    """Production workflow manager for multi-agent legal system"""
    
    def __init__(self):
        # Initialize the actual workflow graph
        self.workflow = LegalWorkflowGraph()
        
        # Initialize logs
        self.reset_logs()
    
    def reset_logs(self):
        """Reset all agent logs"""
        self.supervisor_log = "🎯 Supervisor Agent (llama-3.1-8b-instant)\nInitializing workflow...\n"
        self.assistant_log = "📋 Legal Assistant (gemma2-9b-it)\nReady to extract facts...\n"
        self.researcher_log = "🔍 Legal Researcher (meta-llama/llama-4-maverick-17b)\nChromaDB ready...\n"
        self.lawyer_log = "⚖️ Senior Lawyer (moonshot4-kimi-k2)\nLegal analysis framework loaded...\n"
        self.editor_log = "📄 Legal Editor (meta-llama/llama-4-maverick-17b)\nDocument templates ready...\n"
        self.workflow_log = "🔄 Multi-Agent Workflow\nReady to process legal case...\n"
        self.final_output = ""
    
    def update_log(self, agent_name: str, message: str):
        """Update specific agent log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        
        if agent_name == "supervisor":
            self.supervisor_log += formatted_msg
        elif agent_name == "assistant":
            self.assistant_log += formatted_msg
        elif agent_name == "researcher":
            self.researcher_log += formatted_msg
        elif agent_name == "lawyer":
            self.lawyer_log += formatted_msg
        elif agent_name == "editor":
            self.editor_log += formatted_msg
        
        self.workflow_log += f"[{timestamp}] {agent_name.upper()}: {message}\n"
    
    def process_case(self, case_text: str):
        """Process legal case through multi-agent workflow"""
        
        if not case_text.strip():
            yield ("Please enter a legal case description.", 
                   "Supervisor: No input provided", "Assistant: Waiting...", 
                   "Researcher: Waiting...", "Lawyer: Waiting...", "")
            return
        
        self.reset_logs()
        
        try:
            # Initialize state for the workflow
            self.update_log("workflow", "🎯 Starting multi-agent legal workflow...")
            yield (self.workflow_log, "🎯 Supervisor: Initializing workflow...", 
                   "📋 Assistant: Ready to extract facts...", 
                   "🔍 Researcher: Database ready...", 
                   "⚖️ Lawyer: Analysis framework loaded...", "")
            
            # Execute the real workflow
            self.update_log("workflow", "🔄 Executing LangGraph workflow...")
            yield (self.workflow_log, "🎯 Supervisor: Routing to agents...", 
                   "📋 Assistant: Processing query...", 
                   "🔍 Researcher: Searching knowledge base...", 
                   "⚖️ Lawyer: Analyzing legal issues...", "")
            
            # Execute workflow using the proper method
            results = self.workflow.run_workflow(case_text, max_iterations=3)
            
            # Extract results from workflow results
            if results.get('success', False):
                final_analysis = results.get('final_document', 'Workflow completed successfully.')
                structured_facts = results.get('structured_facts', [])
                legal_analysis = results.get('legal_analysis', [])
                iteration_count = results.get('iteration_count', 0)
            else:
                final_analysis = f"Workflow error: {results.get('error', 'Unknown error')}"
                structured_facts = []
                legal_analysis = []
                iteration_count = 0
                
                
            # Update agent logs with workflow information
            supervisor_msg = f"🎯 Supervisor: Completed {iteration_count} iterations"
            assistant_msg = f"📋 Assistant: Extracted {len(structured_facts)} facts"
            researcher_msg = f"🔍 Researcher: Retrieved documents from knowledge base"
            lawyer_msg = f"⚖️ Lawyer: Generated {len(legal_analysis)} analysis steps"
            
            self.final_output = final_analysis
            self.update_log("workflow", "✅ Multi-agent workflow completed successfully!")
            
            yield (self.workflow_log, supervisor_msg, assistant_msg, 
                   researcher_msg, lawyer_msg, final_analysis)
            
        except Exception as e:
            error_msg = f"❌ Error in workflow: {str(e)}"
            self.update_log("workflow", error_msg)
            yield (self.workflow_log, f"🎯 Supervisor: {error_msg}", 
                   "📋 Assistant: Workflow interrupted", 
                   "🔍 Researcher: Workflow interrupted", 
                   "⚖️ Lawyer: Workflow interrupted", f"Error: {str(e)}")

# Initialize workflow
workflow = LegalAIWorkflow()

# Sample case for demonstration
SAMPLE_CASE = """PT. Bank Bola dunia sebagai Bank yang didirikan menurut Hukum Indonesia berdasarkan Akta Pendirian Perseroan Terbatas No. 7 Tanggal 06 Nopember 1999, yang dibuat dihadapan Notaris Teddy Anwar, SH dengan Pengesahan Menteri Kehakiman No.C-2 12.859.HT.01.01 Tahun 2001, yang diumumkan dalam Lembaran Negara Tahun 2001 Nomor 1800, berkantor Pusat di Jakarta Jl. Sudirman No. 66. 

Pada tanggal 1 Februari 2004, Ali Ali selaku Direktur Utama PT. Bank Bola Dunia melalui Akte Perjanjian Hutang Piutang Nomor 100, yang dibuat dihadapan Notaris Jali Jali, SH memberikan pinjaman uang kepada John Haha dalam kapasitasnya sebagai Direktur Utama PT. Manca Negara yang mempunyai Kantor Cabang di Surabaya, Yogyakarta, dan Medan serta berkantor Pusat di Jakarta Jl. Sabang No. 123, berupa pinjaman uang Rp.120.000.000.000 (seratus dua puluh miliyar rupiah), dengan jangka waktu pengembalian uang selama 2 (dua) tahun.

Dalam perjanjian hutang piutang tanggal 1 Februari 2004, PT. Manca Negara telah menyerahkan jaminan, berupa: 1. Sebidang tanah dan bangunannya, dikenal terletak di Jl. Lalu Lalang No. 99, Jakarta, sebagaimana dinyatakan dalam Sertifikat Hak Milik Nomor 31 seluas 1.000 m2.

Pertanyaan: Buatlah Surat kuasa khusus dari PT. Bank Bola Dunia kepada Advokat Baba, yang beralamat Kantor di Jakarta Jl. Bacang. No. 13?"""

# Create Gradio Interface - Matching Prototype Design
with gr.Blocks(
    title="Multi-Agent Legal AI System", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 800px !important; margin: 0 auto; }
    .block { border-radius: 8px; margin: 8px 0; }
    .summary-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; border-radius: 12px; padding: 1.5rem; 
        text-align: center; margin: 1rem 0;
    }
    """
) as demo:
    
    # Compact Header - Matching Prototype
    gr.Markdown("## 🏛️ Multi-Agent Legal AI System")
    gr.Markdown("*by: EZ4ENCE*")
    
    # Input Section (Portrait layout)
    with gr.Column():
        case_input = gr.Textbox(
            label="📋 Input Kasus Hukum",
            value=SAMPLE_CASE,
            lines=4,
            max_lines=4,
            placeholder="Masukkan studi kasus hukum di sini..."
        )
        
        process_btn = gr.Button(
            "🚀 Proses dengan Multi-Agent System", 
            variant="primary",
            size="lg"
        )
    
    # Output Section (Stacked vertically for portrait)
    with gr.Column():
        workflow_log = gr.Textbox(
            label="🔄 Workflow Progress",
            value="Klik tombol 'Proses' untuk memulai workflow...",
            lines=18,
            max_lines=18,
            interactive=False
        )
        
        # 4 Agent Debug Columns - Matching Prototype Layout
        gr.Markdown("### 🤖 Agents Debug Log")
        with gr.Row():
            supervisor_output = gr.Textbox(
                label="🎯 Supervisor",
                value="Waiting...",
                lines=10,
                max_lines=10,
                interactive=False
            )
            
            assistant_output = gr.Textbox(
                label="📋 Legal Assistant", 
                value="Waiting...",
                lines=10,
                max_lines=10,
                interactive=False
            )
            
            researcher_output = gr.Textbox(
                label="🔍 Legal Researcher",
                value="Waiting...",
                lines=10,
                max_lines=10,
                interactive=False
            )
            
            lawyer_output = gr.Textbox(
                label="⚖️ Senior Lawyer",
                value="Waiting...",
                lines=10,
                max_lines=10,
                interactive=False
            )
        
        final_output = gr.Textbox(
            label="� Final Legal Document",
            lines=8,
            max_lines=8,
            placeholder="Dokumen hukum final akan muncul di sini...",
            show_copy_button=True
        )
    
    # Event handler - Using the Real Workflow
    process_btn.click(
        fn=workflow.process_case,
        inputs=[case_input],
        outputs=[workflow_log, supervisor_output, assistant_output, researcher_output, lawyer_output, final_output]
    )
    
    # Footer - Matching Prototype
    gr.Markdown("""
    ### 🔧 Teknologi: LangGraph, HuggingFace ChromaDB, GroqAPI, LangChain
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,  # Changed port to avoid conflict
        share=True,
        show_error=True,
        debug=True
    )
