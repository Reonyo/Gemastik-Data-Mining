# 🏛️ Multi-Agent Legal AI System

**Gemastik Data Mining Competition - Advanced Legal Document Processing**

A sophisticated multi-agent AI system for Indonesian legal document analysis and processing, implementing directed cyclic graph workflows with specialized legal agents using LangGraph.

## 🎯 Project Overview

This system employs a **5-agent architecture** with LangGraph workflow orchestration:

- **🎯 Supervisor Agent** (llama-3.1-8b-instant): Workflow orchestration and intelligent routing
- **📋 Legal Assistant** (gemma2-9b-it): Fact extraction and case preprocessing  
- **🔍 Legal Researcher** (meta-llama/llama-4-maverick-17b): Legal database search with ChromaDB
- **⚖️ Senior Lawyer** (moonshotai/kimi-k2-instruct): Comprehensive legal analysis with tools
- **📄 Legal Editor** (meta-llama/llama-4-maverick-17b): Legal document generation and formatting

## 🚀 Key Features

### Multi-Agent Workflow
- **LangGraph Implementation** with StateGraph and conditional routing
- **Directed Cyclic Graph** with iterative agent collaboration
- **Intelligent routing** based on workflow state and task completion
- **Real-time state management** across agent interactions
- **Error handling** with graceful fallbacks and recovery

### Legal Intelligence
- **ChromaDB vector database** with Indonesian legal document corpus
- **BAAI/bge-m3 embeddings** for semantic document retrieval
- **Tool integration** including Python interpreter for calculations
- **Comprehensive legal analysis** covering multiple law domains
- **Professional document generation** with proper legal formatting

### Production Interface
- **Gradio web application** with real-time workflow monitoring
- **Live agent collaboration** with debug logs and status updates
- **Professional UI** optimized for legal professionals
- **Responsive design** with clean, intuitive interface

## 📊 Academic Evaluation

### Evaluation Framework
| Metric | Multi-Agent System | Simple RAG | Single LLM |
|--------|-------------------|------------|-----------|
| **MCQ Accuracy** | **89.28%** (250/280) | 83.92% (245/280) | 79.64% (223/280) |
| **MCQ Semantic Score** | **0.73** | 0.71 | 0.70 |
| **Essay Semantic Score** | **0.78** | 0.75 | 0.71 |

### Performance Results

#### Multiple-Choice Questions (280 questions)
- **Multi-Agent System (Proposed)**: 89.28% accuracy with 0.73 semantic reasoning
- **Simple RAG System (Baseline)**: 83.92% accuracy with 0.71 semantic reasoning  
- **Single LLM (Baseline)**: 79.64% accuracy with 0.70 semantic reasoning

#### Case Study Questions (Essay format)
- **Multi-Agent System (Proposed)**: 0.78 semantic reasoning score
- **Simple RAG System (Baseline)**: 0.75 semantic reasoning score
- **Single LLM (Baseline)**: 0.71 semantic reasoning score

### Dataset
- **Complete UPA legal examination dataset**: 280 MCQ + case study questions
- **Indonesian legal document corpus** with 100+ legal documents
- **Comprehensive evaluation** across multiple legal domains
- **LLM-judge semantic evaluation** using standardized scoring

## 🛠️ Technology Stack

### Core Framework
- **LangGraph**: Multi-agent workflow orchestration with StateGraph
- **LangChain**: LLM integration and prompt management
- **ChromaDB**: Vector database for legal document storage
- **Gradio**: Professional web interface
- **Python 3.8+**: Core development language

### Language Models & APIs
- **Groq API**: llama-3.1-8b-instant, gemma2-9b-it
- **OpenRouter API**: meta-llama/llama-4-maverick-17b, moonshotai/kimi-k2-instruct
- **Tavily Search**: External legal information retrieval
- **BAAI/bge-m3**: Semantic embeddings for document retrieval

### Data Processing
- **PyPDF2**: PDF document parsing and processing
- **NLTK**: Text preprocessing and tokenization
- **Pandas**: Dataset management and evaluation
- **ChromaDB**: Vector storage and similarity search

## 📁 Project Structure

```
Gemastik-Data-Mining/
├── src/                           # Core source code
│   ├── workflow/                  # LangGraph workflow implementation
│   │   ├── legal_workflow.py      # Main StateGraph workflow
│   │   └── __init__.py
│   ├── agents/                    # Specialized agent implementations
│   │   ├── base_agent.py          # Base agent class with LLM integration
│   │   ├── supervisor.py          # Workflow orchestrator
│   │   ├── assistant.py           # Fact extraction specialist
│   │   ├── researcher.py          # Legal database searcher
│   │   ├── lawyer.py              # Legal analysis expert
│   │   ├── editor.py              # Document generation specialist
│   │   └── __init__.py
│   ├── graph/                     # State management
│   │   ├── agent_state.py         # Workflow state definitions
│   │   └── __init__.py
│   ├── data_processing/           # Data handling and vectorization
│   │   ├── vector_indexer.py      # ChromaDB integration
│   │   ├── document_processor.py  # PDF processing utilities
│   │   └── __init__.py
│   ├── tools/                     # Agent tools and utilities
│   │   ├── search_tools.py        # Internal/external search tools
│   │   ├── python_interpreter.py  # Code execution tool
│   │   └── __init__.py
│   └── config/                    # Configuration management
│       ├── model_config.py        # LLM configurations
│       └── __init__.py
├── evaluation/                    # Academic evaluation framework
│   ├── datasets/                  # Evaluation datasets
│   ├── evaluation_runner.py       # Main evaluation script
│   └── metrics.py                 # Evaluation metrics
├── data/                          # Data storage (gitignored)
│   ├── corpus/                    # Legal document corpus
│   ├── vector_db/                 # ChromaDB storage
│   ├── processed/                 # Processed datasets
│   └── evaluasi/                  # Evaluation datasets
├── legal_ai_app.py               # Main Gradio application
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This documentation
```

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Reonyo/Gemastik-Data-Mining.git
cd Gemastik-Data-Mining
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_HUB_TOKEN=your_hf_token_here
```

### 5. Add Legal Documents
**⚠️ Important**: You need to add your own legal documents to the `data/corpus/` directory. 
The legal document corpus is not included in this repository due to file size limitations.

Add PDF legal documents (Indonesian law) to:
```
data/corpus/
├── your_legal_document_1.pdf
├── your_legal_document_2.pdf
└── ...
```

### 6. Initialize Vector Database
The system will automatically initialize ChromaDB on first run with any legal documents in `data/corpus/`.

### 7. Launch Application
```bash
python legal_ai_app.py
```

Access the interface at `http://localhost:7865`

## 💻 Usage

### Web Interface
1. **Input**: Enter legal case description in the text area
2. **Process**: Click "Analyze Legal Case" to start multi-agent workflow
3. **Monitor**: Watch real-time agent collaboration in the workflow log
4. **Result**: Receive comprehensive legal analysis with structured output

### Example Input
```
Seorang pengusaha membuat kontrak dengan supplier untuk pengadaan barang. 
Namun supplier tidak mengirim barang sesuai jadwal yang disepakati. 
Pengusaha tersebut ingin mengetahui hak-haknya dan langkah hukum yang dapat diambil.
```

### API Integration
```python
from src.workflow.legal_workflow import LegalWorkflowGraph

# Initialize workflow
workflow = LegalWorkflowGraph()

# Process case
results = workflow.run_workflow(
    user_query="Your legal case description here...",
    max_iterations=3
)

print(results['final_document'])
```

### Evaluation
```bash
# Run evaluation on test dataset
python evaluation/evaluation_runner.py
```

## 📊 Performance Metrics

### Evaluation Results
- **MCQ Performance**: **89.28% accuracy** (250/280 questions) on complete legal examination dataset
- **MCQ Semantic Reasoning**: **0.73 average score** (LLM-judge evaluation)
- **Essay Evaluation**: **0.78 average semantic score** for case study questions
- **Improvement over baselines**: +5.36% vs Simple RAG, +9.64% vs Single LLM

### System Capabilities
- **Multi-Agent Collaboration**: 5 specialized agents with defined roles
- **Workflow Iterations**: Average 2-3 iterations per case
- **Document Retrieval**: Semantic search across legal corpus
- **Response Quality**: Professional legal analysis with citations
- **Fact Extraction**: High precision structured information extraction
- **Legal Analysis**: Comprehensive multi-step reasoning with legal basis

## 🏗️ Architecture Details

### LangGraph Workflow
```python
# Workflow structure
workflow = StateGraph(dict)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("legal_assistant", assistant_agent)
workflow.add_node("legal_researcher", researcher_agent)
workflow.add_node("senior_lawyer", lawyer_agent)
workflow.add_node("legal_editor", editor_agent)

# Conditional routing based on state
workflow.add_conditional_edges("supervisor", route_function)
```

### Agent State Management
```python
class AgentState:
    user_query: str
    structured_facts: List[StructuredFact]
    retrieved_documents: List[RetrievedDocument]
    legal_analysis: List[AnalysisStep]
    final_document: str
    iteration_count: int
    is_complete: bool
```

## 🤝 Contributing

### Development Guidelines
1. Fork the repository and create a feature branch
2. Follow Python PEP 8 style guidelines
3. Add type hints and docstrings for new functions
4. Test changes with the evaluation framework
5. Submit pull request with clear description

### Adding New Agents
1. Inherit from `BaseAgent` class
2. Implement required methods: `initialize`, `execute`
3. Add agent to workflow graph in `legal_workflow.py`
4. Update routing logic in supervisor

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Gemastik Competition** for providing the research framework
- **LangChain/LangGraph** for multi-agent workflow capabilities
- **Groq & OpenRouter** for LLM API access
- **ChromaDB** for vector database infrastructure
- **HuggingFace** for embedding models and tokenizers

## 📚 Citation

```bibtex
@misc{legal_ai_multiagent_2024,
  title={Multi-Agent Legal AI System for Indonesian Law using LangGraph},
  author={Reyner Ongkowijoyo},
  year={2024},
  howpublished={Gemastik Data Mining Competition},
  url={https://github.com/Reonyo/Gemastik-Data-Mining}
}
```

## 📞 Contact

**Team EZ4ENCE - Gemastik Data Mining 2024**
- Repository: [https://github.com/Reonyo/Gemastik-Data-Mining](https://github.com/Reonyo/Gemastik-Data-Mining)
- Competition: Gemastik XVII - Data Mining Category

---

<div align="center">
<b>🏛️ Multi-Agent Legal AI System | Indonesian Law Processing | Gemastik 2024 🏛️</b>
</div>

---

**Built with LangGraph** - Advanced multi-agent workflows for legal AI
