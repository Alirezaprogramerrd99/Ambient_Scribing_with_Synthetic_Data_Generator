# Ambient Scribe Teacher

> RAG-Enhanced Teacher Model for Synthetic Clinical Data Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for generating high-quality synthetic clinical dialogue-summary pairs for training Small Language Models (SLMs) in ambient clinical scribing.

**Author:** Alireza Rashidi  
**MSc Project:** Trustworthy SLMs for Ambient Clinical Scribing

**Based on:** Woo et al. (2025) - *Synthetic data distillation enables the extraction of clinical information at scale* (npj Digital Medicine)

---

## 🎯 Overview

This project implements a **teacher-student knowledge distillation** approach for training privacy-preserving AI medical scribes that can:

1. Convert doctor-patient dialogues into structured clinical summaries
2. Run locally on NHS hardware without cloud dependencies
3. Ground outputs in clinical guidelines via RAG

### Key Features

- **🤖 Multiple LLM Backends**: Ollama (local), OpenAI, Anthropic
- **📚 Dual RAG System**: Manual implementation + LlamaIndex
- **✅ Comprehensive Validation**: Structural, clinical, and RAG metrics
- **🎲 Diverse Scenario Generation**: Demographics, specialties, complexity
- **📊 Experiment Tracking**: MLflow and Weights & Biases integration
- **🔧 Modular Architecture**: Easy to extend and customize

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Synthetic Data Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Scenario   │───▶│    RAG       │───▶│   Teacher    │       │
│  │  Generator   │    │  Retriever   │    │    Model     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Scenarios   │    │  Clinical    │    │  Synthetic   │       │
│  │   (JSON)     │    │  Guidelines  │    │   Samples    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │  Validation  │       │
│                                          │   Pipeline   │       │
│                                          └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │   Training   │       │
│                                          │    Data      │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM inference)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ambient-scribe-teacher.git
cd ambient-scribe-teacher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your settings

# Install Ollama and pull a model
ollama pull llama3.1:8b
```

### Optional: Install Advanced NLP

```bash
# For clinical NER
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz

# For RAG evaluation
pip install ragas

# For LlamaIndex RAG
pip install llama-index llama-index-embeddings-huggingface llama-index-vector-stores-chroma
```

---

## 🚀 Quick Start

### Python API

```python
from src.pipeline import SyntheticDataPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    num_scenarios=100,
    teacher_model="llama3.1:8b",
    teacher_provider="ollama",
    use_rag=True,
    knowledge_base_path="./medical_knowledge",
    output_dir="./data/synthetic_output",
)

# Run pipeline
pipeline = SyntheticDataPipeline(config)
result = pipeline.run()

print(f"Generated {result.total_valid} valid samples")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Output: {result.output_path}")
```

### Command Line

```bash
# Basic generation
python -m src.pipeline.synthetic_data_pipeline \
    --num-scenarios 100 \
    --provider ollama \
    --model llama3.1:8b \
    --output-dir ./output

# With RAG
python -m src.pipeline.synthetic_data_pipeline \
    --num-scenarios 100 \
    --use-rag \
    --knowledge-base ./medical_knowledge \
    --output-dir ./output

# Using cloud API
python -m src.pipeline.synthetic_data_pipeline \
    --num-scenarios 50 \
    --provider openai \
    --model gpt-4o \
    --output-dir ./output
```

---

## 📁 Project Structure

```
ambient_scribe_teacher/
├── src/
│   ├── config/              # Configuration management
│   │   ├── settings.py      # Pydantic settings
│   │   ├── prompt_manager.py
│   │   └── prompts/         # Prompt templates
│   ├── models/              # Data schemas
│   │   ├── schemas.py       # Pydantic models
│   │   └── enums.py         # Clinical enumerations
│   ├── utils/               # Utilities
│   │   ├── retry.py         # Retry logic
│   │   └── logging_utils.py # MLflow/W&B tracking
│   ├── knowledge_base/      # RAG components
│   │   ├── document_processor.py
│   │   ├── indexer.py       # Vector store indexing
│   │   ├── retriever.py     # Manual retriever
│   │   ├── llama_index_rag.py  # LlamaIndex retriever
│   │   └── rag_factory.py   # Factory for switching
│   ├── teacher/             # LLM teachers
│   │   ├── base.py          # Abstract base class
│   │   ├── ollama_teacher.py
│   │   └── api_teacher.py   # OpenAI/Anthropic
│   ├── validation/          # Quality validation
│   │   ├── structural.py    # Structure checks
│   │   ├── clinical.py      # Medical validation
│   │   └── rag_metrics.py   # RAG quality
│   ├── scenarios/           # Scenario generation
│   │   └── generator.py
│   └── pipeline/            # Orchestration
│       └── synthetic_data_pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic_output/
├── medical_knowledge/       # Clinical guidelines
├── tests/
├── notebooks/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### Pipeline Configuration

```python
from src.pipeline import PipelineConfig
from src.models import DifficultyLevel, ClinicalSpecialty

config = PipelineConfig(
    # Scenarios
    num_scenarios=100,
    specialties=[
        ClinicalSpecialty.CARDIOLOGY,
        ClinicalSpecialty.RESPIRATORY,
    ],
    complexity_distribution={
        DifficultyLevel.LOW: 0.3,
        DifficultyLevel.MEDIUM: 0.5,
        DifficultyLevel.HIGH: 0.2,
    },
    
    # Teacher
    teacher_provider="ollama",
    teacher_model="llama3.1:8b",
    temperature=0.7,
    
    # RAG
    use_rag=True,
    rag_backend="llama_index",  # or "manual"
    
    # Validation
    enable_validation=True,
    filter_invalid=True,
)
```

---

## 📊 Output Format

Generated samples are saved as JSONL with the following structure:

```json
{
  "id": "ollama_abc123def456",
  "dialogue": [
    {"speaker": "Doctor", "text": "Good morning, what brings you in today?"},
    {"speaker": "Patient", "text": "I've been having chest pain for 3 days..."}
  ],
  "summary": {
    "chief_complaint": "Chest pain for 3 days",
    "history_of_present_illness": "55-year-old male presents with...",
    "assessment": "Suspected stable angina",
    "plan": "Order ECG and troponins...",
    "soap": {
      "S": "...",
      "O": "...",
      "A": "...",
      "P": "..."
    }
  },
  "scenario": {
    "specialty": "Cardiology",
    "urgency": "routine"
  },
  "difficulty": {
    "difficulty_score": 5,
    "difficulty_level": "medium"
  },
  "rag": {
    "rag_enabled": true,
    "sources": ["nice_guidelines/chest_pain.txt"]
  }
}
```

---

## 🧪 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

---

## 📚 References

1. Woo, M., et al. (2025). Synthetic data distillation enables the extraction of clinical information at scale. *npj Digital Medicine*.

2. GitHub: [bbj-lab/clinical-synthetic-data-distil](https://github.com/bbj-lab/clinical-synthetic-data-distil)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- LlamaIndex for RAG framework
- NICE for clinical guidelines

---

*Built with ❤️ for healthcare AI*
