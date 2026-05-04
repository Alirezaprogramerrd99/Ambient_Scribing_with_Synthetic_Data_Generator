# Ambient Scribe Project

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
- **📚 Sophisticated RAG System**: Dense, Dense-rerank, Dense-rerank+Query_Expansion, and Dense-rerank+Query_Expansion+medical filtering; all implemented via LlamaIndex
  
- **✅ Comprehensive Validation**: Structural, clinical, and RAG metrics
- **🎲 Diverse Scenario Generation**: Demographics, specialties, complexity
- **📊 Experiment Tracking**: MLflow and Weights & Biases integration
- **🔧 Modular Architecture**: Easy to extend and customize

---

## 🏗️ Architecture

The system operates in three stages: a **teacher pipeline** that generates synthetic training data, a **student pipeline** that fine-tunes small language models on that data and runs RAG-augmented inference, and an **evaluation framework** that benchmarks student performance using automated metrics, uncertainty quantification, and SMILE explainability.
![Architecture](assets/Frame%202.jpg)


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
```

**For the rest of the installation and setup instructions, please refer to the [SETUP_GUIDE.md](SETUP_GUIDE.md) file in the repository.**

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
│   │   ├── retriever.py     # Dense retriever (ChromaDB, superseded by LlamaIndex)
│   │   ├── llama_index_rag.py  # LlamaIndex retriever (used in production)
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
│   ├── student/             # Student fine-tuning and evaluation
│   │   ├── data_prep.py        # Quality filtering, chat template formatting, splits
│   │   ├── trainer.py          # QLoRA fine-tuning (Unsloth + TRL)
│   │   ├── exporter.py         # LoRA merge, GGUF conversion, Ollama registration
│   │   ├── inference_fixed.py  # RAG-augmented inference (LlamaIndex)
│   │   ├── evaluator.py        # 5-way comparison + LLM-as-a-Judge
│   │   ├── compute_medcon.py   # MEDCON via QuickUMLS (WSL)
│   │   ├── logprobs_analysis.py  # Token log-prob uncertainty quantification
│   │   ├── post_eval_metrics.py  # Post-hoc ROUGE/BERTScore/BLEU/MEDCON
│   │   ├── plot_dissertation_v2.py  # Dissertation tables and plots
│   │   ├── plot_rag_ablation.py  # RAG ablation Table C and plots
│   │   ├── plot_medcon.py      # MEDCON visualisations
│   │   ├── run_temperature_experiment.py  # Temperature sensitivity experiment
│   │   └── run_student_pipeline.py  # End-to-end orchestrator
│   └── pipeline/            # Orchestration
│       └── synthetic_data_pipeline.py
├── data/
│   ├── batch_1/ .. batch_9/   # Raw generation batches (250 scenarios each)
│   ├── training_data/        # Generic formatted training split
│   ├── training_data_llama1b/ # Llama-3.2-1B-Instruct formatted split
│   ├── training_data_llama3b/ # Llama-3.2-3B-Instruct formatted split
│   ├── chroma_db/            # ChromaDB vector store (legacy)
│   ├── llama_index_chroma_db/ # LlamaIndex vector store
│   ├── synthetic_output/     # Ad-hoc generation outputs
│   ├── synthetic_output_llama_index/ # LlamaIndex pipeline results
├── medical_knowledge/       # Clinical guidelines
├── final_results/           # Final evaluation outputs used in the dissertation report
│   ├── dissertation_final_main/      # Dissertation plots and tables
│   ├── final_eval_llama1b_main/      # Llama-3.2-1B main evaluation results
│   ├── final_eval_llama1b_rag_ablation_final/
│   ├── final_eval_llama3b_main/      # Llama-3.2-3B main evaluation results
│   ├── final_eval_llama3b_rag_ablation_final/
│   ├── final_eval_phi_main/          # Phi-3.5-mini main evaluation results
│   ├── final_eval_phi_rag_ablation_final/
│   ├── medcon_results_main/          # MEDCON results (main experiments)
│   ├── medcon_rag_ablation_final/    # MEDCON results (RAG ablation)
│   └── rag_ablation_final_final/     # RAG ablation aggregated results
├── tests/
├── notebooks/
│   ├── logprobs_uq_analysis.ipynb    # Per-token log-probability uncertainty quantification analysis
│   └── smile_explainability.ipynb    # SMILE (SHAP-based) explainability analysis of student model outputs
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📚 RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline was implemented using **LlamaIndex** and **ChromaDB** to ground the system in clinical evidence.

### Indexing and Retrieval

Clinical guidelines are processed offline into 512-token chunks with 50-token overlap using a semantic section strategy, then embedded with the `BAAI/bge-base-en-v1.5` bi-encoder (768-dim) and stored in a persistent ChromaDB collection. At generation time, each clinical scenario is expanded by `MedicalQueryProcessor` (synonym injection, ICD-10 terms, body-system keywords) before the top-5 retrieved chunks (cosine similarity threshold 0.35) are prepended to the teacher prompt, explicitly instructing GPT-4o-mini to align its management plans, investigations, and safety-netting advice with the retrieved evidence.

<p align="center">
  <img src="assets/Screenshot%202026-04-21%20152908.jpg" alt="RAG Pipeline" width="850"/>
</p>

| Stage | Component | Detail |
|---|---|---|
| Indexing | `DocumentProcessor` | Text cleaning, abbreviation standardisation |
| Chunking | `DocChunker` | 512-token chunks, 50-token overlap, semantic section strategy |
| Embedding | `EmbeddingManager` | `BAAI/bge-base-en-v1.5` (768-dim bi-encoder) |
| Vector store | ChromaDB | Persistent, `medical_knowledge` collection |
| Query expansion | `MedicalQueryProcessor` | Synonym injection, ICD-10 terms, body-system keywords |
| Retrieval | BGE encoder + cosine search | Top-5, threshold >= 0.35 |

### RAG Ablation Configurations

Four progressive retrieval configurations were compared in the ablation study. A shared query extraction step (`_extract_query_from_dialogue`) is applied first across all configurations: the first 3 patient utterances and up to 2 diagnostic doctor lines are concatenated into a query string capped at 500 characters.

<p align="center">
  <img src="assets/Screenshot%202026-04-30%20150421.jpg" alt="RAG Ablation Configurations" width="850"/>
</p>

| Config | Method | Description |
|---|---|---|
| `dense_only` | BGE + ChromaDB top-5 | Direct cosine similarity retrieval, no reranking (Karpukhin et al., 2020) |
| `dense_rerank` | BGE + ChromaDB top-10 + Cross-Encoder top-5 | Over-fetches 10 candidates, reranked to top-5 by `ms-marco-MiniLM-L-6-v2` (Nogueira & Cho, 2020) |
| `dense_rerank_qe` | MedicalQueryProcessor + BGE + ChromaDB top-20 + Cross-Encoder top-5 | Adds rule-based medical query expansion before encoding; 2x over-fetch (top-20) passed to reranker |
| `full_medical` | All above + Clinical Relevance Filter | Further post-filters reranked candidates by symptom and anatomical entity overlap to produce final top-5 |

---

## Teacher Model

The teacher module (`src/teacher/`) is the core component responsible for generating synthetic clinical dialogue-summary pairs. It is built around an abstract `BaseTeacher` class that standardises the interface across three LLM backends.

### Class Hierarchy

```
BaseTeacher  (src/teacher/base.py - abstract)
├── OllamaTeacher      local inference via Ollama HTTP API
├── OpenAITeacher      cloud inference via OpenAI (GPT-4o, GPT-4o-mini, ...)
└── AnthropicTeacher   cloud inference via Anthropic (Claude)
```

### Generation Workflow

For every clinical scenario the teacher executes the following steps in sequence:

1. **Query expansion** - `MedicalQueryProcessor` enriches the scenario description with clinical synonyms and related terminology before RAG retrieval.
2. **Guideline retrieval** - If a retriever is configured, relevant documents are fetched from the vector store using cosine similarity (default threshold: 0.35). A fallback message is inserted when no documents exceed the threshold.
3. **Prompt construction** - The expanded scenario and retrieved guideline context are injected into a structured clinical prompt template managed by `ClinicalPrompts`.
4. **LLM call with retries** - The configured backend is called with exponential back-off retry logic. The number of attempts is controlled by `max_retries` in `GenerationConfig`.
5. **Response parsing and validation** - The raw JSON response is parsed and validated against Pydantic schemas (`ClinicalDialogue`, `ClinicalSummary`). Malformed responses trigger a retry.
6. **Difficulty assessment** - A second LLM call at temperature 0.3 assigns a complexity score from 1 to 10 following the Woo et al. (2025) methodology. A heuristic fallback based on dialogue length, HPI word count, differential diagnoses, plan complexity, and polypharmacy is used when the LLM call fails.
7. **Sample assembly** - Dialogue, summary, scenario metadata, generation metadata, RAG metadata, and difficulty metadata are combined into a `SyntheticSample` and returned as a `GenerationResult`.

### Supported Backends and Models

| Backend | Class | Supported Models |
|---|---|---|
| Ollama (local) | `OllamaTeacher` | `llama3.1:8b`, `llama3.1:70b`, `mistral:7b`, `mixtral:8x7b`, `phi3:mini`, `medllama2:7b`, `meditron:7b` |
| OpenAI | `OpenAITeacher` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo` |
| Anthropic | `AnthropicTeacher` | `claude-sonnet-4-20250514`, `claude-3-opus-20240229` |

Use `get_recommended_model(use_case, max_vram_gb)` to select an Ollama model automatically based on available VRAM.

### Difficulty Scoring

Each generated sample is assigned a numeric difficulty score following the Woo et al. (2025) methodology. The LLM evaluates the dialogue and summary at low temperature. If that call fails, a heuristic evaluator scores the same features:

| Score | Level | Typical Characteristics |
|---|---|---|
| 1-3 | Low | Single complaint, straightforward diagnosis, short dialogue |
| 4-6 | Medium | Comorbidities present, requires differential reasoning |
| 7-10 | High | Complex polypharmacy, multi-system involvement, red-flag features |

The heuristic fallback evaluates five factors: dialogue turn count, HPI word count, presence of multiple differentials in the assessment, plan comprehensiveness (referrals, monitoring, follow-up), and polypharmacy (three or more medications). Reasoning steps and contributing factors are stored in `DifficultyMetadata` for curriculum learning and data filtering downstream.

### Python API

```python
from src.teacher import create_teacher
from src.knowledge_base import create_rag_system

# Optional: attach a RAG retriever
retriever = create_rag_system(
    backend="llama_index",
    documents_dir="./medical_knowledge",
)

# Create an Ollama teacher (local)
teacher = create_teacher(
    provider="ollama",
    model_name="llama3.1:8b",
    retriever=retriever,
    temperature=0.7,
    max_tokens=4096,
)

# Generate a single sample
result = teacher.generate(
    scenario="62-year-old male with exertional chest pain radiating to the left arm",
    use_rag=True,
    use_llm_difficulty_assessment=True,
)

if result.success:
    sample = result.sample
    print(sample.dialogue_text)
    print(sample.summary.assessment)
    print(f"Difficulty: {sample.difficulty.difficulty_level.value} "
          f"(score {sample.difficulty.difficulty_score})")

# Create an OpenAI teacher (cloud)
teacher_openai = create_teacher(
    provider="openai",
    model_name="gpt-4o-mini",
    retriever=retriever,
)

# Create an Anthropic teacher (cloud)
teacher_anthropic = create_teacher(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    retriever=retriever,
)
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
    rag_backend="llama_index",
    
    # Validation
    enable_validation=True,
    filter_invalid=True,
)
```

---

## Output Format

Generated samples are saved as JSONL (one JSON object per line). Each record is a complete `SyntheticSample` containing the dialogue, clinical summary, and all metadata produced during generation.

### Complete Schema

The following is a real sample from batch_3 (`id: openai_3e55043cf15c`). The dialogue is truncated for readability; actual samples contain the full turn sequence.

```json
{
  "id": "openai_3e55043cf15c",
  "dialogue": [
    {"speaker": "Doctor",  "text": "Good morning, I'm Dr. Smith. What brings you in today?", "turn_number": 0},
    {"speaker": "Patient", "text": "Hi, I've been experiencing palpitations for the past few hours.", "turn_number": 1},
    {"speaker": "Doctor",  "text": "I'm sorry to hear that. Can you tell me more about these palpitations? When did they start?", "turn_number": 2},
    {"speaker": "Patient", "text": "They started this morning, and they feel pretty mild, but I'm also feeling a bit dizzy.", "turn_number": 3},
    {"speaker": "...", "text": "... (25 turns total)", "turn_number": null}
  ],
  "summary": {
    "chief_complaint": "Palpitations for a few hours",
    "history_of_present_illness": "A 22-year-old female presents with palpitations that started this morning, mild in severity. Symptoms are associated with dizziness rated 4/10, chest discomfort, and shortness of breath. Symptoms worsen with spicy food and stress, and improve with antacids.",
    "past_medical_history": "No significant past medical history.",
    "medications": "None.",
    "allergies": "No known drug allergies (NKDA).",
    "social_history": "No smoking, no significant alcohol use, occupation not discussed.",
    "family_history": "Non-contributory.",
    "physical_examination": "Vital signs: BP 120/80 mmHg, HR 90 bpm, SpO2 98%. Cardiac exam shows regular heart sounds. No signs of respiratory distress.",
    "assessment": "Working diagnosis includes palpitations possibly related to anxiety or GERD. Differential diagnoses include acute coronary syndrome, stable angina, and other gastrointestinal causes.",
    "plan": "Perform ECG and troponin tests to rule out acute coronary syndrome. Consider lifestyle modifications; if symptoms persist, discuss potential medications. Follow up in 1-2 weeks, and educate patient on red flag symptoms.",
    "safety_netting": "Patient should seek immediate medical attention if experiencing chest pain lasting over 15 minutes, pain radiating to jaw or arm, sudden severe pain, or any changes in condition.",
    "soap": {
      "S": "22-year-old female with palpitations, dizziness, chest discomfort, and shortness of breath.",
      "O": "BP 120/80 mmHg, HR 90 bpm, SpO2 98%. Regular heart sounds; no respiratory distress.",
      "A": "Palpitations likely due to anxiety or GERD; rule out acute coronary syndrome.",
      "P": "ECG and troponin tests; lifestyle changes; follow-up in 1-2 weeks; educate on red flags."
    }
  },
  "scenario": {
    "scenario_text": "22-year-old female presenting with palpitations for few hours, mild severity, associated with dizziness, chest discomfort, shortness of breath, worse with spicy food, stress, better with antacids, bland diet",
    "specialty": "Cardiology",
    "urgency": "routine",
    "age_group": "young_adult",
    "gender": "female"
  },
  "generation": {
    "model_name": "gpt-4o-mini",
    "model_provider": "openai",
    "temperature": 0.7,
    "timestamp": "2026-02-13T02:20:10.623669",
    "generation_time_seconds": 25.57,
    "prompt_tokens": 2943,
    "completion_tokens": 1337
  },
  "rag": {
    "rag_enabled": true,
    "num_sources_retrieved": 5,
    "sources": ["neurology_headache", "chest_pain_guidelines", "cardiology"],
    "retrieval_scores": [0.7612, 0.7140, 0.7112, 0.7109, 0.7067],
    "context_used": "[Source: cardiology]\n# Cardiology Assessment and Management\n## Acute Coronary Syndrome (ACS)\n### Presentation\n- Typical Pain: Central, crushing chest pain radiating to the left arm, jaw, or neck..."
  },
  "difficulty": {
    "difficulty_score": 5,
    "difficulty_level": "medium",
    "reasoning_steps": [
      "Step 1: Identify presenting complaint of palpitations.",
      "Step 2: Gather relevant history including associated symptoms and triggers.",
      "Step 3: Consider differential diagnoses such as anxiety, GERD, and acute coronary syndrome.",
      "Step 4: Determine appropriate investigations like ECG and troponin tests.",
      "Step 5: Formulate management plan including lifestyle modifications and education on red flags."
    ],
    "clinical_complexity_factors": [
      "Multiple symptoms requiring correlation",
      "Potential for serious underlying conditions",
      "Need for risk stratification due to chest discomfort"
    ],
    "rationale": ""
  },
  "validation": {
    "status": "passed",
    "errors": [],
    "warnings": ["Turn 19: 4 consecutive turns by Doctor"],
    "structural_valid": true,
    "clinical_valid": true,
    "rag_faithfulness": 0.7215
  }
}
```

<!-- ### Clinical Summary Fields

| Field | Required | Description |
|---|---|---|
| `chief_complaint` | Yes | Primary reason for visit (min 3 words) |
| `history_of_present_illness` | Yes | Onset, duration, character, severity, aggravating/relieving factors, associated symptoms |
| `past_medical_history` | No | Relevant prior conditions |
| `medications` | No | Current medications with dosages |
| `allergies` | No | Drug allergies or NKDA |
| `social_history` | No | Smoking, alcohol, occupation, living situation |
| `family_history` | No | Relevant family medical history |
| `physical_examination` | No | Vital signs and examination findings |
| `assessment` | Yes | Working diagnosis or differential diagnoses |
| `plan` | Yes | Tests, treatments, medications, referrals, follow-up |
| `safety_netting` | No | Warning signs and advice on when to return |
| `soap` | No | Parallel SOAP note with S, O, A, P fields | -->

### Training Format

For downstream fine-tuning, samples can be exported in a simplified instruction format using `sample.to_training_format()`:

```json
{
  "id": "ollama_abc123def456",
  "input": "Doctor: Good morning...\nPatient: I've been having chest pain...",
  "output": "{\"chief_complaint\": \"Chest pain for 3 days\", ...}",
  "difficulty": 7,
  "specialty": "Cardiology"
}
```

---

## Generated Data

### Synthetic Data Batches

Nine generation batches were produced using **gpt-4o-mini** as the teacher model with **LlamaIndex RAG** enabled. Each batch targeted **250 scenarios** with a different random seed to maximise clinical diversity. Every batch folder contains:

- `synthetic_data_*.jsonl` - final valid samples
- `scenarios.jsonl` - input scenarios used
- `intermediate_*.jsonl` - checkpoints saved every 8 samples
- `summary_*.json` - run statistics (counts, timings, success rate)
- `benchmark_openai_gpt-4o-mini_*.json` - benchmark quality score
- `benchmark_report_*.md` - human-readable benchmark report
- `vector_store/` - batch-local RAG index

| Batch | Scenarios | Valid | Failed | Success Rate | Run Time | File Size | Bench Score |
|---|---|---|---|---|---|---|---|
| batch_1 | 250 | 181 | 69 | 72.4% | 11,487 s | 1.73 MB | 0.794 |
| batch_2 | 250 | 189 | 61 | 75.6% | 10,864 s | 1.80 MB | 0.791 |
| batch_3 | 250 | 184 | 66 | 73.6% | 10,885 s | 1.75 MB | 0.799 |
| batch_4 | 250 | 190 | 60 | 76.0% | 13,087 s | 1.81 MB | 0.801 |
| batch_5 | 250 | 186 | 64 | 74.4% | 11,761 s | 1.78 MB | 0.816 |
| batch_6 | 250 | 189 | 61 | 75.6% | 10,603 s | 1.80 MB | 0.828 |
| batch_7 | 250 | 186 | 64 | 74.4% | 11,316 s | 1.77 MB | 0.827 |
| batch_8 | 250 | 191 | 59 | 76.4% | 11,103 s | 1.82 MB | 0.830 |
| batch_9 | 250 | 189 | 61 | 75.6% | 11,788 s | 1.80 MB | 0.825 |
| **Total** | **2,250** | **1,685** | **565** | **74.9% avg** | ~102,893 s | ~16.1 MB | |

### Training Datasets

All three training datasets were prepared from the same **1,685 raw samples** collected across the nine batches. After quality filtering (19.4% removed), **1,358 samples** were retained and split 80/10/10. RAG context was included in 50% of training examples. All sequences fit within the 4,096-token limit (no truncation required).

| Dataset | Base Model | Template | Created | Train | Val | Test | Total | Mean Tokens |
|---|---|---|---|---|---|---|---|---|
| `training_data` | generic-used for phi | - | 2026-02-19 | 1,092 | 133 | 133 | 1,358 | 1,263 |
| `training_data_llama1b` | `unsloth/Llama-3.2-1B-Instruct` | llama3 | 2026-03-10 | 1,092 | 133 | 133 | 1,358 | 1,294 |
| `training_data_llama3b` | `unsloth/Llama-3.2-3B-Instruct` | llama3 | 2026-03-09 | 1,092 | 133 | 133 | 1,358 | 1,294 |

The `llama1b` and `llama3b` variants wrap the same filtered corpus in a model-specific chat template, which accounts for the slightly higher mean token count compared to the generic split.

Each training dataset folder contains:

- `train.jsonl` / `val.jsonl` / `test.jsonl` - data splits
- `dataset_info.json` - creation timestamp, model config, split sizes
- `preparation_stats.json` - raw/filtered counts, specialty and difficulty distributions, token statistics

**Specialty distribution** (identical across all three datasets):

| Specialty | Samples |
|---|---|
| General Practice | 514 |
| Gastroenterology | 184 |
| Cardiology | 134 |
| Urology | 121 |
| Dermatology | 113 |
| Neurology | 107 |
| Geriatrics | 66 |
| Endocrinology | 41 |
| Rheumatology | 36 |
| Psychiatry | 27 |
| ENT, Musculoskeletal, Paediatrics, Infectious Disease, Nephrology, Emergency Medicine | 16 |

Difficulty is predominantly **medium** (1,352 samples) with 6 **high** samples. No low-difficulty samples passed the quality filter.

---

## 🎓 Student Pipeline

The student module (`src/student/`) handles fine-tuning, RAG-augmented inference, and evaluation of small language models on the synthetic data generated by the teacher pipeline.

### Module Overview

| File | Class / Entry Point | Purpose |
|---|---|---|
| `data_prep.py` | `TrainingDataPreparator` | Quality filtering, chat template formatting (llama3/phi3/qwen2), stratified 80/10/10 splits |
| `trainer.py` | `StudentTrainer` | QLoRA (4-bit) fine-tuning via Unsloth + TRL |
| `exporter.py` | `ModelExporter` | Merge LoRA adapters into base model, GGUF conversion (q4_k_m), Ollama registration |
| `inference_fixed.py` | `ClinicalScribeInference` | RAG-augmented inference via LlamaIndex, stop-token handling, output cleaning |
| `evaluator.py` | `StudentEvaluator` + `LLMJudge` | 5-way comparative evaluation + LLM-as-a-Judge scoring (GPT-4o-mini) |
| `compute_medcon.py` | `UMLSConceptExtractor` | MEDCON precision/recall/F1 via QuickUMLS (designed to run in WSL Ubuntu) |
| `logprobs_analysis.py` | script | Token log-probability uncertainty quantification |
| `post_eval_metrics.py` | script | Post-hoc ROUGE/BERTScore/BLEU/MEDCON from evaluation JSONs, no model dependency |
| `plot_dissertation_v2.py` | script | Table A (automated metrics), Table B (LLM Judge), radar charts, bar charts, heatmaps |
| `plot_rag_ablation.py` | script | Table C (RAG ablation) and comparison plots (4 RAG configurations) |
| `plot_medcon.py` | script | MEDCON-specific visualisations |
| `run_temperature_experiment.py` | script | Temperature sensitivity experiment (0.0, 0.1, 0.3, 0.7, 1.0) on `ft_rag` config |
| `run_student_pipeline.py` | script | End-to-end orchestrator: Steps 1 (prepare) through 5 (evaluate) |

### Fine-Tuning (QLoRA)

Models are fine-tuned using **4-bit QLoRA** via [Unsloth](https://github.com/unslothai/unsloth) and TRL's `SFTTrainer`. Key defaults from `TrainingConfig`:

| Parameter | Value |
|---|---|
| LoRA rank (`lora_r`) | 32 |
| LoRA alpha | 64 |
| Target modules | q/k/v/o projections + gate/up/down (MLP) |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Batch size | 4 per device + 4 gradient accumulation steps (effective 16) |
| Epochs | 3 |
| Max sequence length | 4,096 tokens |
| Optimiser | AdamW 8-bit |

Models trained: `unsloth/Llama-3.2-1B-Instruct`, `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Phi-3.5-mini-instruct`.

### Evaluation Framework

The evaluator runs five experimental conditions for each model:

| Config | Fine-tuned | RAG |
|---|---|---|
| `baseline` | No | No |
| `rag_only` | No | LlamaIndex (Dense+Rerank+QE) |
| `ft_only` | Yes | No |
| `ft_rag` | Yes | LlamaIndex (Dense+Rerank+QE) |
| `teacher` | N/A | Reference outputs from GPT-4o-mini |

The RAG ablation study tests four progressive retrieval configurations on the `ft_rag` condition:

| Config | Retrieval Method |
|---|---|
| `dense_only` | Dense retrieval only |
| `dense_rerank` | Dense + cross-encoder reranker |
| `dense_rerank_qe` | Dense + reranker + query expansion |
| `full_medical` | Dense + reranker + query expansion + clinical filtering |

Automated metrics: **ROUGE-1/2/L**, **BERTScore**, **BLEU-1-4**, **METEOR**, **MEDCON** (UMLS CUI overlap).

LLM-as-a-Judge dimensions scored 1-5 by GPT-4o-mini: clinical accuracy, completeness, hallucination, clinical safety, coherence, conciseness.

### Running the Student Pipeline

```bash
# Run individual steps
python src/student/run_student_pipeline.py --prepare-data
python src/student/run_student_pipeline.py --train
python src/student/run_student_pipeline.py --export
python src/student/run_student_pipeline.py --test-inference
python src/student/run_student_pipeline.py --evaluate

# Or run all steps sequentially
python src/student/run_student_pipeline.py --all
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

## 📊 Data Availability

The synthetic clinical dialogue-summary dataset generated in this project is publicly available on Zenodo:

> Rashidi, A. (2026). *Synthetic Clinical Dialogue-Summary Dataset for Ambient Scribing SLM Training*. Zenodo. [https://doi.org/10.5281/zenodo.19986637](https://doi.org/10.5281/zenodo.19986637)

The dataset contains 1,685 validated samples across 9 generation batches, covering 20 clinical specialties, generated using GPT-4o-mini with LlamaIndex RAG grounding.

---

## 📑 References

1. Woo, M., et al. (2025). Synthetic data distillation enables the extraction of clinical information at scale. *npj Digital Medicine*.


---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- LlamaIndex for RAG framework
- NICE for clinical guidelines

---

*Built with ❤️ for healthcare AI*
