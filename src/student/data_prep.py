"""
Data Preparation for Student Model Fine-Tuning (Multi-Model Support)

Loads synthetic data from the teacher pipeline, applies quality filters,
converts to model-specific chat template format, and creates
stratified train/val/test splits.

Supported models:
    - Phi-3.5-mini-instruct  (ChatML: <|system|>...<|end|>)
    - Qwen2.5-3B-Instruct    (ChatML: <|im_start|>...<|im_end|>)
    - Generic HuggingFace     (uses tokenizer.apply_chat_template)

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing

Changes (on March 2026):
    - Added model_type parameter for multi-SLM comparison
    - Template auto-detection from model name
    - Qwen2.5 <|im_start|>/<|im_end|> template support
    - Generic apply_chat_template fallback
"""

import patch_torch

import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Chat Template Registry
# =============================================================================

# REMEMBER:
# Phi-3.5 uses: <|system|>\n{content}<|end|>\n<|user|>\n{content}<|end|>\n<|assistant|>\n{content}<|end|>
# Qwen2.5 uses: <|im_start|>system\n{content}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>

CHAT_TEMPLATES = {
    "phi3": {
        "system_prefix": "<|system|>\n",
        "system_suffix": "<|end|>\n",
        "user_prefix": "<|user|>\n",
        "user_suffix": "<|end|>\n",
        "assistant_prefix": "<|assistant|>\n",
        "assistant_suffix": "<|end|>",
    },
    "qwen2": {
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>",
    },
}

# Map model names to template keys
MODEL_TEMPLATE_MAP = {
    "phi-3.5": "phi3",
    "phi3.5": "phi3",
    "phi-3": "phi3",
    "phi3": "phi3",
    "unsloth/Phi-3.5-mini-instruct": "phi3",
    "microsoft/Phi-3.5-mini-instruct": "phi3",
    "qwen2.5": "qwen2",
    "qwen2": "qwen2",
    "qwen": "qwen2",
    "unsloth/Qwen2.5-3B-Instruct": "qwen2",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2",
}


def detect_template(model_name: str) -> str:
    """
    Detect the chat template key from a model name.
    
    Args:
        model_name: HuggingFace model name or short alias.
    
    Returns:
        Template key (e.g., 'phi3', 'qwen2').
    
    Raises:
        ValueError if model cannot be mapped to a known template.
    """
    model_lower = model_name.lower()
    
    # Exact match first
    if model_name in MODEL_TEMPLATE_MAP:
        return MODEL_TEMPLATE_MAP[model_name]
    
    # Substring match
    for key, template in MODEL_TEMPLATE_MAP.items():
        if key in model_lower:
            return template
    
    raise ValueError(
        f"Cannot auto-detect chat template for '{model_name}'. "
        f"Known models: {list(MODEL_TEMPLATE_MAP.keys())}. "
        f"Set model_type explicitly in DataPrepConfig."
    )


def format_chat_template(
    system_msg: str,
    user_msg: str,
    assistant_msg: str,
    template_key: str,
) -> str:
    """
    Format a conversation into the model-specific chat template.
    
    Args:
        system_msg: System prompt text.
        user_msg: User message text.
        assistant_msg: Assistant response text.
        template_key: One of 'phi3', 'qwen2'.
    
    Returns:
        Formatted string ready for tokenization and training.
    """
    tmpl = CHAT_TEMPLATES[template_key]
    
    return (
        f"{tmpl['system_prefix']}{system_msg}{tmpl['system_suffix']}"
        f"{tmpl['user_prefix']}{user_msg}{tmpl['user_suffix']}"
        f"{tmpl['assistant_prefix']}{assistant_msg}{tmpl['assistant_suffix']}"
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataPrepConfig:
    """Configuration for data preparation."""
    
    # Input
    raw_data_dirs: List[str] = field(default_factory=lambda: ["./data/synthetic_output_llama_index"])
    
    # Output
    output_dir: str = "./data/training_data"
    
    # Model template selection
    # Options: "phi3", "qwen2", or "auto" (auto-detect from base_model)
    model_type: str = "auto"
    base_model: str = "unsloth/Phi-3.5-mini-instruct"
    
    # Filtering thresholds
    min_dialogue_turns: int = 8
    min_doctor_questions: int = 3
    require_both_speakers: bool = True
    require_soap_complete: bool = True
    require_validation_passed: bool = True
    require_clinical_valid: bool = True
    
    # RAG context inclusion
    rag_context_ratio: float = 0.5  # 50% with RAG, 50% without
    
    # Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Tokenisation
    max_seq_length: int = 4096
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a doctor-patient "
    "conversation and relevant clinical guidelines, produce a structured "
    "clinical summary in the specified format. Be accurate and concise. "
    "Only include information explicitly stated in the conversation. "
    "Do not fabricate symptoms, medications, or findings."
)

SUMMARY_INSTRUCTION = (
    "Produce a structured clinical summary with the following sections:\n"
    "- Chief Complaint\n"
    "- History of Present Illness\n"
    "- Past Medical History\n"
    "- Medications\n"
    "- Allergies\n"
    "- Examination Findings\n"
    "- Assessment\n"
    "- Plan\n"
    "- Safety Netting"
)


# =============================================================================
# Helper Functions
# =============================================================================

def _load_samples_from_dir(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all synthetic sample files from a directory.
    
    Handles the actual pipeline output structure:
        batch_N/
        ├── synthetic_data_*.jsonl    ← primary: final combined output
        ├── intermediate_*.jsonl      ← fallback: checkpoint files  
        ├── benchmark_*.json          ← skip (not sample data)
        ├── summary_*.json            ← skip (not sample data)
        ├── scenarios.jsonl           ← skip (not sample data)
        └── benchmark_report_*.md     ← skip
    
    Priority: loads synthetic_data_*.jsonl first (the complete output).
    Falls back to intermediate_*.jsonl only if no synthetic_data files exist.
    """
    data_path = Path(data_dir)
    samples = []
    
    # --- Strategy 1: Look for final synthetic_data JSONL files ---
    synthetic_files = sorted(data_path.glob("**/synthetic_data_*.jsonl"))
    
    if synthetic_files:
        for sf in synthetic_files:
            count = _load_jsonl_samples(sf, samples)
            logger.debug(f"    Loaded {count} samples from {sf.name}")
        
        if samples:
            return samples
    
    # --- Strategy 2: Fall back to intermediate checkpoint files ---
    intermediate_files = sorted(data_path.glob("**/intermediate_*.jsonl"))
    
    if intermediate_files:
        latest = intermediate_files[-1]
        count = _load_jsonl_samples(latest, samples)
        logger.debug(f"    Loaded {count} samples from latest intermediate: {latest.name}")
        
        if samples:
            return samples
    
    # --- Strategy 3: Try any .jsonl file that contains sample data ---
    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    for jf in jsonl_files:
        if any(skip in jf.name for skip in ["scenario", "benchmark"]):
            continue
        _load_jsonl_samples(jf, samples)
    
    if samples:
        return samples
    
    # --- Strategy 4: Try .json files (legacy format support) ---
    json_files = sorted(data_path.glob("**/*.json"))
    for jf in json_files:
        if any(skip in jf.name for skip in ["benchmark", "summary", "report"]):
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                valid = [d for d in data if isinstance(d, dict) and "dialogue" in d]
                samples.extend(valid)
            elif isinstance(data, dict):
                if "samples" in data:
                    valid = [d for d in data["samples"] if "dialogue" in d]
                    samples.extend(valid)
                elif "dialogue" in data and "summary" in data:
                    samples.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")
    
    return samples


def _load_jsonl_samples(filepath: Path, samples_list: List[Dict]) -> int:
    """
    Load samples from a JSONL file, filtering to only valid synthetic samples.
    
    Returns:
        Number of valid samples loaded.
    """
    count = 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "dialogue" in data and "summary" in data:
                        samples_list.append(data)
                        count += 1
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num} in {filepath.name}")
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
    return count


def _extract_dialogue_text(sample: Dict) -> str:
    """Convert dialogue turns to plain text."""
    dialogue = sample.get("dialogue", [])
    lines = []
    for turn in dialogue:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _extract_summary_text(sample: Dict) -> str:
    """Convert clinical summary dict to structured text output."""
    summary = sample.get("summary", {})
    
    sections = []
    
    field_map = [
        ("chief_complaint", "Chief Complaint"),
        ("history_of_present_illness", "History of Present Illness"),
        ("past_medical_history", "Past Medical History"),
        ("medications", "Medications"),
        ("allergies", "Allergies"),
        ("social_history", "Social History"),
        ("family_history", "Family History"),
        ("physical_examination", "Examination Findings"),
        ("assessment", "Assessment"),
        ("plan", "Plan"),
        ("safety_netting", "Safety Netting"),
    ]
    
    for field_key, header in field_map:
        value = summary.get(field_key)
        if value and str(value).strip() and value != "None":
            sections.append(f"**{header}:** {value}")
    
    # Include SOAP if available
    soap = summary.get("soap")
    if soap and isinstance(soap, dict):
        sections.append("")
        sections.append("**SOAP Note:**")
        if soap.get("subjective") or soap.get("S"):
            sections.append(f"S: {soap.get('subjective') or soap.get('S')}")
        if soap.get("objective") or soap.get("O"):
            sections.append(f"O: {soap.get('objective') or soap.get('O')}")
        if soap.get("assessment") or soap.get("A"):
            sections.append(f"A: {soap.get('assessment') or soap.get('A')}")
        if soap.get("plan") or soap.get("P"):
            sections.append(f"P: {soap.get('plan') or soap.get('P')}")
    
    return "\n".join(sections)


def _extract_rag_context(sample: Dict) -> Optional[str]:
    """Extract RAG context from sample metadata."""
    rag = sample.get("rag", {})
    if rag.get("rag_enabled") and rag.get("context_used"):
        return rag["context_used"]
    return None


def _count_doctor_questions(sample: Dict) -> int:
    """Count the number of questions asked by the doctor."""
    dialogue = sample.get("dialogue", [])
    count = 0
    for turn in dialogue:
        if turn.get("speaker") == "Doctor" and "?" in turn.get("text", ""):
            count += 1
    return count


def _has_both_speakers(sample: Dict) -> bool:
    """Check that both Doctor and Patient are present."""
    speakers = {turn.get("speaker") for turn in sample.get("dialogue", [])}
    return "Doctor" in speakers and "Patient" in speakers


def _is_soap_complete(sample: Dict) -> bool:
    """Check that summary has all essential fields."""
    summary = sample.get("summary", {})
    required = ["chief_complaint", "history_of_present_illness", "assessment", "plan"]
    for field_key in required:
        val = summary.get(field_key)
        if not val or len(str(val).strip()) < 5:
            return False
    return True


def _sample_id(sample: Dict) -> str:
    """Get or generate a unique ID for the sample."""
    if "id" in sample:
        return sample["id"]
    content = json.dumps(sample.get("dialogue", []), sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# Main Data Preparator
# =============================================================================

class TrainingDataPreparator:
    """
    Prepares synthetic clinical data for SLM fine-tuning.
    
    Pipeline:
        1. Load raw synthetic samples from teacher pipeline output
        2. Apply quality filters (validation, clinical, structural)
        3. Format as model-specific chat template instruction-tuning examples
        4. Create stratified train/val/test splits
        5. Export as HuggingFace-compatible dataset
    
    Example (Phi-3.5):
        config = DataPrepConfig(
            raw_data_dirs=["./data/batch_1"],
            base_model="unsloth/Phi-3.5-mini-instruct",
        )
        prep = TrainingDataPreparator(config)
        stats = prep.run()
    
    Example (Qwen2.5):
        config = DataPrepConfig(
            raw_data_dirs=["./data/batch_1"],
            base_model="unsloth/Qwen2.5-3B-Instruct",
            output_dir="./data/training_data_qwen",
        )
        prep = TrainingDataPreparator(config)
        stats = prep.run()
    """
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.raw_samples: List[Dict] = []
        self.filtered_samples: List[Dict] = []
        self.formatted_examples: List[Dict] = []
        self.splits: Dict[str, List[Dict]] = {}
        
        # Resolve template
        if config.model_type == "auto":
            self.template_key = detect_template(config.base_model)
        else:
            self.template_key = config.model_type
        
        if self.template_key not in CHAT_TEMPLATES:
            raise ValueError(
                f"Unknown model_type '{self.template_key}'. "
                f"Supported: {list(CHAT_TEMPLATES.keys())}"
            )
        
        logger.info(f"Using chat template: {self.template_key} (model: {config.base_model})")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the full data preparation pipeline.
        
        Returns:
            Dictionary with statistics about the preparation process.
        """
        logger.info("=" * 60)
        logger.info("Starting Data Preparation Pipeline")
        logger.info(f"  Model: {self.config.base_model}")
        logger.info(f"  Template: {self.template_key}")
        logger.info("=" * 60)
        
        # Step 1: Load
        self.raw_samples = self.load_raw_data()
        logger.info(f"Loaded {len(self.raw_samples)} raw samples")
        
        # Step 2: Filter
        self.filtered_samples = self.filter_samples(self.raw_samples)
        logger.info(f"After filtering: {len(self.filtered_samples)} samples")
        
        # Step 3: Format
        self.formatted_examples = self.format_all(self.filtered_samples)
        logger.info(f"Formatted {len(self.formatted_examples)} training examples")
        
        # Step 4: Split
        self.splits = self.create_splits(self.formatted_examples)
        for split_name, split_data in self.splits.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")
        
        # Step 5: Export
        output_dir = Path(self.config.output_dir)
        self.export_splits(self.splits, output_dir)
        
        # Step 6: Statistics
        stats = self.compute_statistics()
        self._save_statistics(stats, output_dir)
        
        logger.info("Data preparation complete!")
        return stats
    
    # -------------------------------------------------------------------------
    # Step 1: Load
    # -------------------------------------------------------------------------
    
    def load_raw_data(self) -> List[Dict]:
        """Load synthetic samples from all configured directories."""
        all_samples = []
        for data_dir in self.config.raw_data_dirs:
            data_path = Path(data_dir)
            
            if not data_path.exists():
                logger.warning(f"  Directory not found: {data_dir}")
                continue
            
            batch_dirs = sorted(data_path.glob("batch_*"))
            
            if batch_dirs:
                logger.info(f"  Found {len(batch_dirs)} batch directories in {data_dir}")
                for batch_dir in batch_dirs:
                    dir_samples = _load_samples_from_dir(str(batch_dir))
                    logger.info(f"    {batch_dir.name}: {len(dir_samples)} samples")
                    all_samples.extend(dir_samples)
            else:
                dir_samples = _load_samples_from_dir(data_dir)
                logger.info(f"  Loaded {len(dir_samples)} samples from {data_dir}")
                all_samples.extend(dir_samples)
        
        # Deduplicate by ID
        seen_ids = set()
        unique = []
        for s in all_samples:
            sid = _sample_id(s)
            if sid not in seen_ids:
                seen_ids.add(sid)
                unique.append(s)
        
        if len(unique) < len(all_samples):
            logger.info(f"  Removed {len(all_samples) - len(unique)} duplicates")
        
        return unique
    
    # -------------------------------------------------------------------------
    # Step 2: Filter
    # -------------------------------------------------------------------------
    
    def filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """Apply multi-stage quality filtering."""
        filter_stats = Counter()
        passed = []
        
        for sample in samples:
            filter_stats["total"] += 1
            
            # Filter 1: Validation status
            if self.config.require_validation_passed:
                validation = sample.get("validation", {})
                status = validation.get("status", "passed")
                if status == "failed":
                    filter_stats["failed_validation"] += 1
                    continue
            
            # Filter 2: Clinical validity
            if self.config.require_clinical_valid:
                validation = sample.get("validation", {})
                if validation.get("clinical_valid") is False:
                    filter_stats["failed_clinical"] += 1
                    continue
            
            # Filter 3: Hallucination check
            validation = sample.get("validation", {})
            errors = validation.get("errors", [])
            has_hallucination = any(
                "hallucination" in e.get("error_type", "").lower()
                for e in errors
            )
            if has_hallucination:
                halluc_errors = [
                    e for e in errors 
                    if "hallucination" in e.get("error_type", "").lower()
                ]
                severe = any(
                    e.get("severity", "") in ("major", "critical")
                    for e in halluc_errors
                )
                if severe:
                    filter_stats["failed_hallucination"] += 1
                    continue
            
            # Filter 4: Dialogue quality
            dialogue = sample.get("dialogue", [])
            if len(dialogue) < self.config.min_dialogue_turns:
                filter_stats["failed_min_turns"] += 1
                continue
            
            if self.config.require_both_speakers and not _has_both_speakers(sample):
                filter_stats["failed_speakers"] += 1
                continue
            
            if _count_doctor_questions(sample) < self.config.min_doctor_questions:
                filter_stats["failed_doctor_questions"] += 1
                continue
            
            # Filter 5: Summary completeness
            if self.config.require_soap_complete and not _is_soap_complete(sample):
                filter_stats["failed_soap_complete"] += 1
                continue
            
            filter_stats["passed"] += 1
            passed.append(sample)
        
        logger.info("  Filter results:")
        for key, count in sorted(filter_stats.items()):
            logger.info(f"    {key}: {count}")
        
        return passed
    
    # -------------------------------------------------------------------------
    # Step 3: Format (multi-model support)
    # -------------------------------------------------------------------------
    
    def format_all(self, samples: List[Dict]) -> List[Dict]:
        """
        Format all samples as chat template instruction-tuning examples.
        
        Uses config.rag_context_ratio to decide which samples include 
        RAG context in the user message. Uses self.template_key to 
        select the correct chat template.
        """
        random.seed(self.config.seed)
        
        # Determine which samples get RAG context
        indices_with_rag_context = set()
        samples_with_rag = [
            i for i, s in enumerate(samples) 
            if _extract_rag_context(s) is not None
        ]
        
        if samples_with_rag:
            n_with_rag = int(len(samples) * self.config.rag_context_ratio)
            n_with_rag = min(n_with_rag, len(samples_with_rag))
            indices_with_rag_context = set(
                random.sample(samples_with_rag, n_with_rag)
            )
        
        formatted = []
        for i, sample in enumerate(samples):
            include_rag = i in indices_with_rag_context
            example = self.format_single(sample, include_rag_context=include_rag)
            if example:
                formatted.append(example)
        
        return formatted
    
    def format_single(
        self, 
        sample: Dict, 
        include_rag_context: bool = False
    ) -> Optional[Dict]:
        """
        Format a single sample using the selected chat template.
        
        Args:
            sample: Raw synthetic sample dict.
            include_rag_context: Whether to include RAG context in the user message.
        
        Returns:
            Dictionary with 'text' (full chat template string) and metadata,
            or None if the sample cannot be formatted.
        """
        dialogue_text = _extract_dialogue_text(sample)
        summary_text = _extract_summary_text(sample)
        
        if not dialogue_text or not summary_text:
            return None
        
        # Build user message
        user_parts = []
        
        if include_rag_context:
            rag_context = _extract_rag_context(sample)
            if rag_context:
                user_parts.append(
                    f"Relevant clinical guidelines:\n{rag_context}\n"
                )
        
        user_parts.append(
            f"Summarise the following clinical consultation:\n\n{dialogue_text}"
        )
        user_parts.append(f"\n\n{SUMMARY_INSTRUCTION}")
        
        user_message = "\n".join(user_parts)
        
        # Build formatted text using the model-specific template
        chatml_text = format_chat_template(
            system_msg=SYSTEM_PROMPT,
            user_msg=user_message,
            assistant_msg=summary_text,
            template_key=self.template_key,
        )
        
        # Extract metadata
        scenario = sample.get("scenario", {})
        difficulty = sample.get("difficulty", {})
        
        return {
            "id": _sample_id(sample),
            "text": chatml_text,
            "specialty": scenario.get("specialty", "General Practice"),
            "difficulty": difficulty.get("difficulty_score", 5) if difficulty else 5,
            "difficulty_level": difficulty.get("difficulty_level", "medium") if difficulty else "medium",
            "has_rag_context": include_rag_context,
            "num_turns": len(sample.get("dialogue", [])),
            "urgency": scenario.get("urgency", "routine"),
            "model_template": self.template_key,
        }
    
    # -------------------------------------------------------------------------
    # Step 4: Split
    # -------------------------------------------------------------------------
    
    def create_splits(
        self, 
        examples: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Create stratified train/val/test splits."""
        random.seed(self.config.seed)
        
        by_specialty: Dict[str, List[Dict]] = {}
        for ex in examples:
            spec = ex.get("specialty", "General Practice")
            by_specialty.setdefault(spec, []).append(ex)
        
        train, val, test = [], [], []
        
        for spec, spec_examples in by_specialty.items():
            random.shuffle(spec_examples)
            n = len(spec_examples)
            n_val = max(1, int(n * self.config.val_ratio))
            n_test = max(1, int(n * self.config.test_ratio))
            n_train = n - n_val - n_test
            
            if n_train < 1:
                train.extend(spec_examples)
                continue
            
            train.extend(spec_examples[:n_train])
            val.extend(spec_examples[n_train:n_train + n_val])
            test.extend(spec_examples[n_train + n_val:])
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return {"train": train, "val": val, "test": test}
    
    # -------------------------------------------------------------------------
    # Step 5: Export
    # -------------------------------------------------------------------------
    
    def export_splits(
        self, 
        splits: Dict[str, List[Dict]], 
        output_dir: Path
    ):
        """Export splits as JSONL files (HuggingFace compatible)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            filepath = output_dir / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"  Exported {len(split_data)} examples to {filepath}")
        
        # Export metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "config": {
                "base_model": self.config.base_model,
                "model_template": self.template_key,
                "rag_context_ratio": self.config.rag_context_ratio,
                "max_seq_length": self.config.max_seq_length,
                "seed": self.config.seed,
            },
            "splits": {name: len(data) for name, data in splits.items()},
            "total": sum(len(data) for data in splits.values()),
        }
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def export_to_huggingface_dataset(self):
        """Export splits as a HuggingFace Dataset object."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        hf_splits = {}
        for split_name, split_data in self.splits.items():
            hf_splits[split_name] = Dataset.from_list(split_data)
        
        return DatasetDict(hf_splits)
    
    # -------------------------------------------------------------------------
    # Step 6: Statistics
    # -------------------------------------------------------------------------
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        stats = {
            "raw_count": len(self.raw_samples),
            "filtered_count": len(self.filtered_samples),
            "formatted_count": len(self.formatted_examples),
            "filter_rate": (
                1 - len(self.filtered_samples) / len(self.raw_samples)
                if self.raw_samples else 0
            ),
            "model_template": self.template_key,
            "base_model": self.config.base_model,
        }
        
        for split_name, split_data in self.splits.items():
            stats[f"{split_name}_count"] = len(split_data)
        
        specialty_counts = Counter(
            ex.get("specialty", "Unknown") 
            for ex in self.formatted_examples
        )
        stats["specialty_distribution"] = dict(specialty_counts)
        
        difficulty_counts = Counter(
            ex.get("difficulty_level", "unknown") 
            for ex in self.formatted_examples
        )
        stats["difficulty_distribution"] = dict(difficulty_counts)
        
        with_rag = sum(1 for ex in self.formatted_examples if ex.get("has_rag_context"))
        stats["actual_rag_ratio"] = with_rag / len(self.formatted_examples) if self.formatted_examples else 0
        
        lengths = [len(ex["text"]) / 4 for ex in self.formatted_examples]
        if lengths:
            stats["token_stats"] = {
                "mean": sum(lengths) / len(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "over_max_seq": sum(1 for l in lengths if l > self.config.max_seq_length),
            }
        
        return stats
    
    def _save_statistics(self, stats: Dict, output_dir: Path):
        """Save statistics to a JSON file."""
        with open(output_dir / "preparation_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("\n" + "=" * 60)
        logger.info("Dataset Statistics")
        logger.info("=" * 60)
        logger.info(f"  Model template:       {stats['model_template']}")
        logger.info(f"  Raw samples loaded:   {stats['raw_count']}")
        logger.info(f"  After filtering:      {stats['filtered_count']}")
        logger.info(f"  Filter rejection rate: {stats['filter_rate']:.1%}")
        logger.info(f"  Training examples:    {stats.get('train_count', 0)}")
        logger.info(f"  Validation examples:  {stats.get('val_count', 0)}")
        logger.info(f"  Test examples:        {stats.get('test_count', 0)}")
        if "token_stats" in stats:
            ts = stats["token_stats"]
            logger.info(f"  Avg token length:     {ts['mean']:.0f}")
            logger.info(f"  Over max_seq_length:  {ts['over_max_seq']}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Prepare training data for student model")
    parser.add_argument("--data-dirs", nargs="+", required=True, help="Directories with synthetic data")
    parser.add_argument("--output-dir", default="./data/training_data", help="Output directory")
    parser.add_argument("--rag-ratio", type=float, default=0.5, help="Fraction of examples with RAG context")
    parser.add_argument("--min-turns", type=int, default=8, help="Minimum dialogue turns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model", 
        default="unsloth/Phi-3.5-mini-instruct",
        help="Base model name (determines chat template). E.g.: "
             "unsloth/Phi-3.5-mini-instruct, unsloth/Qwen2.5-3B-Instruct"
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "phi3", "qwen2"],
        help="Chat template type (auto-detected from --model if 'auto')"
    )
    
    args = parser.parse_args()
    
    config = DataPrepConfig(
        raw_data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        rag_context_ratio=args.rag_ratio,
        min_dialogue_turns=args.min_turns,
        seed=args.seed,
        base_model=args.model,
        model_type=args.model_type,
    )
    
    prep = TrainingDataPreparator(config)
    stats = prep.run()
    
    print(f"\n✓ Data preparation complete. Output: {args.output_dir}")
    print(f"  Template: {prep.template_key}")
    print(f"  Train: {stats['train_count']}, Val: {stats['val_count']}, Test: {stats['test_count']}")
