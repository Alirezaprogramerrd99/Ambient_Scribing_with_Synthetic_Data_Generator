# Literature Comparison Table

## Our Results vs Published Work

| Paper / Model | Params | Task | ROUGE-L | BERTScore-F1 | BLEU-4 | MEDCON-F1 | Notes |
|---|---|---|---|---|---|---|---|
| **Phi-3.5 (3.8B) (Ours)** | **SLM** | **Clinical scribing** | **0.6248** | **0.9391** | **0.5463** | **0.8446** | **QLoRA + RAG** |
| **Llama-3.2 (3B) (Ours)** | **SLM** | **Clinical scribing** | **0.6245** | **0.9390** | **0.5514** | **0.8419** | **QLoRA + RAG** |
| **Llama-3.2 (1B) (Ours)** | **SLM** | **Clinical scribing** | **0.5906** | **0.9349** | **0.5273** | **0.8596** | **QLoRA + RAG** |
| MediGen (Leong et al., 2024) | 8B | Medical report generation | 0.5800 | 0.7200 | — | — | Fine-tuned, dialogue-to-report |
| LLaMA-Clinic (Wang et al., 2024) | 13B | Clinical note generation | — | — | — | — | SFT + DistillDirect RL, expert-rated 4.2/5 |
| Radiology Reports (2024) | 8B | Radiology conclusion generation | 0.4628 | 0.8054 | — | — | LoRA r=16, RTX 3090 |
| Discharge Summary DoRA (2026) | 7B | Discharge summary generation | — | — | — | — | DoRA > LoRA > QLoRA across models |
| MIMIC-IV Benchmark (2025) | 27B | Clinical note summarization | — | — | — | — | Best overall extractive summarization |
| German Discharge (2025) | 8B | Discharge summary (German) | 0.2500 | 0.6400 | — | — | Prompt engineering only, no fine-tuning |
| ACI-Bench (Yim et al., 2023) | Various | Visit note generation | — | — | — | — | Benchmark dataset, uses ROUGE+BERTScore+MEDCON |

**Notes:**
- Our models are 1B-3.8B parameters, significantly smaller than most literature (7B-27B)
- ROUGE-L and BERTScore are not directly comparable across different datasets and tasks
- Our evaluation uses synthetic dialogue-to-summary data; papers vary in data sources
- MEDCON (regex) is an approximation; QuickUMLS provides more accurate medical concept extraction