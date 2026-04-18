# Student Model Evaluation Report
**Date:** 2026-03-09T13:28:27.511802
**Test Samples:** 5
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini (no RAG) | 0.660 | N/A | 13.5 | 4.10 | 4.60 | 3.40 |
| Fine-tuned Phi-3.5-mini + RAG | 0.660 | N/A | 13.5 | 4.14 | 4.80 | 3.40 |
| Teacher (gpt-4o-mini + RAG) | 0.415 | N/A | 5.3 | 4.77 | 5.00 | 5.00 |

**Skipped configurations:**
- baseline: Unsloth: No config file found - are you sure the `model_name` is correct?
If you're using a model on your local device, confirm if the folder location exists.
If you're using a HuggingFace online model, check if it exists.
- rag_only: Unsloth: No config file found - are you sure the `model_name` is correct?
If you're using a model on your local device, confirm if the folder location exists.
If you're using a HuggingFace online model, check if it exists.

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.660 | 0.000 | 13.5 | 4.21 |
| manual | 0.660 | 0.000 | 13.3 | 4.27 |
| hybrid | 0.660 | 0.000 | 14.0 | 4.16 |
