# Student Model Evaluation Report
**Date:** 2026-03-25T15:36:26.683969
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.311 | N/A | 9.4 | 4.50 | 5.00 | 5.00 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.329 | N/A | 9.9 | 4.67 | 5.00 | 5.00 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.631 | N/A | 11.0 | 4.27 | 5.00 | 3.50 |
| Fine-tuned Llama-3.2-3B + RAG | 0.605 | N/A | 10.5 | 4.27 | 5.00 | 3.50 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.601 | 0.693 | 10.6 | 3.94 |
| dense_rerank | 0.625 | 0.000 | 10.8 | 4.10 |
