# Student Model Evaluation Report
**Date:** 2026-03-25T17:45:00.240760
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.379 | N/A | 10.0 | 4.74 | 4.98 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.377 | N/A | 9.8 | 4.76 | 4.98 | 5.00 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.632 | N/A | 12.6 | 4.44 | 4.86 | 3.88 |
| Fine-tuned Phi-3.5-mini + RAG | 0.634 | N/A | 13.4 | 4.45 | 4.84 | 4.10 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.534 | 0.728 | 13.7 | 4.36 |
| dense_rerank | 0.634 | 0.000 | 14.2 | 4.45 |
