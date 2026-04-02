# Student Model Evaluation Report
**Date:** 2026-03-25T20:53:28.887781
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-1B (no fine-tuning, no RAG) | 0.293 | N/A | 5.9 | 3.72 | 4.04 | 2.82 |
| Base Llama-3.2-1B + RAG (no fine-tuning) | 0.297 | N/A | 5.7 | 3.73 | 4.10 | 2.78 |
| Fine-tuned Llama-3.2-1B (no RAG) | 0.598 | N/A | 6.6 | 4.08 | 4.58 | 3.22 |
| Fine-tuned Llama-3.2-1B + RAG | 0.600 | N/A | 6.7 | 4.10 | 4.56 | 3.12 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.598 | 0.728 | 6.7 | 4.13 |
| dense_rerank | 0.604 | 0.000 | 6.5 | 4.12 |
