# Student Model Evaluation Report
**Date:** 2026-03-25T15:42:47.851646
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-1B (no fine-tuning, no RAG) | 0.316 | N/A | 5.6 | 4.00 | 4.50 | 3.50 |
| Base Llama-3.2-1B + RAG (no fine-tuning) | 0.286 | N/A | 5.5 | 3.83 | 3.50 | 3.00 |
| Fine-tuned Llama-3.2-1B (no RAG) | 0.565 | N/A | 6.7 | 3.94 | 4.00 | 2.50 |
| Fine-tuned Llama-3.2-1B + RAG | 0.563 | N/A | 6.4 | 3.85 | 4.50 | 2.50 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.564 | 0.693 | 6.4 | 3.85 |
| dense_rerank | 0.549 | 0.000 | 6.1 | 3.94 |
