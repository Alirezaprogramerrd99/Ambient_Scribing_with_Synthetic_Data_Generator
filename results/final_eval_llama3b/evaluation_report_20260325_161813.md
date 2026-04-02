# Student Model Evaluation Report
**Date:** 2026-03-25T16:15:32.839561
**Test Samples:** 1
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.319 | N/A | 9.9 | 4.67 | 5.00 | 5.00 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.315 | N/A | 10.2 | 4.67 | 5.00 | 5.00 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.579 | N/A | 14.5 | 4.20 | 5.00 | 3.00 |
| Fine-tuned Llama-3.2-3B + RAG | 0.578 | N/A | 11.0 | 4.33 | 5.00 | 4.00 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.551 | 0.721 | 11.1 | 4.20 |
| dense_rerank | 0.586 | 0.000 | 10.4 | 4.33 |
