# Student Model Evaluation Report
**Date:** 2026-03-25T16:18:26.277902
**Test Samples:** 1
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-1B (no fine-tuning, no RAG) | 0.336 | N/A | 5.8 | 4.50 | 5.00 | 5.00 |
| Base Llama-3.2-1B + RAG (no fine-tuning) | 0.292 | N/A | 6.6 | 4.00 | 4.00 | 3.00 |
| Fine-tuned Llama-3.2-1B (no RAG) | 0.536 | N/A | 7.3 | 4.33 | 5.00 | 4.00 |
| Fine-tuned Llama-3.2-1B + RAG | 0.523 | N/A | 6.8 | 4.50 | 5.00 | 4.00 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.574 | 0.721 | 6.6 | 4.20 |
| dense_rerank | 0.524 | 0.000 | 6.9 | 4.33 |
