# Student Model Evaluation Report
**Date:** 2026-03-25T16:11:10.035019
**Test Samples:** 1
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.400 | N/A | 9.7 | 4.67 | 5.00 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.384 | N/A | 8.5 | 4.67 | 5.00 | 5.00 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.572 | N/A | 14.8 | 4.33 | 5.00 | 4.00 |
| Fine-tuned Phi-3.5-mini + RAG | 0.590 | N/A | 14.1 | 4.33 | 5.00 | 4.00 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.505 | 0.721 | 13.5 | 4.33 |
| dense_rerank | 0.617 | 0.000 | 13.4 | 4.20 |
