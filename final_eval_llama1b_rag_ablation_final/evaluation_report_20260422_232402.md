# Student Model Evaluation Report
**Date:** 2026-04-22T22:38:26.680005
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-1B + RAG | 0.602 | N/A | 6.4 | 4.08 | 4.58 | 3.10 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 4.1 | 4.75 | 4.98 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.599 | 0.728 | 6.5 | 4.15 |
| dense_rerank | 0.596 | 0.000 | 6.4 | 4.11 |
| dense_rerank_qe | 0.600 | 0.000 | 6.2 | 4.09 |
| full_medical | 0.595 | 0.000 | 6.3 | 4.18 |
