# Student Model Evaluation Report
**Date:** 2026-04-14T10:28:35.986038
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-1B + RAG | 0.600 | N/A | 6.2 | 4.13 | 4.56 | 3.26 |
| Teacher (gpt-4o-mini + RAG) | 0.427 | N/A | 4.4 | 4.74 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.599 | 0.728 | 6.1 | 4.15 |
| dense_rerank | 0.600 | 0.000 | 6.2 | 4.10 |
| dense_rerank_qe | 0.602 | 0.000 | 6.2 | 4.13 |
| full_medical | 0.599 | 0.000 | 6.2 | 4.15 |
