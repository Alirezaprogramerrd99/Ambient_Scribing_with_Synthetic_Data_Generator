# Student Model Evaluation Report
**Date:** 2026-04-13T18:29:36.035145
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.633 | N/A | 11.3 | 4.32 | 4.78 | 3.70 |
| Teacher (gpt-4o-mini + RAG) | 0.428 | N/A | 4.5 | 4.79 | 4.98 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.631 | 0.728 | 11.1 | 4.33 |
| dense_rerank | 0.636 | 0.000 | 11.4 | 4.39 |
| dense_rerank_qe | 0.633 | 0.000 | 11.6 | 4.36 |
| full_medical | 0.630 | 0.000 | 10.9 | 4.31 |
