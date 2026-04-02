# Student Model Evaluation Report
**Date:** 2026-03-24T15:18:57.275792
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-1B + RAG | 0.583 | N/A | 7.0 | 3.91 | 4.25 | 2.80 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.589 | 0.731 | 6.6 | 4.01 |
| dense_rerank | 0.583 | 0.000 | 7.1 | 4.00 |
| dense_rerank_qe | 0.594 | 0.000 | 6.1 | 4.01 |
| full_medical | 0.592 | 0.000 | 6.3 | 4.00 |
