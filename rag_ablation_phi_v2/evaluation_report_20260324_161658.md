# Student Model Evaluation Report
**Date:** 2026-03-24T15:46:37.998398
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.527 | N/A | 12.3 | 4.34 | 4.75 | 3.75 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.517 | 0.731 | 12.1 | 4.36 |
| dense_rerank | 0.627 | 0.000 | 12.7 | 4.23 |
| dense_rerank_qe | 0.616 | 0.000 | 12.3 | 4.32 |
| full_medical | 0.633 | 0.000 | 12.8 | 4.38 |
