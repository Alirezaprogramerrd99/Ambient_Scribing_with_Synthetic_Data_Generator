# Student Model Evaluation Report
**Date:** 2026-03-23T18:38:18.699481
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.520 | N/A | 13.3 | 4.24 | 4.70 | 3.85 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.511 | 0.731 | 13.2 | 4.20 |
| dense_rerank | 0.631 | 0.000 | 13.9 | 4.32 |
| dense_rerank_qe | 0.601 | 0.000 | 13.1 | 4.37 |
| full_medical | 0.625 | 0.000 | 13.8 | 4.37 |
