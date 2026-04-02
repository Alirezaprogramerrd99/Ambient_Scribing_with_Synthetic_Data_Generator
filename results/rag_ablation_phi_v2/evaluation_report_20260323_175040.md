# Student Model Evaluation Report
**Date:** 2026-03-23T17:22:17.068025
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.512 | N/A | 13.4 | 4.38 | 4.85 | 4.10 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.514 | 0.731 | 13.3 | 4.27 |
| dense_rerank | 0.605 | 0.000 | 13.3 | 4.30 |
| dense_rerank_qe | 0.596 | 0.000 | 13.3 | 4.29 |
| full_medical | 0.613 | 0.000 | 13.4 | 4.33 |
