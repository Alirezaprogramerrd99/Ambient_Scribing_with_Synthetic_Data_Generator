# Student Model Evaluation Report
**Date:** 2026-04-13T16:27:17.090396
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.599 | N/A | 12.7 | 4.35 | 5.00 | 3.50 |
| Teacher (gpt-4o-mini + RAG) | 0.411 | N/A | 4.2 | 4.67 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.554 | 0.693 | 16.0 | 3.92 |
| dense_rerank | 0.610 | 0.000 | 13.9 | 4.25 |
| dense_rerank_qe | 0.657 | 0.000 | 13.1 | 4.20 |
| full_medical | 0.598 | 0.000 | 13.4 | 4.27 |
