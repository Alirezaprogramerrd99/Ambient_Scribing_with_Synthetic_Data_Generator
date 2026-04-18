# Student Model Evaluation Report
**Date:** 2026-04-13T16:57:40.902488
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.633 | N/A | 13.7 | 4.48 | 4.88 | 4.16 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 5.3 | 4.75 | 4.96 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.531 | 0.728 | 13.7 | 4.33 |
| dense_rerank | 0.636 | 0.000 | 14.3 | 4.40 |
| dense_rerank_qe | 0.629 | 0.000 | 14.3 | 4.42 |
| full_medical | 0.631 | 0.000 | 14.4 | 4.42 |
