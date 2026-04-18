# Student Model Evaluation Report
**Date:** 2026-03-23T19:44:54.834544
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.594 | N/A | 10.9 | 4.22 | 4.55 | 3.45 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.603 | 0.731 | 11.1 | 4.24 |
| dense_rerank | 0.615 | 0.000 | 10.8 | 4.26 |
| dense_rerank_qe | 0.617 | 0.000 | 11.1 | 4.42 |
| full_medical | 0.615 | 0.000 | 11.2 | 4.47 |
