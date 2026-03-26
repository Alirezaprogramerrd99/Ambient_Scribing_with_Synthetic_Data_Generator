# Student Model Evaluation Report
**Date:** 2026-03-23T19:15:08.828585
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.598 | N/A | 11.1 | 4.23 | 4.55 | 3.55 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.604 | 0.731 | 10.6 | 4.32 |
| dense_rerank | 0.612 | 0.000 | 10.4 | 4.29 |
| dense_rerank_qe | 0.619 | 0.000 | 10.6 | 4.39 |
| full_medical | 0.621 | 0.000 | 10.7 | 4.21 |
