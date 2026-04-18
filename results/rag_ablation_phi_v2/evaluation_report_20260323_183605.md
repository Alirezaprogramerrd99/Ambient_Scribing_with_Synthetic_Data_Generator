# Student Model Evaluation Report
**Date:** 2026-03-23T18:01:15.721805
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini (no RAG) | 0.611 | N/A | 12.7 | 4.32 | 4.75 | 3.80 |
| Fine-tuned Phi-3.5-mini + RAG | 0.507 | N/A | 13.2 | 4.27 | 4.80 | 3.75 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.499 | 0.731 | 12.5 | 4.20 |
| dense_rerank | 0.634 | 0.000 | 13.8 | 4.41 |
| dense_rerank_qe | 0.604 | 0.000 | 13.7 | 4.38 |
| full_medical | 0.620 | 0.000 | 14.0 | 4.19 |
