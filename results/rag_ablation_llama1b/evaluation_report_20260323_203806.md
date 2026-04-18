# Student Model Evaluation Report
**Date:** 2026-03-23T20:13:15.417958
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-1B + RAG | 0.573 | N/A | 6.6 | 4.06 | 4.35 | 3.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.586 | 0.731 | 5.9 | 4.13 |
| dense_rerank | 0.595 | 0.000 | 6.2 | 4.07 |
| dense_rerank_qe | 0.592 | 0.000 | 6.5 | 4.07 |
| full_medical | 0.587 | 0.000 | 6.4 | 4.00 |
