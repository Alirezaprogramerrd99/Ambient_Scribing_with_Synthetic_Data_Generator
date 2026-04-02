# Student Model Evaluation Report
**Date:** 2026-03-24T14:25:01.881631
**Test Samples:** 20
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.612 | N/A | 10.4 | 4.31 | 4.75 | 3.55 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.608 | 0.731 | 10.7 | 4.26 |
| dense_rerank | 0.617 | 0.000 | 10.5 | 4.42 |
| dense_rerank_qe | 0.614 | 0.000 | 10.7 | 4.25 |
| full_medical | 0.618 | 0.000 | 10.6 | 4.33 |
