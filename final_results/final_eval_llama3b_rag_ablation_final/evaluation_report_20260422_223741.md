# Student Model Evaluation Report
**Date:** 2026-04-22T21:30:18.716491
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.634 | N/A | 11.3 | 4.30 | 4.80 | 3.64 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 4.9 | 4.76 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.626 | 0.728 | 11.4 | 4.35 |
| dense_rerank | 0.632 | 0.000 | 11.1 | 4.33 |
| dense_rerank_qe | 0.636 | 0.000 | 11.6 | 4.35 |
| full_medical | 0.632 | 0.000 | 11.6 | 4.31 |
