# Student Model Evaluation Report
**Date:** 2026-04-13T16:38:43.758363
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.623 | N/A | 14.5 | 4.58 | 5.00 | 4.50 |
| Teacher (gpt-4o-mini + RAG) | 0.408 | N/A | 4.8 | 4.83 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.545 | 0.693 | 14.8 | 3.92 |
| dense_rerank | 0.599 | 0.000 | 11.6 | 4.35 |
| dense_rerank_qe | 0.641 | 0.000 | 12.7 | 4.27 |
| full_medical | 0.618 | 0.000 | 14.5 | 4.44 |
