# Student Model Evaluation Report
**Date:** 2026-04-13T16:46:50.694095
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-1B + RAG | 0.544 | N/A | 6.7 | 3.83 | 4.00 | 3.00 |
| Teacher (gpt-4o-mini + RAG) | 0.418 | N/A | 4.4 | 4.92 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.597 | 0.693 | 6.6 | 3.94 |
| dense_rerank | 0.552 | 0.000 | 8.2 | 3.75 |
| dense_rerank_qe | 0.569 | 0.000 | 6.1 | 3.94 |
| full_medical | 0.549 | 0.000 | 7.2 | 4.00 |
