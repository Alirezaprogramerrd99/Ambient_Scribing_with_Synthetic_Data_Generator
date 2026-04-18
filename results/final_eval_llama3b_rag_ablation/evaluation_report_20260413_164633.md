# Student Model Evaluation Report
**Date:** 2026-04-13T16:42:42.180078
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Llama-3.2-3B + RAG | 0.602 | N/A | 12.3 | 3.83 | 4.00 | 2.50 |
| Teacher (gpt-4o-mini + RAG) | 0.410 | N/A | 5.1 | 4.67 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.613 | 0.693 | 11.3 | 4.27 |
| dense_rerank | 0.626 | 0.000 | 10.9 | 4.27 |
| dense_rerank_qe | 0.611 | 0.000 | 11.2 | 3.83 |
| full_medical | 0.603 | 0.000 | 11.5 | 4.27 |
