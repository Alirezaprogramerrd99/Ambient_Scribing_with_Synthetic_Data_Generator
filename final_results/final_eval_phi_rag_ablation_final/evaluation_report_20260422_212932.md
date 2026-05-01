# Student Model Evaluation Report
**Date:** 2026-04-22T20:08:08.616617
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.635 | N/A | 14.2 | 4.46 | 4.84 | 4.00 |
| Teacher (gpt-4o-mini + RAG) | 0.423 | N/A | 5.2 | 4.76 | 4.98 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.536 | 0.728 | 14.1 | 4.33 |
| dense_rerank | 0.627 | 0.000 | 14.3 | 4.36 |
| dense_rerank_qe | 0.632 | 0.000 | 14.5 | 4.43 |
| full_medical | 0.632 | 0.000 | 14.5 | 4.33 |
