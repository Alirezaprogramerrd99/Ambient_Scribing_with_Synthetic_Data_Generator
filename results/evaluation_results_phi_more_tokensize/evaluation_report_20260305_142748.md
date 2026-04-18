# Student Model Evaluation Report
**Date:** 2026-03-05T11:11:38.585085
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.383 | N/A | 9.9 | 4.72 | 4.98 | 4.98 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.328 | N/A | 29.1 | 4.32 | 4.76 | 4.10 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.026 | N/A | 45.7 | 1.07 | 1.40 | 1.08 |
| Fine-tuned Phi-3.5-mini + RAG | 0.028 | N/A | 16.5 | 2.70 | 3.26 | 3.14 |
| Teacher (gpt-4o-mini + RAG) | 0.427 | N/A | 6.4 | 4.88 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.028 | 0.729 | 16.7 | 2.76 |
| manual | 0.026 | 0.000 | 46.6 | 1.07 |
| hybrid | 0.028 | 0.729 | 16.5 | 2.72 |
