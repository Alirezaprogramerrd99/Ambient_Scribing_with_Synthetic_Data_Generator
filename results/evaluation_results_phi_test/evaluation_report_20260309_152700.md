# Student Model Evaluation Report
**Date:** 2026-03-09T14:42:50.483504
**Test Samples:** 10
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.383 | N/A | 9.0 | 4.75 | 5.00 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.307 | N/A | 33.7 | 4.08 | 4.60 | 3.60 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.647 | N/A | 13.4 | 4.18 | 4.70 | 3.80 |
| Fine-tuned Phi-3.5-mini + RAG | 0.574 | N/A | 13.2 | 4.39 | 4.80 | 4.00 |
| Teacher (gpt-4o-mini + RAG) | 0.408 | N/A | 5.8 | 4.87 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.574 | 0.722 | 13.7 | 4.44 |
| manual | 0.647 | 0.000 | 13.4 | 4.19 |
| hybrid | 0.574 | 0.722 | 14.5 | 4.41 |
