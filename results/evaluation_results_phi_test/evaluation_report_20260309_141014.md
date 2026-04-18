# Student Model Evaluation Report
**Date:** 2026-03-09T13:56:44.508835
**Test Samples:** 5
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.371 | N/A | 10.1 | 4.77 | 5.00 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.371 | N/A | 9.9 | 4.73 | 5.00 | 5.00 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.660 | N/A | 14.5 | 4.14 | 4.60 | 3.40 |
| Fine-tuned Phi-3.5-mini + RAG | 0.660 | N/A | 15.2 | 4.10 | 4.80 | 3.40 |
| Teacher (gpt-4o-mini + RAG) | 0.415 | N/A | 9.1 | 4.87 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.660 | 0.000 | 15.0 | 4.03 |
| manual | 0.660 | 0.000 | 15.4 | 4.16 |
| hybrid | 0.660 | 0.000 | 14.1 | 4.16 |
