# Student Model Evaluation Report
**Date:** 2026-03-11T17:28:24.373737
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.383 | N/A | 10.1 | 4.75 | 4.96 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.328 | N/A | 29.6 | 4.30 | 4.74 | 4.02 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.640 | N/A | 13.0 | 4.40 | 4.80 | 4.02 |
| Fine-tuned Phi-3.5-mini + RAG | 0.596 | N/A | 14.8 | 4.35 | 4.82 | 3.86 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 5.6 | 4.88 | 5.00 | 4.98 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.596 | 0.729 | 15.0 | 4.33 |
| manual | 0.640 | 0.000 | 14.3 | 4.38 |
| hybrid | 0.596 | 0.729 | 15.4 | 4.36 |
