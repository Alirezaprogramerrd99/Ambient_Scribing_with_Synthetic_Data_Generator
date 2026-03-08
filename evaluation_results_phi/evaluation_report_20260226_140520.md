# Student Model Evaluation Report
**Date:** 2026-02-25T15:22:26.663866
**Test Samples:** 50
**Student Model:** phi35-clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.381 | 9.7 | 4.72 | 4.98 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.340 | 20.2 | 4.22 | 4.72 | 3.86 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.448 | 25.3 | 3.55 | 3.88 | 2.26 |
| Fine-tuned Phi-3.5-mini + RAG | 0.451 | 14.7 | 3.59 | 4.08 | 2.24 |
| Teacher (gpt-4o-mini + RAG) | 0.427 | 5.8 | 4.90 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.451 | 0.729 | 14.8 | 3.62 |
| manual | 0.448 | 0.000 | 25.1 | 3.57 |
| hybrid | 0.451 | 0.729 | 14.6 | 3.61 |
