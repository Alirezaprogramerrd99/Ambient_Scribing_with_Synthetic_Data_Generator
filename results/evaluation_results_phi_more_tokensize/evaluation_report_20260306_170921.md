# Student Model Evaluation Report
**Date:** 2026-03-05T16:04:15.881734
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.383 | N/A | 9.6 | 4.73 | 4.98 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.328 | N/A | 28.8 | 4.30 | 4.78 | 4.06 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.640 | N/A | 13.4 | 4.38 | 4.74 | 3.94 |
| Fine-tuned Phi-3.5-mini + RAG | 0.596 | N/A | 14.3 | 4.36 | 4.82 | 3.88 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 6.6 | 4.87 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.596 | 0.729 | 14.5 | 4.33 |
| manual | 0.640 | 0.000 | 14.1 | 4.39 |
| hybrid | 0.596 | 0.729 | 14.7 | 4.35 |
