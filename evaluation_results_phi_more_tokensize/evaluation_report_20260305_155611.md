# Student Model Evaluation Report
**Date:** 2026-03-05T15:43:30.030197
**Test Samples:** 5
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.371 | N/A | 8.6 | 4.77 | 5.00 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.341 | N/A | 20.8 | 4.33 | 4.80 | 4.40 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.660 | N/A | 14.4 | 4.21 | 4.80 | 3.40 |
| Fine-tuned Phi-3.5-mini + RAG | 0.611 | N/A | 14.2 | 4.34 | 4.80 | 3.80 |
| Teacher (gpt-4o-mini + RAG) | 0.407 | N/A | 7.3 | 4.83 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.611 | 0.709 | 13.8 | 4.27 |
| manual | 0.660 | 0.000 | 13.1 | 4.23 |
| hybrid | 0.611 | 0.709 | 17.9 | 4.24 |
