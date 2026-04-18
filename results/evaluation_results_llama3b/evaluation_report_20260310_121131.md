# Student Model Evaluation Report
**Date:** 2026-03-10T12:03:36.536350
**Test Samples:** 3
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.312 | N/A | 10.3 | 4.55 | 5.00 | 4.67 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.306 | N/A | 11.4 | 4.12 | 4.67 | 3.33 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.606 | N/A | 10.6 | 4.18 | 4.67 | 3.33 |
| Fine-tuned Llama-3.2-3B + RAG | 0.624 | N/A | 10.1 | 4.07 | 4.67 | 3.33 |
| Teacher (gpt-4o-mini + RAG) | 0.401 | N/A | 6.5 | 4.72 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.624 | 0.718 | 10.4 | 4.12 |
| manual | 0.606 | 0.000 | 10.9 | 4.18 |
| hybrid | 0.624 | 0.718 | 10.0 | 4.12 |
