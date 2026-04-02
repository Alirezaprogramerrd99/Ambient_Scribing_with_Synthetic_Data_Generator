# Student Model Evaluation Report
**Date:** 2026-03-11T14:13:20.142620
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-1B (no fine-tuning, no RAG) | 0.294 | N/A | 5.1 | 3.69 | 4.10 | 2.72 |
| Base Llama-3.2-1B + RAG (no fine-tuning) | 0.284 | N/A | 5.5 | 3.43 | 3.70 | 2.48 |
| Fine-tuned Llama-3.2-1B (no RAG) | 0.602 | N/A | 5.5 | 4.09 | 4.54 | 3.08 |
| Fine-tuned Llama-3.2-1B + RAG | 0.604 | N/A | 5.7 | 4.17 | 4.58 | 3.40 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 5.7 | 4.90 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.604 | 0.729 | 5.6 | 4.21 |
| manual | 0.602 | 0.000 | 5.6 | 4.10 |
| hybrid | 0.604 | 0.729 | 5.6 | 4.18 |
