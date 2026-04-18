# Student Model Evaluation Report
**Date:** 2026-03-11T15:24:42.306798
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.338 | N/A | 9.9 | 4.65 | 4.96 | 4.92 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.331 | N/A | 10.7 | 4.50 | 4.82 | 4.52 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.632 | N/A | 9.9 | 4.34 | 4.84 | 3.68 |
| Fine-tuned Llama-3.2-3B + RAG | 0.628 | N/A | 10.2 | 4.33 | 4.70 | 3.70 |
| Teacher (gpt-4o-mini + RAG) | 0.432 | N/A | 6.3 | 4.90 | 5.00 | 5.00 |

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| llama_index | 0.628 | 0.729 | 10.4 | 4.30 |
| manual | 0.632 | 0.000 | 10.5 | 4.33 |
| hybrid | 0.628 | 0.729 | 10.7 | 4.31 |
