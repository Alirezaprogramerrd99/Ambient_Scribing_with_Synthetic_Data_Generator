# Student Model Evaluation Report
**Date:** 2026-04-23T22:09:29.124761
**Test Samples:** 100
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.345 | N/A | 9.4 | 4.51 | 4.99 | 4.99 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.343 | N/A | 9.7 | 4.48 | 4.99 | 4.97 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.630 | N/A | 10.5 | 4.33 | 4.79 | 3.66 |
| Fine-tuned Llama-3.2-3B + RAG | 0.632 | N/A | 10.5 | 4.38 | 4.80 | 3.79 |
| Teacher (gpt-4o-mini + RAG) | 0.429 | N/A | 3.3 | 4.76 | 5.00 | 5.00 |
