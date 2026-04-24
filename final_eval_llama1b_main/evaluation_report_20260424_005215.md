# Student Model Evaluation Report
**Date:** 2026-04-23T23:44:01.668647
**Test Samples:** 100
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-1B (no fine-tuning, no RAG) | 0.297 | N/A | 5.5 | 3.76 | 4.18 | 2.83 |
| Base Llama-3.2-1B + RAG (no fine-tuning) | 0.297 | N/A | 5.5 | 3.74 | 4.20 | 2.82 |
| Fine-tuned Llama-3.2-1B (no RAG) | 0.601 | N/A | 6.3 | 4.13 | 4.58 | 3.22 |
| Fine-tuned Llama-3.2-1B + RAG | 0.601 | N/A | 6.4 | 4.09 | 4.62 | 3.15 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 4.0 | 4.77 | 5.00 | 5.00 |
