# Student Model Evaluation Report
**Date:** 2026-04-23T20:17:39.946994
**Test Samples:** 100
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.383 | N/A | 9.8 | 4.70 | 4.98 | 4.93 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.384 | N/A | 9.2 | 4.72 | 4.99 | 4.98 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.633 | N/A | 13.2 | 4.41 | 4.89 | 3.91 |
| Fine-tuned Phi-3.5-mini + RAG | 0.640 | N/A | 13.3 | 4.38 | 4.80 | 3.93 |
| Teacher (gpt-4o-mini + RAG) | 0.430 | N/A | 4.3 | 4.75 | 4.99 | 5.00 |
