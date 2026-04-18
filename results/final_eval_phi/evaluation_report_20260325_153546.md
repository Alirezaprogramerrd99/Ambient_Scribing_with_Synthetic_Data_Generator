# Student Model Evaluation Report
**Date:** 2026-03-25T15:27:03.641008
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Phi-3.5-mini (no fine-tuning, no RAG) | 0.362 | N/A | 8.9 | 4.75 | 5.00 | 5.00 |
| Base Phi-3.5-mini + RAG (no fine-tuning) | 0.230 | N/A | 55.6 | 3.83 | 5.00 | 3.00 |
| Fine-tuned Phi-3.5-mini (no RAG) | 0.639 | N/A | 14.2 | 4.33 | 5.00 | 4.00 |
| Fine-tuned Phi-3.5-mini + RAG | 0.579 | N/A | 14.1 | 4.10 | 5.00 | 3.00 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.552 | 0.693 | 14.2 | 3.94 |
| dense_rerank | 0.629 | 0.000 | 12.7 | 4.35 |
