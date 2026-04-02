# Student Model Evaluation Report
**Date:** 2026-03-25T19:33:19.966899
**Test Samples:** 50
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Base Llama-3.2-3B (no fine-tuning, no RAG) | 0.342 | N/A | 10.3 | 4.52 | 4.98 | 4.96 |
| Base Llama-3.2-3B + RAG (no fine-tuning) | 0.338 | N/A | 10.2 | 4.47 | 4.94 | 4.92 |
| Fine-tuned Llama-3.2-3B (no RAG) | 0.638 | N/A | 11.2 | 4.40 | 4.80 | 3.84 |
| Fine-tuned Llama-3.2-3B + RAG | 0.635 | N/A | 11.1 | 4.37 | 4.84 | 3.78 |

**Skipped configurations:**
- teacher: name 'outputs' is not defined

## 2. RAG Backend Comparison

| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) | Judge Overall |
|---|---|---|---|---|
| dense_only | 0.623 | 0.728 | 11.5 | 4.28 |
| dense_rerank | 0.632 | 0.000 | 11.3 | 4.31 |
