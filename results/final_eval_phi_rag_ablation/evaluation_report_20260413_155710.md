# Student Model Evaluation Report
**Date:** 2026-04-13T15:55:10.352430
**Test Samples:** 2
**Student Model:** clinical-scribe
**Judge Model:** gpt-4o-mini

## 1. Comparative Experiment Results

| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) | Judge Overall | Judge Safety | Judge Halluc |
|---|---|---|---|---|---|---|
| Fine-tuned Phi-3.5-mini + RAG | 0.584 | N/A | 16.7 | 4.35 | 5.00 | 3.50 |
| Teacher (gpt-4o-mini + RAG) | 0.418 | N/A | 4.9 | 4.67 | 5.00 | 5.00 |
