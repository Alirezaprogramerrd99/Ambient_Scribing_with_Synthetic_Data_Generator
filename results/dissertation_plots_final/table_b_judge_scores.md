# Table B: LLM-as-Judge Evaluation (GPT-4o-mini, 1-5 scale)

Higher is better for all dimensions. Halluc.↑ means higher = less hallucination.

| Model | Config | RAG | Clin.Acc | Complete | Halluc.↑ | Safety | Coherence | Concise | Overall | Critical Err |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5 (3.8B) | FT | None | 4.16 | 4.22 | 3.88 | 4.86 | 4.96 | 4.52 | 4.44 | — |
|  | FT+RAG | Dense+Rerank | 4.20 | 4.16 | 4.10 | 4.84 | 4.90 | 4.48 | 4.45 | — |
|  | Base | None | 4.76 | 4.52 | 5.00 | 4.98 | 4.92 | 4.30 | 4.74 | — |
|  | Base+RAG | Dense+Rerank | 4.82 | 4.56 | 5.00 | 4.98 | 4.88 | 4.32 | 4.76 | — |
|  | Teacher | None (API) | 4.86 | 4.36 | 5.00 | 5.00 | 4.96 | 4.52 | 4.77 | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (3B) | FT | None | 4.16 | 4.26 | 3.84 | 4.80 | 5.00 | 4.32 | 4.40 | — |
|  | FT+RAG | Dense+Rerank | 4.02 | 4.26 | 3.78 | 4.84 | 4.96 | 4.34 | 4.37 | — |
|  | Base | None | 4.46 | 4.08 | 4.96 | 4.98 | 4.66 | 3.94 | 4.52 | — |
|  | Base+RAG | Dense+Rerank | 4.38 | 4.08 | 4.92 | 4.94 | 4.60 | 3.94 | 4.47 | — |
|  | Teacher | None (API) | 4.86 | 4.36 | 5.00 | 5.00 | 4.96 | 4.52 | 4.77 | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (1B) | FT | None | 3.72 | 4.10 | 3.22 | 4.58 | 4.70 | 4.10 | 4.08 | — |
|  | FT+RAG | Dense+Rerank | 3.76 | 4.14 | 3.12 | 4.56 | 4.82 | 4.08 | 4.10 | — |
|  | Base | None | 3.26 | 3.88 | 2.82 | 4.04 | 4.40 | 3.72 | 3.72 | — |
|  | Base+RAG | Dense+Rerank | 3.40 | 4.02 | 2.78 | 4.10 | 4.34 | 3.68 | 3.73 | — |
|  | Teacher | None (API) | 4.86 | 4.36 | 5.00 | 5.00 | 4.96 | 4.52 | 4.77 | — |
|---|---|---|---|---|---|---|---|---|---|---|