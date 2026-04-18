# Table C: RAG Ablation Study

All results use fine-tuned models. Teacher (GPT-4o-mini) shown as reference.
Per-section values show ROUGE-L for key clinical sections.

| Model | RAG Config | Scope | ROUGE-L | ROUGE-1 | MEDCON-F1 | Judge Avg |
|---|---|---|---|---|---|---|
| Phi-3.5 (3.8B) | Dense Only | **Overall** | — | — | 0.725 | 4.36 |
| | | HPI | 0.510 | 0.600 | — | — |
| | | Assessment | 0.472 | 0.513 | — | — |
| | | Plan | 0.403 | 0.454 | — | — |
|  | Dense+Rerank | **Overall** | — | — | 0.761 | 4.45 |
| | | HPI | 0.515 | 0.594 | — | — |
| | | Assessment | 0.500 | 0.557 | — | — |
| | | Plan | 0.512 | 0.569 | — | — |
| | Teacher (ref.) | **Overall** | — | — | 0.657 | 4.77 |
|---|---|---|---|---|---|---|
| Llama-3.2 (3B) | Dense Only | **Overall** | — | — | 0.770 | 4.28 |
| | | HPI | 0.507 | 0.597 | — | — |
| | | Assessment | 0.480 | 0.551 | — | — |
| | | Plan | 0.466 | 0.531 | — | — |
|  | Dense+Rerank | **Overall** | — | — | 0.781 | 4.31 |
| | | HPI | 0.503 | 0.599 | — | — |
| | | Assessment | 0.494 | 0.556 | — | — |
| | | Plan | 0.495 | 0.571 | — | — |
| | Teacher (ref.) | **Overall** | — | — | 0.657 | 4.77 |
|---|---|---|---|---|---|---|
| Llama-3.2 (1B) | Dense Only | **Overall** | — | — | 0.750 | 4.13 |
| | | HPI | 0.437 | 0.524 | — | — |
| | | Assessment | 0.453 | 0.509 | — | — |
| | | Plan | 0.455 | 0.524 | — | — |
|  | Dense+Rerank | **Overall** | — | — | 0.752 | 4.12 |
| | | HPI | 0.450 | 0.541 | — | — |
| | | Assessment | 0.500 | 0.555 | — | — |
| | | Plan | 0.483 | 0.549 | — | — |
| | Teacher (ref.) | **Overall** | — | — | 0.657 | 4.77 |
|---|---|---|---|---|---|---|