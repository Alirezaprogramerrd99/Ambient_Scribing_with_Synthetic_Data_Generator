# Post-Evaluation Metrics Comparison Report
**Generated:** 2026-03-26 18:22

## Configuration: baseline

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.3689 | 0.3319 | 0.2829 |
| Avg Time (s) | 9.9817 | 10.3188 | 5.8941 |
| Judge Overall | 4.74/5 | 4.52/5 | 3.72/5 |
| Accuracy | 4.76/5 | 4.46/5 | 3.26/5 |
| Completeness | 4.52/5 | 4.08/5 | 3.88/5 |
| Hallucination | 5.00/5 | 4.96/5 | 2.82/5 |
| Safety | 4.98/5 | 4.98/5 | 4.04/5 |
| Coherence | 4.92/5 | 4.66/5 | 4.40/5 |
| Conciseness | 4.30/5 | 3.94/5 | 3.72/5 |
| BERTScore-F1 | 0.8697 | 0.8661 | 0.8659 |
| BERTScore-P | 0.8819 | 0.8669 | 0.8736 |
| BERTScore-R | 0.8581 | 0.8654 | 0.8584 |
| BLEU-1 | 0.3253 | 0.5634 | 0.4941 |
| BLEU-4 | 0.1151 | 0.2932 | 0.1958 |
| MEDCON-F1 | 0.7599 | 0.7444 | 0.6866 |
| MEDCON-P | 0.8375 | 0.7785 | 0.7978 |
| MEDCON-R | 0.7119 | 0.7299 | 0.6258 |

## Configuration: rag_only

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.3667 | 0.3270 | 0.2865 |
| Avg Time (s) | 9.8463 | 10.2059 | 5.7421 |
| Judge Overall | 4.76/5 | 4.47/5 | 3.73/5 |
| Accuracy | 4.82/5 | 4.38/5 | 3.40/5 |
| Completeness | 4.56/5 | 4.08/5 | 4.02/5 |
| Hallucination | 5.00/5 | 4.92/5 | 2.78/5 |
| Safety | 4.98/5 | 4.94/5 | 4.10/5 |
| Coherence | 4.88/5 | 4.60/5 | 4.34/5 |
| Conciseness | 4.32/5 | 3.94/5 | 3.68/5 |
| BERTScore-F1 | 0.8699 | 0.8665 | 0.8666 |
| BERTScore-P | 0.8833 | 0.8675 | 0.8750 |
| BERTScore-R | 0.8570 | 0.8655 | 0.8584 |
| BLEU-1 | 0.3239 | 0.5600 | 0.4979 |
| BLEU-4 | 0.1154 | 0.2929 | 0.2000 |
| MEDCON-F1 | 0.7354 | 0.7428 | 0.6973 |
| MEDCON-P | 0.8125 | 0.7980 | 0.8071 |
| MEDCON-R | 0.6893 | 0.7163 | 0.6349 |

## Configuration: ft_only

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.6217 | 0.6282 | 0.5887 |
| Avg Time (s) | 12.5657 | 11.1905 | 6.5545 |
| Judge Overall | 4.44/5 | 4.40/5 | 4.08/5 |
| Accuracy | 4.16/5 | 4.16/5 | 3.72/5 |
| Completeness | 4.22/5 | 4.26/5 | 4.10/5 |
| Hallucination | 3.88/5 | 3.84/5 | 3.22/5 |
| Safety | 4.86/5 | 4.80/5 | 4.58/5 |
| Coherence | 4.96/5 | 5.00/5 | 4.70/5 |
| Conciseness | 4.52/5 | 4.32/5 | 4.10/5 |
| BERTScore-F1 | 0.9400 | 0.9401 | 0.9351 |
| BERTScore-P | 0.9448 | 0.9394 | 0.9356 |
| BERTScore-R | 0.9354 | 0.9408 | 0.9347 |
| BLEU-1 | 0.7479 | 0.7677 | 0.7462 |
| BLEU-4 | 0.5459 | 0.5607 | 0.5283 |
| MEDCON-F1 | 0.8581 | 0.8649 | 0.8399 |
| MEDCON-P | 0.9141 | 0.9028 | 0.8818 |
| MEDCON-R | 0.8324 | 0.8497 | 0.8215 |

## Configuration: ft_rag

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.6245 | 0.6247 | 0.5904 |
| Avg Time (s) | 13.4225 | 11.0747 | 6.6823 |
| Judge Overall | 4.45/5 | 4.37/5 | 4.10/5 |
| Accuracy | 4.20/5 | 4.02/5 | 3.76/5 |
| Completeness | 4.16/5 | 4.26/5 | 4.14/5 |
| Hallucination | 4.10/5 | 3.78/5 | 3.12/5 |
| Safety | 4.84/5 | 4.84/5 | 4.56/5 |
| Coherence | 4.90/5 | 4.96/5 | 4.82/5 |
| Conciseness | 4.48/5 | 4.34/5 | 4.08/5 |
| BERTScore-F1 | 0.9391 | 0.9390 | 0.9349 |
| BERTScore-P | 0.9444 | 0.9386 | 0.9349 |
| BERTScore-R | 0.9340 | 0.9393 | 0.9349 |
| BLEU-1 | 0.7424 | 0.7627 | 0.7460 |
| BLEU-4 | 0.5463 | 0.5514 | 0.5273 |
| MEDCON-F1 | 0.8446 | 0.8419 | 0.8596 |
| MEDCON-P | 0.8869 | 0.8915 | 0.8967 |
| MEDCON-R | 0.8267 | 0.8190 | 0.8440 |

## Configuration: teacher

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | — | — | — |
| Avg Time (s) | — | — | — |
| Judge Overall | — | — | — |
| Accuracy | — | — | — |
| Completeness | — | — | — |
| Hallucination | — | — | — |
| Safety | — | — | — |
| Coherence | — | — | — |
| Conciseness | — | — | — |
| BERTScore-F1 | — | — | — |
| BERTScore-P | — | — | — |
| BERTScore-R | — | — | — |
| BLEU-1 | — | — | — |
| BLEU-4 | — | — | — |
| MEDCON-F1 | — | — | — |
| MEDCON-P | — | — | — |
| MEDCON-R | — | — | — |

## RAG Backend Comparison

### Backend: dense_only

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.5245 | 0.6137 | 0.5884 |
| Avg Time (s) | 13.6752 | 11.4627 | 6.7119 |
| Judge Overall | 4.36/5 | 4.28/5 | 4.13/5 |
| Accuracy | 4.16/5 | 4.02/5 | 3.84/5 |
| Completeness | 4.22/5 | 4.22/5 | 4.10/5 |
| Hallucination | 4.00/5 | 3.60/5 | 3.30/5 |
| Safety | 4.80/5 | 4.72/5 | 4.58/5 |
| Coherence | 4.68/5 | 4.88/5 | 4.78/5 |
| Conciseness | 4.28/5 | 4.16/5 | 4.10/5 |
| BERTScore-F1 | 0.9194 | 0.9364 | 0.9353 |
| BERTScore-P | 0.9347 | 0.9351 | 0.9357 |
| BERTScore-R | 0.9051 | 0.9377 | 0.9349 |
| BLEU-1 | 0.5421 | 0.7493 | 0.7503 |
| BLEU-4 | 0.4025 | 0.5413 | 0.5295 |
| MEDCON-F1 | 0.7899 | 0.8764 | 0.8408 |
| MEDCON-P | 0.8879 | 0.8972 | 0.8714 |
| MEDCON-R | 0.7526 | 0.8742 | 0.8313 |

### Backend: dense_rerank

| Metric | Phi-3.5 (3.8B) | Llama-3.2 (3B) | Llama-3.2 (1B) |
|---|---|---|---|
| ROUGE-L | 0.6241 | 0.6238 | 0.5946 |
| Avg Time (s) | 14.1801 | 11.2849 | 6.5109 |
| Judge Overall | 4.45/5 | 4.31/5 | 4.12/5 |
| Accuracy | 4.20/5 | 3.96/5 | 3.80/5 |
| Completeness | 4.20/5 | 4.22/5 | 4.12/5 |
| Hallucination | 3.96/5 | 3.72/5 | 3.24/5 |
| Safety | 4.88/5 | 4.74/5 | 4.56/5 |
| Coherence | 4.94/5 | 4.98/5 | 4.74/5 |
| Conciseness | 4.46/5 | 4.20/5 | 4.10/5 |
| BERTScore-F1 | 0.9409 | 0.9386 | 0.9357 |
| BERTScore-P | 0.9451 | 0.9376 | 0.9359 |
| BERTScore-R | 0.9367 | 0.9396 | 0.9356 |
| BLEU-1 | 0.7510 | 0.7601 | 0.7486 |
| BLEU-4 | 0.5521 | 0.5512 | 0.5289 |
| MEDCON-F1 | 0.8637 | 0.8501 | 0.8510 |
| MEDCON-P | 0.9050 | 0.8876 | 0.8986 |
| MEDCON-R | 0.8488 | 0.8384 | 0.8276 |
