# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7894 | 0.7300 | 0.7512 | 107.1 | 116.3 | 50 |
| ft_rag | 0.7859 | 0.7240 | 0.7459 | 106.6 | 116.3 | 50 |
| baseline | 0.6974 | 0.5791 | 0.6273 | 96.5 | 116.3 | 50 |
| rag_only | 0.6994 | 0.5822 | 0.6305 | 96.7 | 116.3 | 50 |
| teacher | 0.7207 | 0.6130 | 0.6578 | 99.1 | 116.3 | 50 |
| rag_dense_only | 0.8177 | 0.6672 | 0.7217 | 94.6 | 116.3 | 50 |
| rag_dense_rerank | 0.7877 | 0.7321 | 0.7493 | 107.6 | 116.3 | 50 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7843 | 0.7821 | 0.7795 | 116.2 | 116.3 | 50 |
| ft_rag | 0.7839 | 0.7770 | 0.7765 | 115.6 | 116.3 | 50 |
| baseline | 0.6181 | 0.5937 | 0.6020 | 111.6 | 116.3 | 50 |
| rag_only | 0.6057 | 0.5909 | 0.5942 | 113.4 | 116.3 | 50 |
| teacher | 0.7186 | 0.6111 | 0.6555 | 99.0 | 116.3 | 50 |
| rag_dense_only | 0.7750 | 0.7749 | 0.7714 | 116.6 | 116.3 | 50 |
| rag_dense_rerank | 0.7893 | 0.7844 | 0.7832 | 115.8 | 116.3 | 50 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7531 | 0.7560 | 0.7519 | 116.7 | 116.3 | 50 |
| ft_rag | 0.7624 | 0.7552 | 0.7560 | 115.2 | 116.3 | 50 |
| baseline | 0.6244 | 0.5445 | 0.5769 | 101.9 | 116.3 | 50 |
| rag_only | 0.6218 | 0.5403 | 0.5742 | 100.8 | 116.3 | 50 |
| teacher | 0.7185 | 0.6142 | 0.6570 | 99.5 | 116.3 | 50 |
| rag_dense_only | 0.7617 | 0.7441 | 0.7501 | 113.6 | 116.3 | 50 |
| rag_dense_rerank | 0.7595 | 0.7537 | 0.7539 | 115.5 | 116.3 | 50 |
