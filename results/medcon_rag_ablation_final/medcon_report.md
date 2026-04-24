# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.7956 | 0.7623 | 0.7769 | 66.3 | 69.4 | 50 |
| teacher | 0.7426 | 0.6007 | 0.6614 | 56.3 | 69.4 | 50 |
| rag_dense_only | 0.8390 | 0.6788 | 0.7426 | 56.3 | 69.4 | 50 |
| rag_dense_rerank | 0.7990 | 0.7450 | 0.7652 | 64.8 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.7993 | 0.7604 | 0.7776 | 65.9 | 69.4 | 50 |
| rag_full_medical | 0.7980 | 0.7528 | 0.7673 | 65.4 | 69.4 | 50 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.8007 | 0.7882 | 0.7927 | 68.2 | 69.4 | 50 |
| teacher | 0.7488 | 0.5978 | 0.6619 | 55.5 | 69.4 | 50 |
| rag_dense_only | 0.7952 | 0.7857 | 0.7883 | 68.6 | 69.4 | 50 |
| rag_dense_rerank | 0.7987 | 0.7877 | 0.7912 | 68.4 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.8046 | 0.7980 | 0.7997 | 68.8 | 69.4 | 50 |
| rag_full_medical | 0.8021 | 0.7892 | 0.7938 | 68.3 | 69.4 | 50 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.7841 | 0.7760 | 0.7775 | 68.6 | 69.4 | 50 |
| teacher | 0.7405 | 0.5928 | 0.6556 | 55.7 | 69.4 | 50 |
| rag_dense_only | 0.7832 | 0.7652 | 0.7721 | 67.7 | 69.4 | 50 |
| rag_dense_rerank | 0.7821 | 0.7694 | 0.7740 | 68.2 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.7790 | 0.7698 | 0.7726 | 68.5 | 69.4 | 50 |
| rag_full_medical | 0.7717 | 0.7638 | 0.7658 | 68.6 | 69.4 | 50 |
