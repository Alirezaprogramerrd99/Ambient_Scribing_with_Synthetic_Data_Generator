# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.8002 | 0.7486 | 0.7685 | 64.9 | 69.4 | 50 |
| teacher | 0.7426 | 0.5967 | 0.6593 | 55.9 | 69.4 | 50 |
| rag_dense_only | 0.8462 | 0.6766 | 0.7443 | 55.7 | 69.4 | 50 |
| rag_dense_rerank | 0.7988 | 0.7664 | 0.7803 | 66.4 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.7935 | 0.7506 | 0.7640 | 65.6 | 69.4 | 50 |
| rag_full_medical | 0.7956 | 0.7638 | 0.7779 | 66.5 | 69.4 | 50 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.8016 | 0.7893 | 0.7937 | 68.3 | 69.4 | 50 |
| teacher | 0.7366 | 0.5956 | 0.6560 | 56.1 | 69.4 | 50 |
| rag_dense_only | 0.8071 | 0.7957 | 0.7994 | 68.4 | 69.4 | 50 |
| rag_dense_rerank | 0.7988 | 0.7884 | 0.7919 | 68.5 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.8027 | 0.7922 | 0.7954 | 68.5 | 69.4 | 50 |
| rag_full_medical | 0.8057 | 0.7923 | 0.7973 | 68.2 | 69.4 | 50 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_rag | 0.7804 | 0.7722 | 0.7741 | 68.6 | 69.4 | 50 |
| teacher | 0.7459 | 0.5981 | 0.6609 | 55.8 | 69.4 | 50 |
| rag_dense_only | 0.7891 | 0.7695 | 0.7768 | 67.6 | 69.4 | 50 |
| rag_dense_rerank | 0.7780 | 0.7689 | 0.7715 | 68.5 | 69.4 | 50 |
| rag_dense_rerank_qe | 0.7806 | 0.7699 | 0.7727 | 68.4 | 69.4 | 50 |
| rag_full_medical | 0.7797 | 0.7633 | 0.7696 | 67.8 | 69.4 | 50 |
