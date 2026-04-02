# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7907 | 0.7492 | 0.7666 | 109.8 | 116.3 | 50 |
| ft_rag | 0.7996 | 0.7362 | 0.7576 | 106.5 | 116.3 | 50 |
| baseline | 0.6953 | 0.5832 | 0.6290 | 97.6 | 116.3 | 50 |
| rag_only | 0.7031 | 0.5847 | 0.6320 | 96.7 | 116.3 | 50 |
| teacher | 0.7205 | 0.6122 | 0.6575 | 98.8 | 116.3 | 50 |
| rag_dense_only | 0.8144 | 0.6698 | 0.7248 | 95.6 | 116.3 | 50 |
| rag_dense_rerank | 0.7855 | 0.7443 | 0.7614 | 109.6 | 116.3 | 50 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7954 | 0.7872 | 0.7883 | 115.1 | 116.3 | 50 |
| ft_rag | 0.7936 | 0.7749 | 0.7804 | 113.6 | 116.3 | 50 |
| baseline | 0.6045 | 0.6019 | 0.5995 | 115.7 | 116.3 | 50 |
| rag_only | 0.6064 | 0.6005 | 0.6003 | 114.6 | 116.3 | 50 |
| teacher | 0.7205 | 0.6122 | 0.6575 | 98.8 | 116.3 | 50 |
| rag_dense_only | 0.7687 | 0.7783 | 0.7697 | 118.1 | 116.3 | 50 |
| rag_dense_rerank | 0.7834 | 0.7845 | 0.7811 | 116.5 | 116.3 | 50 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7523 | 0.7434 | 0.7449 | 114.9 | 116.3 | 50 |
| ft_rag | 0.7578 | 0.7595 | 0.7558 | 116.7 | 116.3 | 50 |
| baseline | 0.6231 | 0.5578 | 0.5848 | 104.1 | 116.3 | 50 |
| rag_only | 0.6257 | 0.5555 | 0.5836 | 103.3 | 116.3 | 50 |
| teacher | 0.7205 | 0.6122 | 0.6575 | 98.8 | 116.3 | 50 |
| rag_dense_only | 0.7574 | 0.7488 | 0.7500 | 115.3 | 116.3 | 50 |
| rag_dense_rerank | 0.7545 | 0.7547 | 0.7517 | 116.7 | 116.3 | 50 |
