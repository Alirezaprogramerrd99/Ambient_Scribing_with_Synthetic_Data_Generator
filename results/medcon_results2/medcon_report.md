# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.8013 | 0.7445 | 0.7658 | 64.6 | 69.4 | 50 |
| ft_rag | 0.8007 | 0.7473 | 0.7668 | 64.8 | 69.4 | 50 |
| baseline | 0.7362 | 0.5747 | 0.6421 | 54.2 | 69.4 | 50 |
| rag_only | 0.7329 | 0.5719 | 0.6393 | 54.2 | 69.4 | 50 |
| teacher | 0.7410 | 0.5959 | 0.6577 | 55.9 | 69.4 | 50 |
| rag_dense_only | 0.8443 | 0.6707 | 0.7360 | 55.3 | 69.4 | 50 |
| rag_dense_rerank | 0.7971 | 0.7497 | 0.7655 | 65.1 | 69.4 | 50 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.8030 | 0.7924 | 0.7959 | 68.4 | 69.4 | 50 |
| ft_rag | 0.8016 | 0.7872 | 0.7925 | 68.1 | 69.4 | 50 |
| baseline | 0.6231 | 0.5933 | 0.6048 | 66.1 | 69.4 | 50 |
| rag_only | 0.6155 | 0.5870 | 0.5976 | 66.1 | 69.4 | 50 |
| teacher | 0.7407 | 0.5943 | 0.6566 | 55.8 | 69.4 | 50 |
| rag_dense_only | 0.7957 | 0.7910 | 0.7915 | 69.0 | 69.4 | 50 |
| rag_dense_rerank | 0.8051 | 0.7939 | 0.7979 | 68.3 | 69.4 | 50 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7806 | 0.7699 | 0.7733 | 68.3 | 69.4 | 50 |
| ft_rag | 0.7879 | 0.7682 | 0.7756 | 67.5 | 69.4 | 50 |
| baseline | 0.6888 | 0.5299 | 0.5960 | 53.1 | 69.4 | 50 |
| rag_only | 0.6879 | 0.5248 | 0.5924 | 52.7 | 69.4 | 50 |
| teacher | 0.7410 | 0.5977 | 0.6587 | 56.1 | 69.4 | 50 |
| rag_dense_only | 0.7904 | 0.7647 | 0.7752 | 67.0 | 69.4 | 50 |
| rag_dense_rerank | 0.7801 | 0.7706 | 0.7732 | 68.5 | 69.4 | 50 |
