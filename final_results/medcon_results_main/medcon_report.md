# MEDCON (Medical Concept Overlap) Results

Computed using QuickUMLS with UMLS 2025AB.
MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)
between generated and reference clinical summaries.

## Phi-3.5 (3.8B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.8017 | 0.7719 | 0.7846 | 67.6 | 70.3 | 100 |
| ft_rag | 0.8020 | 0.7684 | 0.7828 | 67.3 | 70.3 | 100 |
| baseline | 0.7400 | 0.5812 | 0.6481 | 55.4 | 70.3 | 100 |
| rag_only | 0.7425 | 0.5699 | 0.6417 | 53.9 | 70.3 | 100 |
| teacher | 0.7524 | 0.5942 | 0.6612 | 55.6 | 70.3 | 100 |

## Llama-3.2 (3B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.8094 | 0.7890 | 0.7974 | 68.5 | 70.3 | 100 |
| ft_rag | 0.8058 | 0.7862 | 0.7940 | 68.6 | 70.3 | 100 |
| baseline | 0.6149 | 0.5832 | 0.5966 | 66.5 | 70.3 | 100 |
| rag_only | 0.6178 | 0.5835 | 0.5979 | 66.5 | 70.3 | 100 |
| teacher | 0.7544 | 0.5962 | 0.6634 | 55.6 | 70.3 | 100 |

## Llama-3.2 (1B)

| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |
|---|---|---|---|---|---|---|
| ft_only | 0.7862 | 0.7729 | 0.7775 | 69.0 | 70.3 | 100 |
| ft_rag | 0.7873 | 0.7779 | 0.7806 | 69.4 | 70.3 | 100 |
| baseline | 0.6899 | 0.5280 | 0.5953 | 53.8 | 70.3 | 100 |
| rag_only | 0.6865 | 0.5240 | 0.5920 | 53.5 | 70.3 | 100 |
| teacher | 0.7519 | 0.5968 | 0.6630 | 55.8 | 70.3 | 100 |
