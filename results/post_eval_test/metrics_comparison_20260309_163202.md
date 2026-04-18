# Post-Evaluation Metrics Comparison Report
**Generated:** 2026-03-09 16:32

## Configuration: baseline

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.3722 |
| Avg Time (s) | 9.0400 |
| Judge Overall | 4.75/5 |
| Accuracy | 4.80/5 |
| Completeness | 4.40/5 |
| Hallucination | 5.00/5 |
| Safety | 5.00/5 |
| Coherence | 4.90/5 |
| Conciseness | 4.40/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.3011 |
| BLEU-4 | 0.1098 |
| MEDCON-F1 | 0.6934 |
| MEDCON-P | 0.7897 |
| MEDCON-R | 0.6605 |

## Configuration: rag_only

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.3013 |
| Avg Time (s) | 33.7027 |
| Judge Overall | 4.08/5 |
| Accuracy | 4.40/5 |
| Completeness | 4.30/5 |
| Hallucination | 3.60/5 |
| Safety | 4.60/5 |
| Coherence | 3.80/5 |
| Conciseness | 3.80/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.3379 |
| BLEU-4 | 0.1051 |
| MEDCON-F1 | 0.7355 |
| MEDCON-P | 0.7944 |
| MEDCON-R | 0.7276 |

## Configuration: ft_only

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.6373 |
| Avg Time (s) | 13.4310 |
| Judge Overall | 4.18/5 |
| Accuracy | 3.70/5 |
| Completeness | 4.00/5 |
| Hallucination | 3.80/5 |
| Safety | 4.70/5 |
| Coherence | 4.80/5 |
| Conciseness | 4.00/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.7584 |
| BLEU-4 | 0.5576 |
| MEDCON-F1 | 0.8227 |
| MEDCON-P | 0.8592 |
| MEDCON-R | 0.8225 |

## Configuration: ft_rag

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.5620 |
| Avg Time (s) | 13.1788 |
| Judge Overall | 4.39/5 |
| Accuracy | 4.20/5 |
| Completeness | 4.20/5 |
| Hallucination | 4.00/5 |
| Safety | 4.80/5 |
| Coherence | 4.90/5 |
| Conciseness | 4.20/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.6174 |
| BLEU-4 | 0.4675 |
| MEDCON-F1 | 0.7802 |
| MEDCON-P | 0.8594 |
| MEDCON-R | 0.7858 |

## Configuration: teacher

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.3997 |
| Avg Time (s) | 5.7781 |
| Judge Overall | 4.87/5 |
| Accuracy | 4.90/5 |
| Completeness | 4.70/5 |
| Hallucination | 5.00/5 |
| Safety | 5.00/5 |
| Coherence | 5.00/5 |
| Conciseness | 4.60/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.5074 |
| BLEU-4 | 0.2896 |
| MEDCON-F1 | 0.7290 |
| MEDCON-P | 0.7620 |
| MEDCON-R | 0.7159 |

## RAG Backend Comparison

### Backend: hybrid

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.5620 |
| Avg Time (s) | 14.5137 |
| Judge Overall | 4.41/5 |
| Accuracy | 4.20/5 |
| Completeness | 4.20/5 |
| Hallucination | 3.90/5 |
| Safety | 4.80/5 |
| Coherence | 4.90/5 |
| Conciseness | 4.40/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.6174 |
| BLEU-4 | 0.4675 |
| MEDCON-F1 | 0.7802 |
| MEDCON-P | 0.8594 |
| MEDCON-R | 0.7858 |

### Backend: llama_index

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.5620 |
| Avg Time (s) | 13.7071 |
| Judge Overall | 4.44/5 |
| Accuracy | 4.20/5 |
| Completeness | 4.20/5 |
| Hallucination | 4.00/5 |
| Safety | 4.80/5 |
| Coherence | 5.00/5 |
| Conciseness | 4.40/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.6174 |
| BLEU-4 | 0.4675 |
| MEDCON-F1 | 0.7802 |
| MEDCON-P | 0.8594 |
| MEDCON-R | 0.7858 |

### Backend: manual

| Metric | Phi-3.5 (3.8B) |
|---|---|
| ROUGE-L | 0.6373 |
| Avg Time (s) | 13.4137 |
| Judge Overall | 4.19/5 |
| Accuracy | 3.70/5 |
| Completeness | 4.10/5 |
| Hallucination | 3.70/5 |
| Safety | 4.60/5 |
| Coherence | 4.80/5 |
| Conciseness | 4.20/5 |
| BERTScore-F1 | — |
| BERTScore-P | — |
| BERTScore-R | — |
| BLEU-1 | 0.7584 |
| BLEU-4 | 0.5576 |
| MEDCON-F1 | 0.8227 |
| MEDCON-P | 0.8592 |
| MEDCON-R | 0.8225 |
