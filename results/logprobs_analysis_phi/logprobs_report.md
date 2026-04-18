# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence
for each generated token. Lower logprobs indicate higher uncertainty,
which has been shown to correlate with factual errors and hallucinations
(Kadavath et al., 2022; Kuhn et al., 2023).

## Phi-3.5 (3.8B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0545 |
| Std logprob | 0.0081 |
| Mean perplexity | 1.06 |
| Mean frac low-conf tokens | 0.0000 |
| Num samples | 20 |
| Correlation (logprob vs hallucination) | 0.1808 |
| Correlation (logprob vs overall quality) | 0.0050 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0349 | 1.04 | 0.0000 | 20 |
| HPI | -0.1064 | 1.11 | 0.0000 | 20 |
| Past Medical Hx | -0.0032 | 1.00 | 0.0000 | 20 |
| Medications | -0.0124 | 1.01 | 0.0000 | 20 |
| Allergies | -0.0200 | 1.02 | 0.0000 | 19 |
| Examination | -0.0474 | 1.05 | 0.0000 | 18 |
| Assessment | -0.0736 | 1.08 | 0.0000 | 17 |
| Plan | -0.0709 | 1.07 | 0.0000 | 17 |
| Safety Netting | -0.0406 | 1.04 | 0.0000 | 12 |
