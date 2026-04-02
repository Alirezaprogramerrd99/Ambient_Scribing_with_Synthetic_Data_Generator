# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence
for each generated token. Lower logprobs indicate higher uncertainty,
which has been shown to correlate with factual errors and hallucinations
(Kadavath et al., 2022; Kuhn et al., 2023).

## Llama3.2 (1B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0745 |
| Std logprob | 0.0123 |
| Mean perplexity | 1.08 |
| Mean frac low-conf tokens | 0.0005 |
| Num samples | 20 |
| Correlation (logprob vs hallucination) | -0.1860 |
| Correlation (logprob vs overall quality) | -0.1666 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0271 | 1.03 | 0.0000 | 20 |
| HPI | -0.1150 | 1.12 | 0.0023 | 20 |
| Past Medical Hx | -0.0132 | 1.01 | 0.0000 | 20 |
| Medications | -0.0205 | 1.02 | 0.0000 | 20 |
| Allergies | -0.0312 | 1.03 | 0.0000 | 20 |
| Examination | -0.0715 | 1.07 | 0.0000 | 20 |
| Assessment | -0.1119 | 1.12 | 0.0000 | 20 |
| Plan | -0.0809 | 1.09 | 0.0000 | 20 |
| Safety Netting | -0.0661 | 1.07 | 0.0000 | 20 |
