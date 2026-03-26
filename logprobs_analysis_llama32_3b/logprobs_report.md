# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence
for each generated token. Lower logprobs indicate higher uncertainty,
which has been shown to correlate with factual errors and hallucinations
(Kadavath et al., 2022; Kuhn et al., 2023).

## Llama3.2 (3B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0681 |
| Std logprob | 0.0131 |
| Mean perplexity | 1.07 |
| Mean frac low-conf tokens | 0.0006 |
| Num samples | 20 |
| Correlation (logprob vs hallucination) | 0.3350 |
| Correlation (logprob vs overall quality) | 0.2501 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0514 | 1.06 | 0.0000 | 20 |
| HPI | -0.1138 | 1.12 | 0.0031 | 20 |
| Past Medical Hx | -0.0189 | 1.02 | 0.0000 | 20 |
| Medications | -0.0067 | 1.01 | 0.0000 | 20 |
| Allergies | -0.0190 | 1.02 | 0.0000 | 20 |
| Examination | -0.0502 | 1.05 | 0.0000 | 20 |
| Assessment | -0.0897 | 1.10 | 0.0000 | 20 |
| Plan | -0.0959 | 1.10 | 0.0000 | 20 |
| Safety Netting | -0.0592 | 1.06 | 0.0000 | 20 |
