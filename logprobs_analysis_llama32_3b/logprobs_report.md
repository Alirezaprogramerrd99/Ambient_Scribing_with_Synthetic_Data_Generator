# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence
for each generated token. Lower logprobs indicate higher uncertainty,
which has been shown to correlate with factual errors and hallucinations
(Kadavath et al., 2022; Kuhn et al., 2023).

### Key Metrics

- **Mean Log Probability**: Average confidence across all tokens.
  More negative values indicate lower overall confidence.
- **Perplexity**: exp(-mean_logprob). Lower is more confident.
  Equivalent to the geometric mean of inverse token probabilities.
- **Fraction Low-Confidence Tokens**: Proportion of tokens with
  logprob < -3.0, indicating uncertain predictions.
- **Per-Section Confidence**: Identifies which clinical sections
  the model is least confident about.

## Llama3.2 (3B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0681 |
| Std logprob | 0.0131 |
| Mean perplexity | 1.07 |
| Mean frac low-conf tokens | 0.0006 |
| Num samples | 20 |
| Correlation (logprob ↔ hallucination) | 0.3350 |
| Correlation (logprob ↔ overall quality) | 0.2501 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf |
|---|---|---|---|
| Chief Complaint | -0.0514 | 1.06 | 0.0000 |
| Medications | 0.0000 | 1.00 | 0.0000 |
| Allergies | -0.0196 | 1.02 | 0.0000 |
| Examination | -0.0502 | 1.05 | 0.0000 |
| Assessment | -0.0897 | 1.10 | 0.0000 |
| Plan | -0.0958 | 1.10 | 0.0000 |
| Safety Netting | -0.0592 | 1.06 | 0.0000 |
