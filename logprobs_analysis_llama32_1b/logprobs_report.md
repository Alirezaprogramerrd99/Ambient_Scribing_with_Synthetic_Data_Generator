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

## Llama3.2 (1B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0745 |
| Std logprob | 0.0123 |
| Mean perplexity | 1.08 |
| Mean frac low-conf tokens | 0.0005 |
| Num samples | 20 |
| Correlation (logprob ↔ hallucination) | -0.1860 |
| Correlation (logprob ↔ overall quality) | -0.1666 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf |
|---|---|---|---|
| Chief Complaint | -0.0271 | 1.03 | 0.0000 |
| Medications | -0.0203 | 1.02 | 0.0000 |
| Allergies | -0.0312 | 1.03 | 0.0000 |
| Examination | -0.0715 | 1.07 | 0.0000 |
| Assessment | -0.1119 | 1.12 | 0.0000 |
| Plan | -0.0806 | 1.08 | 0.0000 |
| Safety Netting | -0.0663 | 1.07 | 0.0000 |
