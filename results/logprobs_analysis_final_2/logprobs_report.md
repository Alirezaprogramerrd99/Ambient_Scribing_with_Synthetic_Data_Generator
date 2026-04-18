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
| Mean logprob | -0.0260 |
| Std logprob | 0.0064 |
| Mean perplexity | 1.03 |
| Mean frac low-conf tokens | 0.0350 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | 0.3991 |
| Correlation (logprob vs overall quality) | 0.1242 |

### Hallucination Detection AUROC

AUROC measures whether uncertainty scores can distinguish hallucinated
from non-hallucinated outputs (Kuhn et al., 2023; Xiong et al., 2024).
AUROC = 0.5 is random; AUROC = 1.0 is perfect detection.

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL (-sum logprobs) | 0.8531 | Useful |
| Length-Norm NLL (-mean logprob) | 0.8158 | Useful |
| Perplexity (exp(-mean logprob)) | 0.8158 | Useful |
| Hallucination threshold | <= 3/5 |  |
| N correct / N incorrect | 38 / 12 |  |

### Calibration (ECE)

Expected Calibration Error measures the gap between token-level
confidence (exp(mean_logprob)) and output quality (Guo et al., 2017).
**Caveat:** Token-level confidence reflects how probable the model
considers its chosen tokens, NOT the probability of factual correctness.
A model can produce high-probability hallucinations. ECE here quantifies
HOW overconfident models are, not whether they are calibrated classifiers.

| ECE | 0.2144 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0141 | 1.01 | 0.0178 | 50 |
| HPI | -0.0446 | 1.05 | 0.0350 | 50 |
| Past Medical Hx | -0.0083 | 1.01 | 0.0119 | 49 |
| Medications | -0.0044 | 1.00 | 0.0025 | 50 |
| Allergies | -0.0156 | 1.02 | 0.0220 | 49 |
| Examination | -0.0193 | 1.02 | 0.0207 | 49 |
| Assessment | -0.0301 | 1.03 | 0.0285 | 49 |
| Plan | -0.0330 | 1.03 | 0.0324 | 49 |
| Safety Netting | -0.0237 | 1.02 | 0.0323 | 49 |

## Llama-3.2 (3B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0200 |
| Std logprob | 0.0064 |
| Mean perplexity | 1.02 |
| Mean frac low-conf tokens | 0.0298 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | -0.0787 |
| Correlation (logprob vs overall quality) | -0.0454 |

### Hallucination Detection AUROC

AUROC measures whether uncertainty scores can distinguish hallucinated
from non-hallucinated outputs (Kuhn et al., 2023; Xiong et al., 2024).
AUROC = 0.5 is random; AUROC = 1.0 is perfect detection.

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL (-sum logprobs) | 0.5503 | Near-random |
| Length-Norm NLL (-mean logprob) | 0.5104 | Near-random |
| Perplexity (exp(-mean logprob)) | 0.5104 | Near-random |
| Hallucination threshold | <= 3/5 |  |
| N correct / N incorrect | 32 / 18 |  |

### Calibration (ECE)

Expected Calibration Error measures the gap between token-level
confidence (exp(mean_logprob)) and output quality (Guo et al., 2017).
**Caveat:** Token-level confidence reflects how probable the model
considers its chosen tokens, NOT the probability of factual correctness.
A model can produce high-probability hallucinations. ECE here quantifies
HOW overconfident models are, not whether they are calibrated classifiers.

| ECE | 0.3402 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0083 | 1.01 | 0.0100 | 50 |
| HPI | -0.0406 | 1.04 | 0.0398 | 50 |
| Past Medical Hx | -0.0056 | 1.01 | 0.0072 | 50 |
| Medications | -0.0007 | 1.00 | 0.0028 | 50 |
| Allergies | -0.0079 | 1.01 | 0.0102 | 50 |
| Examination | -0.0122 | 1.01 | 0.0204 | 50 |
| Assessment | -0.0260 | 1.03 | 0.0329 | 50 |
| Plan | -0.0252 | 1.03 | 0.0351 | 50 |
| Safety Netting | -0.0136 | 1.01 | 0.0229 | 50 |

## Llama-3.2 (1B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0229 |
| Std logprob | 0.0077 |
| Mean perplexity | 1.02 |
| Mean frac low-conf tokens | 0.0338 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | -0.1831 |
| Correlation (logprob vs overall quality) | -0.2069 |

### Hallucination Detection AUROC

AUROC measures whether uncertainty scores can distinguish hallucinated
from non-hallucinated outputs (Kuhn et al., 2023; Xiong et al., 2024).
AUROC = 0.5 is random; AUROC = 1.0 is perfect detection.

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL (-sum logprobs) | 0.4724 | Near-random |
| Length-Norm NLL (-mean logprob) | 0.4076 | Near-random |
| Perplexity (exp(-mean logprob)) | 0.4076 | Near-random |
| Hallucination threshold | <= 3/5 |  |
| N correct / N incorrect | 15 / 35 |  |

### Calibration (ECE)

Expected Calibration Error measures the gap between token-level
confidence (exp(mean_logprob)) and output quality (Guo et al., 2017).
**Caveat:** Token-level confidence reflects how probable the model
considers its chosen tokens, NOT the probability of factual correctness.
A model can produce high-probability hallucinations. ECE here quantifies
HOW overconfident models are, not whether they are calibrated classifiers.

| ECE | 0.6774 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0044 | 1.00 | 0.0058 | 50 |
| HPI | -0.0432 | 1.04 | 0.0415 | 50 |
| Past Medical Hx | -0.0046 | 1.00 | 0.0076 | 50 |
| Medications | -0.0063 | 1.01 | 0.0070 | 50 |
| Allergies | -0.0093 | 1.01 | 0.0178 | 50 |
| Examination | -0.0153 | 1.02 | 0.0230 | 50 |
| Assessment | -0.0335 | 1.03 | 0.0433 | 50 |
| Plan | -0.0273 | 1.03 | 0.0330 | 50 |
| Safety Netting | -0.0167 | 1.02 | 0.0291 | 50 |
