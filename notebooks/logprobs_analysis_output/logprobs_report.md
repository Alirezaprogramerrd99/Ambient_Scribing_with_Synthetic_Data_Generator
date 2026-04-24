# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence for each generated token. Lower logprobs indicate higher uncertainty, which has been shown to correlate with factual errors and hallucinations (Kadavath et al., 2022; Kuhn et al., 2023).

## Phi-3.5 (3.8B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0280 |
| Std logprob | 0.0066 |
| Mean perplexity | 1.0284 |
| Mean frac low-conf tokens | 0.0001 |
| Num samples | 100 |
| Correlation (logprob vs hallucination) | 0.1478 |
| Correlation (logprob vs overall quality) | 0.1637 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.5460 | Near-random |
| Length-Norm NLL | 0.5320 | Near-random |
| Perplexity | 0.5320 | Near-random |
| N correct / N incorrect | 69 / 31 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.2824 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0163 | 1.0164 | 0.0000 | 100 |
| HPI | -0.0504 | 1.0516 | 0.0007 | 100 |
| Past Medical Hx | -0.0072 | 1.0072 | 0.0000 | 100 |
| Medications | -0.0047 | 1.0047 | 0.0000 | 100 |
| Allergies | -0.0189 | 1.0191 | 0.0000 | 100 |
| Examination | -0.0189 | 1.0190 | 0.0000 | 100 |
| Assessment | -0.0307 | 1.0312 | 0.0000 | 100 |
| Plan | -0.0365 | 1.0371 | 0.0000 | 100 |
| Safety Netting | -0.0232 | 1.0235 | 0.0000 | 100 |

## Llama-3.2 (3B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0193 |
| Std logprob | 0.0064 |
| Mean perplexity | 1.0195 |
| Mean frac low-conf tokens | 0.0005 |
| Num samples | 100 |
| Correlation (logprob vs hallucination) | -0.1227 |
| Correlation (logprob vs overall quality) | -0.0857 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.4743 | Near-random |
| Length-Norm NLL | 0.4541 | Near-random |
| Perplexity | 0.4541 | Near-random |
| N correct / N incorrect | 65 / 35 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.3309 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0072 | 1.0073 | 0.0000 | 100 |
| HPI | -0.0356 | 1.0362 | 0.0019 | 100 |
| Past Medical Hx | -0.0101 | 1.0101 | 0.0000 | 100 |
| Medications | -0.0015 | 1.0015 | 0.0000 | 100 |
| Allergies | -0.0067 | 1.0067 | 0.0000 | 100 |
| Examination | -0.0138 | 1.0139 | 0.0000 | 100 |
| Assessment | -0.0295 | 1.0300 | 0.0009 | 100 |
| Plan | -0.0223 | 1.0225 | 0.0000 | 100 |
| Safety Netting | -0.0153 | 1.0154 | 0.0000 | 100 |

## Llama-3.2 (1B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0235 |
| Std logprob | 0.0072 |
| Mean perplexity | 1.0238 |
| Mean frac low-conf tokens | 0.0009 |
| Num samples | 100 |
| Correlation (logprob vs hallucination) | -0.0212 |
| Correlation (logprob vs overall quality) | -0.0027 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.5285 | Near-random |
| Length-Norm NLL | 0.5087 | Near-random |
| Perplexity | 0.5087 | Near-random |
| N correct / N incorrect | 32 / 68 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.6568 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0042 | 1.0042 | 0.0000 | 100 |
| HPI | -0.0475 | 1.0486 | 0.0037 | 100 |
| Past Medical Hx | -0.0044 | 1.0044 | 0.0000 | 100 |
| Medications | -0.0048 | 1.0048 | 0.0000 | 99 |
| Allergies | -0.0084 | 1.0084 | 0.0000 | 99 |
| Examination | -0.0158 | 1.0159 | 0.0000 | 100 |
| Assessment | -0.0345 | 1.0351 | 0.0011 | 100 |
| Plan | -0.0320 | 1.0325 | 0.0000 | 100 |
| Safety Netting | -0.0156 | 1.0157 | 0.0001 | 100 |
