# Uncertainty Quantification Analysis via Log Probabilities

## Scientific Background

Token-level log probabilities provide a measure of model confidence for each generated token. Lower logprobs indicate higher uncertainty, which has been shown to correlate with factual errors and hallucinations (Kadavath et al., 2022; Kuhn et al., 2023).

## Phi-3.5 (3.8B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0273 |
| Std logprob | 0.0066 |
| Mean perplexity | 1.0277 |
| Mean frac low-conf tokens | 0.0001 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | 0.0764 |
| Correlation (logprob vs overall quality) | 0.0626 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.6216 | Useful |
| Length-Norm NLL | 0.6008 | Useful |
| Perplexity | 0.6008 | Useful |
| N correct / N incorrect | 37 / 13 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.2331 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0150 | 1.0151 | 0.0000 | 50 |
| HPI | -0.0496 | 1.0509 | 0.0007 | 50 |
| Past Medical Hx | -0.0101 | 1.0102 | 0.0000 | 49 |
| Medications | -0.0027 | 1.0027 | 0.0000 | 50 |
| Allergies | -0.0187 | 1.0189 | 0.0000 | 49 |
| Examination | -0.0203 | 1.0205 | 0.0000 | 49 |
| Assessment | -0.0281 | 1.0285 | 0.0000 | 48 |
| Plan | -0.0350 | 1.0356 | 0.0000 | 48 |
| Safety Netting | -0.0228 | 1.0231 | 0.0000 | 48 |

## Llama-3.2 (3B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0193 |
| Std logprob | 0.0091 |
| Mean perplexity | 1.0195 |
| Mean frac low-conf tokens | 0.0007 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | -0.2788 |
| Correlation (logprob vs overall quality) | -0.1741 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.3148 | Near-random |
| Length-Norm NLL | 0.2689 | Near-random |
| Perplexity | 0.2689 | Near-random |
| N correct / N incorrect | 27 / 23 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.4410 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0095 | 1.0096 | 0.0000 | 50 |
| HPI | -0.0371 | 1.0378 | 0.0030 | 50 |
| Past Medical Hx | -0.0062 | 1.0063 | 0.0000 | 50 |
| Medications | -0.0014 | 1.0014 | 0.0000 | 50 |
| Allergies | -0.0074 | 1.0074 | 0.0000 | 50 |
| Examination | -0.0125 | 1.0126 | 0.0000 | 50 |
| Assessment | -0.0309 | 1.0314 | 0.0000 | 50 |
| Plan | -0.0223 | 1.0226 | 0.0003 | 50 |
| Safety Netting | -0.0143 | 1.0144 | 0.0000 | 50 |

## Llama-3.2 (1B)

### Overall Confidence

| Metric | Value |
|---|---|
| Mean logprob | -0.0260 |
| Std logprob | 0.0068 |
| Mean perplexity | 1.0263 |
| Mean frac low-conf tokens | 0.0009 |
| Num samples | 50 |
| Correlation (logprob vs hallucination) | -0.0217 |
| Correlation (logprob vs overall quality) | -0.0447 |

### Hallucination Detection AUROC

AUROC = 0.5 → random; AUROC = 1.0 → perfect detection. Binarisation: judge_hallucination > 3 = correct (Kuhn et al., 2023; Xiong et al., 2024).

| Uncertainty Measure | AUROC | Interpretation |
|---|---|---|
| Sequence NLL | 0.5092 | Near-random |
| Length-Norm NLL | 0.4853 | Near-random |
| Perplexity | 0.4853 | Near-random |
| N correct / N incorrect | 16 / 34 | |

### Calibration (ECE)

**Caveat:** Token-level confidence = exp(mean_logprob) reflects how probable the model considers its own token choices, NOT the probability of factual correctness. A model can produce high-probability hallucinations. ECE quantifies the overconfidence gap (Guo et al., 2017).

| ECE | 0.6544 |

### Per-Section Confidence

| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |
|---|---|---|---|---|
| Chief Complaint | -0.0052 | 1.0052 | 0.0000 | 50 |
| HPI | -0.0486 | 1.0498 | 0.0032 | 50 |
| Past Medical Hx | -0.0104 | 1.0105 | 0.0000 | 50 |
| Medications | -0.0114 | 1.0115 | 0.0000 | 50 |
| Allergies | -0.0134 | 1.0135 | 0.0000 | 50 |
| Examination | -0.0149 | 1.0150 | 0.0000 | 50 |
| Assessment | -0.0390 | 1.0397 | 0.0009 | 50 |
| Plan | -0.0289 | 1.0293 | 0.0005 | 50 |
| Safety Netting | -0.0188 | 1.0190 | 0.0002 | 50 |
