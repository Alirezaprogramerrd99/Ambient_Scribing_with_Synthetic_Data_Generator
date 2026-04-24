# Table UQ: Uncertainty Quantification Metrics

Config: `ft_rag` (fine-tuned + Dense+Rerank RAG).  50-sample ACI-Bench 2023 test set.
AUROC binarisation: `judge_hallucination > 3` = correct (Kuhn et al. 2023; Xiong et al. 2024 Table 5).
ECE: token confidence = `exp(mean_logprob)` vs binary output correctness (Guo et al. 2017; Xiong et al. 2024 §3.2).
↑ = higher is better; ↓ = lower is better.
AUROC, ECE and r: overall-level only — per-section judge labels unavailable.

| Model | Scope | Mean LogProb ↑ | Std LogProb | Perplexity ↓ | Frac < −2 nats ↓ | AUROC (LN-NLL) ↑ | AUROC (Seq-NLL) ↑ | ECE ↓ | r(lp, halluc) | r(lp, overall) |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5 (3.8B) | **Overall** | -0.0280 | 0.0066 | 1.0284 | 0.0001 | 0.5320 | 0.5460 | 0.2824 | 0.1478 | 0.1637 |
|  | Chief Complaint (n=100) | -0.0163 | 0.0227 | 1.0164 | 0.0000 | — | — | — | — | — |
|  | HPI (n=100) | -0.0504 | 0.0203 | 1.0516 | 0.0007 | — | — | — | — | — |
|  | Past Medical Hx (n=100) | -0.0072 | 0.0233 | 1.0072 | 0.0000 | — | — | — | — | — |
|  | Medications (n=100) | -0.0047 | 0.0149 | 1.0047 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=100) | -0.0189 | 0.0136 | 1.0191 | 0.0000 | — | — | — | — | — |
|  | Examination (n=100) | -0.0189 | 0.0174 | 1.0190 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=100) | -0.0307 | 0.0269 | 1.0312 | 0.0000 | — | — | — | — | — |
|  | Plan (n=100) | -0.0365 | 0.0254 | 1.0371 | 0.0000 | — | — | — | — | — |
|  | Safety Netting (n=100) | -0.0232 | 0.0112 | 1.0235 | 0.0000 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (3B) | **Overall** | -0.0193 | 0.0064 | 1.0195 | 0.0005 | 0.4541 | 0.4743 | 0.3309 | -0.1227 | -0.0857 |
|  | Chief Complaint (n=100) | -0.0072 | 0.0179 | 1.0073 | 0.0000 | — | — | — | — | — |
|  | HPI (n=100) | -0.0356 | 0.0189 | 1.0362 | 0.0019 | — | — | — | — | — |
|  | Past Medical Hx (n=100) | -0.0101 | 0.0291 | 1.0101 | 0.0000 | — | — | — | — | — |
|  | Medications (n=100) | -0.0015 | 0.0098 | 1.0015 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=100) | -0.0067 | 0.0120 | 1.0067 | 0.0000 | — | — | — | — | — |
|  | Examination (n=100) | -0.0138 | 0.0165 | 1.0139 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=100) | -0.0295 | 0.0334 | 1.0300 | 0.0009 | — | — | — | — | — |
|  | Plan (n=100) | -0.0223 | 0.0228 | 1.0225 | 0.0000 | — | — | — | — | — |
|  | Safety Netting (n=100) | -0.0153 | 0.0114 | 1.0154 | 0.0000 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (1B) | **Overall** | -0.0235 | 0.0072 | 1.0238 | 0.0009 | 0.5087 | 0.5285 | 0.6568 | -0.0212 | -0.0027 |
|  | Chief Complaint (n=100) | -0.0042 | 0.0167 | 1.0042 | 0.0000 | — | — | — | — | — |
|  | HPI (n=100) | -0.0475 | 0.0226 | 1.0486 | 0.0037 | — | — | — | — | — |
|  | Past Medical Hx (n=100) | -0.0044 | 0.0163 | 1.0044 | 0.0000 | — | — | — | — | — |
|  | Medications (n=99) | -0.0048 | 0.0262 | 1.0048 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=99) | -0.0084 | 0.0131 | 1.0084 | 0.0000 | — | — | — | — | — |
|  | Examination (n=100) | -0.0158 | 0.0200 | 1.0159 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=100) | -0.0345 | 0.0374 | 1.0351 | 0.0011 | — | — | — | — | — |
|  | Plan (n=100) | -0.0320 | 0.0256 | 1.0325 | 0.0000 | — | — | — | — | — |
|  | Safety Netting (n=100) | -0.0156 | 0.0117 | 1.0157 | 0.0001 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|

## Sample Prevalence (AUROC context)

| Model | N Correct (halluc > 3) | N Hallucinated | N Total | % Hallucinated |
|---|---|---|---|---|
| Phi-3.5 (3.8B) | 69 | 31 | 100 | 31.0% |
| Llama-3.2 (3B) | 65 | 35 | 100 | 35.0% |
| Llama-3.2 (1B) | 32 | 68 | 100 | 68.0% |

## Column Definitions

| Column | Definition | Reference |
|---|---|---|
| Mean LogProb ↑ | Mean token log-prob; higher (less negative) = more confident | Kadavath et al. (2022); Xiong et al. (2024) |
| Std LogProb | Cross-sample std of mean_logprob; variability of confidence across encounters | — |
| Perplexity ↓ | exp(−mean_logprob); lower = more confident | Kuhn et al. (2023) §3.3 |
| Frac < −2 nats ↓ | Fraction of tokens with logprob < −2.0 (below 13.5% prob) | Xiong et al. (2024) |
| AUROC (LN-NLL) ↑ | AUROC with length-norm NLL as uncertainty score; random = 0.5 | Kuhn et al. (2023) §6; Xiong et al. (2024) Table 5 |
| AUROC (Seq-NLL) ↑ | AUROC with raw sequence NLL; biased toward longer outputs | Kuhn et al. (2023) |
| ECE ↓ | Expected Calibration Error; token confidence vs output correctness gap | Guo et al. (2017) |
| r(lp, halluc) | Pearson r(mean_logprob, judge_hallucination 1–5); overall only | — |
| r(lp, overall) | Pearson r(mean_logprob, judge_overall 1–5); overall only | — |