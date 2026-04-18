# Table UQ: Uncertainty Quantification Metrics

Config: `ft_rag` (fine-tuned + Dense+Rerank RAG).  50-sample ACI-Bench 2023 test set.
AUROC binarisation: `judge_hallucination > 3` = correct (Kuhn et al. 2023; Xiong et al. 2024 Table 5).
ECE: token confidence = `exp(mean_logprob)` vs binary output correctness (Guo et al. 2017; Xiong et al. 2024 §3.2).
↑ = higher is better; ↓ = lower is better.
AUROC, ECE and r: overall-level only — per-section judge labels unavailable.

| Model | Scope | Mean LogProb ↑ | Std LogProb | Perplexity ↓ | Frac < −2 nats ↓ | AUROC (LN-NLL) ↑ | AUROC (Seq-NLL) ↑ | ECE ↓ | r(lp, halluc) | r(lp, overall) |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5 (3.8B) | **Overall** | -0.0273 | 0.0066 | 1.0277 | 0.0001 | 0.6008 | 0.6216 | 0.2331 | 0.0764 | 0.0626 |
|  | Chief Complaint (n=50) | -0.0150 | 0.0203 | 1.0151 | 0.0000 | — | — | — | — | — |
|  | HPI (n=50) | -0.0496 | 0.0183 | 1.0509 | 0.0007 | — | — | — | — | — |
|  | Past Medical Hx (n=49) | -0.0101 | 0.0290 | 1.0102 | 0.0000 | — | — | — | — | — |
|  | Medications (n=50) | -0.0027 | 0.0092 | 1.0027 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=49) | -0.0187 | 0.0115 | 1.0189 | 0.0000 | — | — | — | — | — |
|  | Examination (n=49) | -0.0203 | 0.0186 | 1.0205 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=48) | -0.0281 | 0.0258 | 1.0285 | 0.0000 | — | — | — | — | — |
|  | Plan (n=48) | -0.0350 | 0.0200 | 1.0356 | 0.0000 | — | — | — | — | — |
|  | Safety Netting (n=48) | -0.0228 | 0.0101 | 1.0231 | 0.0000 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (3B) | **Overall** | -0.0193 | 0.0091 | 1.0195 | 0.0007 | 0.2689 | 0.3148 | 0.4410 | -0.2788 | -0.1741 |
|  | Chief Complaint (n=50) | -0.0095 | 0.0229 | 1.0096 | 0.0000 | — | — | — | — | — |
|  | HPI (n=50) | -0.0371 | 0.0211 | 1.0378 | 0.0030 | — | — | — | — | — |
|  | Past Medical Hx (n=50) | -0.0062 | 0.0202 | 1.0063 | 0.0000 | — | — | — | — | — |
|  | Medications (n=50) | -0.0014 | 0.0071 | 1.0014 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=50) | -0.0074 | 0.0128 | 1.0074 | 0.0000 | — | — | — | — | — |
|  | Examination (n=50) | -0.0125 | 0.0194 | 1.0126 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=50) | -0.0309 | 0.0351 | 1.0314 | 0.0000 | — | — | — | — | — |
|  | Plan (n=50) | -0.0223 | 0.0240 | 1.0226 | 0.0003 | — | — | — | — | — |
|  | Safety Netting (n=50) | -0.0143 | 0.0133 | 1.0144 | 0.0000 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (1B) | **Overall** | -0.0260 | 0.0068 | 1.0263 | 0.0009 | 0.4853 | 0.5092 | 0.6544 | -0.0217 | -0.0447 |
|  | Chief Complaint (n=50) | -0.0052 | 0.0169 | 1.0052 | 0.0000 | — | — | — | — | — |
|  | HPI (n=50) | -0.0486 | 0.0202 | 1.0498 | 0.0032 | — | — | — | — | — |
|  | Past Medical Hx (n=50) | -0.0104 | 0.0349 | 1.0105 | 0.0000 | — | — | — | — | — |
|  | Medications (n=50) | -0.0114 | 0.0266 | 1.0115 | 0.0000 | — | — | — | — | — |
|  | Allergies (n=50) | -0.0134 | 0.0181 | 1.0135 | 0.0000 | — | — | — | — | — |
|  | Examination (n=50) | -0.0149 | 0.0218 | 1.0150 | 0.0000 | — | — | — | — | — |
|  | Assessment (n=50) | -0.0390 | 0.0406 | 1.0397 | 0.0009 | — | — | — | — | — |
|  | Plan (n=50) | -0.0289 | 0.0297 | 1.0293 | 0.0005 | — | — | — | — | — |
|  | Safety Netting (n=50) | -0.0188 | 0.0128 | 1.0190 | 0.0002 | — | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|---|

## Sample Prevalence (AUROC context)

| Model | N Correct (halluc > 3) | N Hallucinated | N Total | % Hallucinated |
|---|---|---|---|---|
| Phi-3.5 (3.8B) | 37 | 13 | 50 | 26.0% |
| Llama-3.2 (3B) | 27 | 23 | 50 | 46.0% |
| Llama-3.2 (1B) | 16 | 34 | 50 | 68.0% |

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