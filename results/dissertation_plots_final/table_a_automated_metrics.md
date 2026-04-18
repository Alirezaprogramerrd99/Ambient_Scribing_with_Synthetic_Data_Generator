# Table A: Automated Evaluation Metrics

All metrics computed on 50-sample test set. Per-section values show ROUGE-L.
MEDCON uses QuickUMLS + UMLS 2025AB where available, regex fallback otherwise.

| Model | Config | RAG | Scope | ROUGE-1 | ROUGE-L | BLEU-4 | MEDCON-F1 | BERTScore | Perplexity |
|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5 (3.8B) | FT | None | **Overall** | 0.721 | 0.622 | 0.546 | 0.767 | 0.940 | — |
| | | | Chief Complaint | 0.882 | 0.870 | — | — | — | — |
| | | | HPI | 0.590 | 0.512 | — | — | — | — |
| | | | Past Medical Hx | 0.833 | 0.833 | — | — | — | — |
| | | | Medications | 0.891 | 0.887 | — | — | — | — |
| | | | Allergies | 0.791 | 0.790 | — | — | — | — |
| | | | Examination | 0.733 | 0.698 | — | — | — | — |
| | | | Assessment | 0.573 | 0.508 | — | — | — | — |
| | | | Plan | 0.565 | 0.503 | — | — | — | — |
| | | | Safety Netting | 0.676 | 0.589 | — | — | — | — |
|  | FT+RAG | Dense+Rerank | **Overall** | 0.720 | 0.625 | 0.546 | 0.758 | 0.939 | — |
| | | | Chief Complaint | 0.862 | 0.848 | — | — | — | — |
| | | | HPI | 0.601 | 0.525 | — | — | — | — |
| | | | Past Medical Hx | 0.829 | 0.829 | — | — | — | — |
| | | | Medications | 0.877 | 0.877 | — | — | — | — |
| | | | Allergies | 0.763 | 0.763 | — | — | — | — |
| | | | Examination | 0.713 | 0.681 | — | — | — | — |
| | | | Assessment | 0.574 | 0.512 | — | — | — | — |
| | | | Plan | 0.568 | 0.503 | — | — | — | — |
| | | | Safety Netting | 0.683 | 0.601 | — | — | — | — |
|  | Base | None | **Overall** | 0.521 | 0.369 | 0.115 | 0.629 | 0.870 | — |
| | | | Chief Complaint | 0.465 | 0.440 | — | — | — | — |
| | | | HPI | 0.397 | 0.286 | — | — | — | — |
| | | | Past Medical Hx | 0.544 | 0.544 | — | — | — | — |
| | | | Medications | 0.532 | 0.530 | — | — | — | — |
| | | | Allergies | 0.266 | 0.260 | — | — | — | — |
| | | | Examination | 0.524 | 0.468 | — | — | — | — |
| | | | Assessment | 0.362 | 0.302 | — | — | — | — |
| | | | Plan | 0.451 | 0.377 | — | — | — | — |
| | | | Safety Netting | 0.233 | 0.162 | — | — | — | — |
|  | Base+RAG | Dense+Rerank | **Overall** | 0.519 | 0.367 | 0.115 | 0.632 | 0.870 | — |
| | | | Chief Complaint | 0.472 | 0.449 | — | — | — | — |
| | | | HPI | 0.395 | 0.290 | — | — | — | — |
| | | | Past Medical Hx | 0.502 | 0.501 | — | — | — | — |
| | | | Medications | 0.502 | 0.500 | — | — | — | — |
| | | | Allergies | 0.264 | 0.257 | — | — | — | — |
| | | | Examination | 0.527 | 0.471 | — | — | — | — |
| | | | Assessment | 0.337 | 0.286 | — | — | — | — |
| | | | Plan | 0.443 | 0.367 | — | — | — | — |
| | | | Safety Netting | 0.241 | 0.167 | — | — | — | — |
|  | Teacher | None (API) | **Overall** | 0.558 | 0.416 | 0.284 | 0.657 | 0.875 | — |
| | | | Chief Complaint | 0.565 | 0.558 | — | — | — | — |
| | | | HPI | 0.505 | 0.396 | — | — | — | — |
| | | | Past Medical Hx | 0.603 | 0.603 | — | — | — | — |
| | | | Medications | 0.512 | 0.510 | — | — | — | — |
| | | | Allergies | 0.284 | 0.284 | — | — | — | — |
| | | | Examination | 0.538 | 0.486 | — | — | — | — |
| | | | Assessment | 0.421 | 0.351 | — | — | — | — |
| | | | Plan | 0.491 | 0.421 | — | — | — | — |
| | | | Safety Netting | 0.265 | 0.193 | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (3B) | FT | None | **Overall** | 0.728 | 0.629 | 0.561 | 0.788 | 0.940 | — |
| | | | Chief Complaint | 0.863 | 0.843 | — | — | — | — |
| | | | HPI | 0.588 | 0.490 | — | — | — | — |
| | | | Past Medical Hx | 0.851 | 0.848 | — | — | — | — |
| | | | Medications | 0.895 | 0.895 | — | — | — | — |
| | | | Allergies | 0.844 | 0.844 | — | — | — | — |
| | | | Examination | 0.732 | 0.694 | — | — | — | — |
| | | | Assessment | 0.563 | 0.493 | — | — | — | — |
| | | | Plan | 0.563 | 0.492 | — | — | — | — |
| | | | Safety Netting | 0.696 | 0.617 | — | — | — | — |
|  | FT+RAG | Dense+Rerank | **Overall** | 0.724 | 0.625 | 0.551 | 0.780 | 0.939 | — |
| | | | Chief Complaint | 0.862 | 0.840 | — | — | — | — |
| | | | HPI | 0.600 | 0.504 | — | — | — | — |
| | | | Past Medical Hx | 0.855 | 0.855 | — | — | — | — |
| | | | Medications | 0.871 | 0.871 | — | — | — | — |
| | | | Allergies | 0.842 | 0.841 | — | — | — | — |
| | | | Examination | 0.734 | 0.696 | — | — | — | — |
| | | | Assessment | 0.539 | 0.473 | — | — | — | — |
| | | | Plan | 0.557 | 0.486 | — | — | — | — |
| | | | Safety Netting | 0.691 | 0.614 | — | — | — | — |
|  | Base | None | **Overall** | 0.498 | 0.332 | 0.293 | 0.600 | 0.866 | — |
| | | | Chief Complaint | 0.348 | 0.313 | — | — | — | — |
| | | | HPI | 0.450 | 0.328 | — | — | — | — |
| | | | Past Medical Hx | 0.409 | 0.407 | — | — | — | — |
| | | | Medications | 0.568 | 0.566 | — | — | — | — |
| | | | Allergies | 0.156 | 0.147 | — | — | — | — |
| | | | Examination | 0.503 | 0.439 | — | — | — | — |
| | | | Assessment | 0.257 | 0.211 | — | — | — | — |
| | | | Plan | 0.406 | 0.331 | — | — | — | — |
| | | | Safety Netting | 0.271 | 0.195 | — | — | — | — |
|  | Base+RAG | Dense+Rerank | **Overall** | 0.493 | 0.327 | 0.293 | 0.600 | 0.866 | — |
| | | | Chief Complaint | 0.345 | 0.313 | — | — | — | — |
| | | | HPI | 0.451 | 0.332 | — | — | — | — |
| | | | Past Medical Hx | 0.393 | 0.391 | — | — | — | — |
| | | | Medications | 0.541 | 0.536 | — | — | — | — |
| | | | Allergies | 0.132 | 0.128 | — | — | — | — |
| | | | Examination | 0.488 | 0.425 | — | — | — | — |
| | | | Assessment | 0.248 | 0.200 | — | — | — | — |
| | | | Plan | 0.405 | 0.331 | — | — | — | — |
| | | | Safety Netting | 0.263 | 0.187 | — | — | — | — |
|  | Teacher | None (API) | **Overall** | 0.558 | 0.416 | 0.284 | 0.657 | 0.875 | — |
| | | | Chief Complaint | 0.565 | 0.558 | — | — | — | — |
| | | | HPI | 0.505 | 0.396 | — | — | — | — |
| | | | Past Medical Hx | 0.603 | 0.603 | — | — | — | — |
| | | | Medications | 0.512 | 0.510 | — | — | — | — |
| | | | Allergies | 0.284 | 0.284 | — | — | — | — |
| | | | Examination | 0.538 | 0.486 | — | — | — | — |
| | | | Assessment | 0.421 | 0.351 | — | — | — | — |
| | | | Plan | 0.491 | 0.421 | — | — | — | — |
| | | | Safety Netting | 0.265 | 0.193 | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2 (1B) | FT | None | **Overall** | 0.697 | 0.589 | 0.528 | 0.745 | 0.935 | — |
| | | | Chief Complaint | 0.826 | 0.808 | — | — | — | — |
| | | | HPI | 0.524 | 0.434 | — | — | — | — |
| | | | Past Medical Hx | 0.742 | 0.742 | — | — | — | — |
| | | | Medications | 0.859 | 0.849 | — | — | — | — |
| | | | Allergies | 0.815 | 0.813 | — | — | — | — |
| | | | Examination | 0.708 | 0.678 | — | — | — | — |
| | | | Assessment | 0.540 | 0.477 | — | — | — | — |
| | | | Plan | 0.528 | 0.463 | — | — | — | — |
| | | | Safety Netting | 0.659 | 0.576 | — | — | — | — |
|  | FT+RAG | Dense+Rerank | **Overall** | 0.699 | 0.591 | 0.527 | 0.756 | 0.935 | — |
| | | | Chief Complaint | 0.821 | 0.802 | — | — | — | — |
| | | | HPI | 0.539 | 0.446 | — | — | — | — |
| | | | Past Medical Hx | 0.736 | 0.736 | — | — | — | — |
| | | | Medications | 0.839 | 0.835 | — | — | — | — |
| | | | Allergies | 0.819 | 0.818 | — | — | — | — |
| | | | Examination | 0.718 | 0.685 | — | — | — | — |
| | | | Assessment | 0.557 | 0.489 | — | — | — | — |
| | | | Plan | 0.545 | 0.474 | — | — | — | — |
| | | | Safety Netting | 0.653 | 0.572 | — | — | — | — |
|  | Base | None | **Overall** | 0.457 | 0.283 | 0.196 | 0.585 | 0.866 | — |
| | | | Chief Complaint | 0.271 | 0.237 | — | — | — | — |
| | | | HPI | 0.414 | 0.328 | — | — | — | — |
| | | | Past Medical Hx | 0.365 | 0.364 | — | — | — | — |
| | | | Medications | 0.261 | 0.254 | — | — | — | — |
| | | | Allergies | 0.238 | 0.229 | — | — | — | — |
| | | | Examination | 0.380 | 0.311 | — | — | — | — |
| | | | Assessment | 0.200 | 0.149 | — | — | — | — |
| | | | Plan | 0.306 | 0.230 | — | — | — | — |
| | | | Safety Netting | 0.247 | 0.151 | — | — | — | — |
|  | Base+RAG | Dense+Rerank | **Overall** | 0.464 | 0.287 | 0.200 | 0.584 | 0.867 | — |
| | | | Chief Complaint | 0.274 | 0.239 | — | — | — | — |
| | | | HPI | 0.410 | 0.321 | — | — | — | — |
| | | | Past Medical Hx | 0.358 | 0.355 | — | — | — | — |
| | | | Medications | 0.251 | 0.246 | — | — | — | — |
| | | | Allergies | 0.239 | 0.230 | — | — | — | — |
| | | | Examination | 0.381 | 0.319 | — | — | — | — |
| | | | Assessment | 0.217 | 0.171 | — | — | — | — |
| | | | Plan | 0.306 | 0.236 | — | — | — | — |
| | | | Safety Netting | 0.250 | 0.147 | — | — | — | — |
|  | Teacher | None (API) | **Overall** | 0.558 | 0.416 | 0.284 | 0.657 | 0.875 | — |
| | | | Chief Complaint | 0.565 | 0.558 | — | — | — | — |
| | | | HPI | 0.505 | 0.396 | — | — | — | — |
| | | | Past Medical Hx | 0.603 | 0.603 | — | — | — | — |
| | | | Medications | 0.512 | 0.510 | — | — | — | — |
| | | | Allergies | 0.284 | 0.284 | — | — | — | — |
| | | | Examination | 0.538 | 0.486 | — | — | — | — |
| | | | Assessment | 0.421 | 0.351 | — | — | — | — |
| | | | Plan | 0.491 | 0.421 | — | — | — | — |
| | | | Safety Netting | 0.265 | 0.193 | — | — | — | — |
|---|---|---|---|---|---|---|---|---|---|