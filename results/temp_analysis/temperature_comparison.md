# Temperature Experiment Results
**Generated:** 2026-03-24 15:05

## FT+RAG Configuration — Effect of Generation Temperature

Scientific references:
- Woo et al. (2025): tested temp 0 vs 1, found temp 0 slightly better
- Renze & Guven (2024): no significant difference for temp 0-1
- Chang et al. (2023): lower temp better for QA with attribution

### Phi-3.5 (3.8B)

| Metric | T=0.0 | T=0.1 | T=0.3 | T=0.7 | T=1.0 |
|---|---|---|---|---|---|
| ROUGE-L | 0.5373 | 0.5378 | 0.5267 | 0.5298 | 0.5168 |
| Judge Overall | 4.1015 | 4.1280 | 4.2600 | 4.2450 | 4.2595 |
| Accuracy | 3.8500 | 3.9000 | 4.0500 | 4.2000 | 4.0500 |
| Hallucination | 3.3500 | 3.4000 | 3.8000 | 3.8500 | 3.7000 |
| Safety | 4.6500 | 4.5500 | 4.6500 | 4.6000 | 4.6500 |
| Coherence | 4.6500 | 4.7500 | 4.6500 | 4.5500 | 4.6500 |
| Avg Time (s) | 12.4 | 13.0 | 12.7 | 12.6 | 14.0 |
| Critical Errors | 1 | 1 | 1 | 0 | 1 |

### Llama-3.2 (3B)

| Metric | T=0.0 | T=0.1 | T=0.3 | T=0.7 | T=1.0 |
|---|---|---|---|---|---|
| ROUGE-L | 0.6186 | 0.6140 | 0.6111 | 0.5826 | 0.5878 |
| Judge Overall | 4.2875 | 4.3165 | 4.3000 | 4.2550 | 4.0930 |
| Accuracy | 4.0500 | 4.0000 | 4.0000 | 4.1000 | 3.7500 |
| Hallucination | 3.6000 | 3.6500 | 3.7500 | 3.6500 | 3.2000 |
| Safety | 4.7500 | 4.8000 | 4.6000 | 4.6500 | 4.5000 |
| Coherence | 5.0000 | 4.9000 | 4.8500 | 4.9500 | 4.7500 |
| Avg Time (s) | 9.6 | 9.9 | 10.8 | 11.2 | 11.1 |
| Critical Errors | 1 | 0 | 0 | 0 | 3 |

### Llama-3.2 (1B)

| Metric | T=0.0 | T=0.1 | T=0.3 | T=0.7 | T=1.0 |
|---|---|---|---|---|---|
| ROUGE-L | 0.5896 | 0.5858 | 0.5764 | 0.5793 | 0.5563 |
| Judge Overall | 3.9655 | 4.0705 | 4.0550 | 4.0635 | 3.9540 |
| Accuracy | 3.5500 | 3.7500 | 3.8500 | 3.7500 | 3.7000 |
| Hallucination | 2.8500 | 3.0500 | 3.1500 | 3.2500 | 3.0000 |
| Safety | 4.4000 | 4.4000 | 4.4500 | 4.5000 | 4.2500 |
| Coherence | 4.6000 | 4.7500 | 4.7000 | 4.7500 | 4.6500 |
| Avg Time (s) | 6.8 | 6.6 | 6.0 | 6.9 | 6.7 |
| Critical Errors | 8 | 6 | 4 | 2 | 7 |
