"""Auto-generated comparison plots."""
import matplotlib.pyplot as plt
import numpy as np
import json

data = {
  "Phi-3.5 (3.8B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.3727976177502853,
        "judge_overall": 4.747000000000001,
        "judge_halluc": 5.0,
        "judge_safety": 4.96,
        "bertscore_f1": 0,
        "bleu4": 0.11518149872525364,
        "medcon_f1": 0.7456683499274832
      },
      "rag_only": {
        "rouge_l": 0.32065580662393145,
        "judge_overall": 4.303000000000002,
        "judge_halluc": 4.02,
        "judge_safety": 4.74,
        "bertscore_f1": 0,
        "bleu4": 0.11470175772766368,
        "medcon_f1": 0.7592936969166945
      },
      "ft_only": {
        "rouge_l": 0.630806160234759,
        "judge_overall": 4.395399999999999,
        "judge_halluc": 4.02,
        "judge_safety": 4.8,
        "bertscore_f1": 0,
        "bleu4": 0.5590075593003174,
        "medcon_f1": 0.8615518998258932
      },
      "ft_rag": {
        "rouge_l": 0.5871293583121509,
        "judge_overall": 4.351199999999999,
        "judge_halluc": 3.86,
        "judge_safety": 4.82,
        "bertscore_f1": 0,
        "bleu4": 0.5031251263202488,
        "medcon_f1": 0.8353784955919723
      },
      "teacher": {
        "rouge_l": 0.420553943255546,
        "judge_overall": 4.8804,
        "judge_halluc": 4.98,
        "judge_safety": 5.0,
        "bertscore_f1": 0,
        "bleu4": 0.29584346205078305,
        "medcon_f1": 0.7717152239205376
      }
    },
    "rag_backends": {
      "llama_index": {
        "rouge_l": 0.5871293583121509,
        "judge_overall": 4.332199999999998,
        "bertscore_f1": 0,
        "bleu4": 0.5031251263202488,
        "medcon_f1": 0.8353784955919723
      },
      "manual": {
        "rouge_l": 0.630806160234759,
        "judge_overall": 4.3806,
        "bertscore_f1": 0,
        "bleu4": 0.5590075593003174,
        "medcon_f1": 0.8615518998258932
      },
      "hybrid": {
        "rouge_l": 0.5871293583121509,
        "judge_overall": 4.3599999999999985,
        "bertscore_f1": 0,
        "bleu4": 0.5031251263202488,
        "medcon_f1": 0.8353784955919723
      }
    }
  },
  "Llama-3.2 (3B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.3289754206749919,
        "judge_overall": 4.6506000000000025,
        "judge_halluc": 4.92,
        "judge_safety": 4.96,
        "bertscore_f1": 0,
        "bleu4": 0.28619648126754116,
        "medcon_f1": 0.7398144536459083
      },
      "rag_only": {
        "rouge_l": 0.3212408890978875,
        "judge_overall": 4.502600000000003,
        "judge_halluc": 4.52,
        "judge_safety": 4.82,
        "bertscore_f1": 0,
        "bleu4": 0.24842917739744086,
        "medcon_f1": 0.7331276486459553
      },
      "ft_only": {
        "rouge_l": 0.623216158994798,
        "judge_overall": 4.3438,
        "judge_halluc": 3.68,
        "judge_safety": 4.84,
        "bertscore_f1": 0,
        "bleu4": 0.5494810303514555,
        "medcon_f1": 0.8563864733930794
      },
      "ft_rag": {
        "rouge_l": 0.6189822568822709,
        "judge_overall": 4.328599999999998,
        "judge_halluc": 3.7,
        "judge_safety": 4.7,
        "bertscore_f1": 0,
        "bleu4": 0.5434930854661062,
        "medcon_f1": 0.8521118691050615
      },
      "teacher": {
        "rouge_l": 0.4233135541137062,
        "judge_overall": 4.9037999999999995,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0,
        "bleu4": 0.2966596283348925,
        "medcon_f1": 0.7765096017382698
      }
    },
    "rag_backends": {
      "llama_index": {
        "rouge_l": 0.6189822568822709,
        "judge_overall": 4.2984,
        "bertscore_f1": 0,
        "bleu4": 0.5434930854661062,
        "medcon_f1": 0.8521118691050615
      },
      "manual": {
        "rouge_l": 0.623216158994798,
        "judge_overall": 4.326999999999999,
        "bertscore_f1": 0,
        "bleu4": 0.5494810303514555,
        "medcon_f1": 0.8563864733930794
      },
      "hybrid": {
        "rouge_l": 0.6189822568822709,
        "judge_overall": 4.3118,
        "bertscore_f1": 0,
        "bleu4": 0.5434930854661062,
        "medcon_f1": 0.8521118691050615
      }
    }
  },
  "Llama-3.2 (1B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.2840389183490136,
        "judge_overall": 3.686400000000001,
        "judge_halluc": 2.72,
        "judge_safety": 4.1,
        "bertscore_f1": 0,
        "bleu4": 0.19825689447569575,
        "medcon_f1": 0.6944203597375289
      },
      "rag_only": {
        "rouge_l": 0.2736232680414429,
        "judge_overall": 3.432,
        "judge_halluc": 2.48,
        "judge_safety": 3.7,
        "bertscore_f1": 0,
        "bleu4": 0.17174216885045523,
        "medcon_f1": 0.6737667689611101
      },
      "ft_only": {
        "rouge_l": 0.5926375326658695,
        "judge_overall": 4.09,
        "judge_halluc": 3.08,
        "judge_safety": 4.54,
        "bertscore_f1": 0,
        "bleu4": 0.532490389910397,
        "medcon_f1": 0.8465862933132783
      },
      "ft_rag": {
        "rouge_l": 0.5938651272870479,
        "judge_overall": 4.174600000000001,
        "judge_halluc": 3.4,
        "judge_safety": 4.58,
        "bertscore_f1": 0,
        "bleu4": 0.5354061976296539,
        "medcon_f1": 0.8488709818869998
      },
      "teacher": {
        "rouge_l": 0.42179931939782567,
        "judge_overall": 4.8972,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0,
        "bleu4": 0.2953710681831126,
        "medcon_f1": 0.7859688069199007
      }
    },
    "rag_backends": {
      "llama_index": {
        "rouge_l": 0.5938651272870479,
        "judge_overall": 4.2054,
        "bertscore_f1": 0,
        "bleu4": 0.5354061976296539,
        "medcon_f1": 0.8488709818869998
      },
      "manual": {
        "rouge_l": 0.5926375326658695,
        "judge_overall": 4.097599999999999,
        "bertscore_f1": 0,
        "bleu4": 0.532490389910397,
        "medcon_f1": 0.8465862933132783
      },
      "hybrid": {
        "rouge_l": 0.5938651272870479,
        "judge_overall": 4.176,
        "bertscore_f1": 0,
        "bleu4": 0.5354061976296539,
        "medcon_f1": 0.8488709818869998
      }
    }
  }
}
labels = ["Phi-3.5 (3.8B)", "Llama-3.2 (3B)", "Llama-3.2 (1B)"]
configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
config_labels = ["Base", "Base+RAG", "FT", "FT+RAG", "Teacher"]

colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

# ---- Figure 1: ROUGE-L across configs ----
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(configs))
width = 0.8 / len(labels)

for i, label in enumerate(labels):
    values = [data.get(label, {}).get("configs", {}).get(c, {}).get("rouge_l", 0) for c in configs]
    bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Configuration")
ax.set_ylabel("ROUGE-L")
ax.set_title("ROUGE-L Comparison Across Models and Configurations")
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(config_labels)
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("comparison_rouge.png", dpi=150)
plt.close()
print("Saved comparison_rouge.png")

# ---- Figure 2: Judge dimensions (ft_rag only) ----
dims = ["judge_overall", "judge_halluc", "judge_safety"]
dim_labels = ["Overall", "Hallucination", "Safety"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dims))
width = 0.8 / len(labels)

for i, label in enumerate(labels):
    ft_rag = data.get(label, {}).get("configs", {}).get("ft_rag", {})
    values = [ft_rag.get(d, 0) for d in dims]
    bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Dimension")
ax.set_ylabel("Score (1-5)")
ax.set_title("LLM Judge Scores: FT+RAG Configuration")
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(dim_labels)
ax.legend()
ax.set_ylim(0, 5.5)
plt.tight_layout()
plt.savefig("comparison_judge.png", dpi=150)
plt.close()
print("Saved comparison_judge.png")

# ---- Figure 3: Radar chart (ft_rag) ----
radar_dims = ["judge_overall", "judge_halluc", "judge_safety", "rouge_l", "bertscore_f1", "medcon_f1"]
radar_labels = ["Overall", "Hallucination", "Safety", "ROUGE-L", "BERTScore", "MEDCON"]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(radar_dims), endpoint=False).tolist()
angles += angles[:1]

for i, label in enumerate(labels):
    ft_rag = data.get(label, {}).get("configs", {}).get("ft_rag", {})
    values = []
    for d in radar_dims:
        v = ft_rag.get(d, 0)
        # Normalise judge scores to 0-1 range (they are 0-5)
        if "judge" in d:
            v = v / 5.0
        values.append(v)
    values += values[:1]
    ax.plot(angles, values, "o-", label=label, color=colors[i], linewidth=2)
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparison Radar: FT+RAG", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("comparison_radar.png", dpi=150)
plt.close()
print("Saved comparison_radar.png")

# ---- Figure 4: RAG Backend Comparison (ROUGE-L per backend per model) ----
all_backends = set()
for label in labels:
    all_backends.update(data.get(label, {}).get("rag_backends", {}).keys())
all_backends = sorted(all_backends)

if all_backends:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_backends))
    width = 0.8 / len(labels)
    
    for i, label in enumerate(labels):
        rag = data.get(label, {}).get("rag_backends", {})
        values = [rag.get(b, {}).get("rouge_l", 0) for b in all_backends]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("RAG Backend")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("RAG Backend Comparison Across Models")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(all_backends)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("comparison_rag_backends.png", dpi=150)
    plt.close()
    print("Saved comparison_rag_backends.png")
else:
    print("No RAG backend data found, skipping RAG backend plot.")

print("\nAll plots generated!")
