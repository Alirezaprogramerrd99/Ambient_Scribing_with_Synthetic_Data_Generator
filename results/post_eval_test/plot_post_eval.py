"""Auto-generated comparison plots."""
import matplotlib.pyplot as plt
import numpy as np
import json

data = {
  "Phi-3.5 (3.8B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.3721679409932165,
        "judge_overall": 4.75,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0,
        "bleu4": 0.1098241008417272,
        "medcon_f1": 0.6934090909090909
      },
      "rag_only": {
        "rouge_l": 0.3013120403246281,
        "judge_overall": 4.082,
        "judge_halluc": 3.6,
        "judge_safety": 4.6,
        "bertscore_f1": 0,
        "bleu4": 0.10514357662816051,
        "medcon_f1": 0.7355122655122655
      },
      "ft_only": {
        "rouge_l": 0.6373497497663818,
        "judge_overall": 4.182,
        "judge_halluc": 3.8,
        "judge_safety": 4.7,
        "bertscore_f1": 0,
        "bleu4": 0.5576245656991106,
        "medcon_f1": 0.8227272727272729
      },
      "ft_rag": {
        "rouge_l": 0.5620061065619533,
        "judge_overall": 4.39,
        "judge_halluc": 4.0,
        "judge_safety": 4.8,
        "bertscore_f1": 0,
        "bleu4": 0.46750048882644674,
        "medcon_f1": 0.7802432155657961
      },
      "teacher": {
        "rouge_l": 0.39973169775602724,
        "judge_overall": 4.867,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0,
        "bleu4": 0.2895782272610207,
        "medcon_f1": 0.729004329004329
      }
    },
    "rag_backends": {
      "llama_index": {
        "rouge_l": 0.5620061065619533,
        "judge_overall": 4.443,
        "bertscore_f1": 0,
        "bleu4": 0.46750048882644674,
        "medcon_f1": 0.7802432155657961
      },
      "manual": {
        "rouge_l": 0.6373497497663818,
        "judge_overall": 4.186,
        "bertscore_f1": 0,
        "bleu4": 0.5576245656991106,
        "medcon_f1": 0.8227272727272729
      },
      "hybrid": {
        "rouge_l": 0.5620061065619533,
        "judge_overall": 4.407,
        "bertscore_f1": 0,
        "bleu4": 0.46750048882644674,
        "medcon_f1": 0.7802432155657961
      }
    }
  }
}
labels = ["Phi-3.5 (3.8B)"]
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
