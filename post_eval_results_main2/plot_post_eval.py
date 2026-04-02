"""Auto-generated comparison plots."""
import matplotlib.pyplot as plt
import numpy as np
import json

data = {
  "Phi-3.5 (3.8B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.36903499804171336,
        "judge_overall": 4.743200000000001,
        "judge_halluc": 5.0,
        "judge_safety": 4.98,
        "bertscore_f1": 0.8697495865821838,
        "bleu4": 0.1151328529148112,
        "medcon_f1": 0.759933989369728
      },
      "rag_only": {
        "rouge_l": 0.3666087161846576,
        "judge_overall": 4.760000000000002,
        "judge_halluc": 5.0,
        "judge_safety": 4.98,
        "bertscore_f1": 0.8698766326904297,
        "bleu4": 0.1153777407761787,
        "medcon_f1": 0.735350091831651
      },
      "ft_only": {
        "rouge_l": 0.6216631536755112,
        "judge_overall": 4.4434,
        "judge_halluc": 3.88,
        "judge_safety": 4.86,
        "bertscore_f1": 0.9399986517429352,
        "bleu4": 0.5459247335929721,
        "medcon_f1": 0.8581331085037256
      },
      "ft_rag": {
        "rouge_l": 0.624770182938587,
        "judge_overall": 4.4544,
        "judge_halluc": 4.1,
        "judge_safety": 4.84,
        "bertscore_f1": 0.9391051721572876,
        "bleu4": 0.5462971609105892,
        "medcon_f1": 0.8446361851406938
      },
      "teacher": {
        "rouge_l": 0.4163045907134565,
        "judge_overall": 4.773999999999998,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0.8754586970806122,
        "bleu4": 0.28427573243824483,
        "medcon_f1": 0.7743170428810052
      }
    },
    "rag_backends": {
      "dense_only": {
        "rouge_l": 0.5254280816319712,
        "judge_overall": 4.358800000000001,
        "bertscore_f1": 0.9194434988498688,
        "bleu4": 0.40253245392445086,
        "medcon_f1": 0.7899439889716977
      },
      "dense_rerank": {
        "rouge_l": 0.6244060063184524,
        "judge_overall": 4.4494,
        "bertscore_f1": 0.9408700537681579,
        "bleu4": 0.5520780644010795,
        "medcon_f1": 0.8637485385455073
      }
    }
  },
  "Llama-3.2 (3B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.3318325785662108,
        "judge_overall": 4.515199999999999,
        "judge_halluc": 4.96,
        "judge_safety": 4.98,
        "bertscore_f1": 0.8661276292800903,
        "bleu4": 0.29316749143762666,
        "medcon_f1": 0.7443678244381807
      },
      "rag_only": {
        "rouge_l": 0.32705792955164,
        "judge_overall": 4.4706,
        "judge_halluc": 4.92,
        "judge_safety": 4.94,
        "bertscore_f1": 0.8664843499660492,
        "bleu4": 0.2928804888902787,
        "medcon_f1": 0.7428148807634978
      },
      "ft_only": {
        "rouge_l": 0.6286000848830482,
        "judge_overall": 4.403799999999999,
        "judge_halluc": 3.84,
        "judge_safety": 4.8,
        "bertscore_f1": 0.9400948762893677,
        "bleu4": 0.5607254441178038,
        "medcon_f1": 0.8648667221863866
      },
      "ft_rag": {
        "rouge_l": 0.6245162302157845,
        "judge_overall": 4.372799999999999,
        "judge_halluc": 3.78,
        "judge_safety": 4.84,
        "bertscore_f1": 0.9389640152454376,
        "bleu4": 0.5514409233820781,
        "medcon_f1": 0.8418809110732759
      },
      "teacher": {
        "rouge_l": 0.4163045907134565,
        "judge_overall": 4.773999999999998,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0.8754586970806122,
        "bleu4": 0.28427573243824483,
        "medcon_f1": 0.7743170428810052
      }
    },
    "rag_backends": {
      "dense_only": {
        "rouge_l": 0.6135933941630454,
        "judge_overall": 4.275999999999999,
        "bertscore_f1": 0.9363952314853669,
        "bleu4": 0.5413409819185453,
        "medcon_f1": 0.8763611429677987
      },
      "dense_rerank": {
        "rouge_l": 0.6237849502951474,
        "judge_overall": 4.307599999999999,
        "bertscore_f1": 0.9385877072811126,
        "bleu4": 0.5511763225700324,
        "medcon_f1": 0.8500656511919515
      }
    }
  },
  "Llama-3.2 (1B)": {
    "configs": {
      "baseline": {
        "rouge_l": 0.282840645444969,
        "judge_overall": 3.7242000000000015,
        "judge_halluc": 2.82,
        "judge_safety": 4.04,
        "bertscore_f1": 0.8658911800384521,
        "bleu4": 0.1958017361423753,
        "medcon_f1": 0.6865662881322325
      },
      "rag_only": {
        "rouge_l": 0.2865472911798049,
        "judge_overall": 3.7276000000000007,
        "judge_halluc": 2.78,
        "judge_safety": 4.1,
        "bertscore_f1": 0.86659423828125,
        "bleu4": 0.19995719563154882,
        "medcon_f1": 0.6972808386002356
      },
      "ft_only": {
        "rouge_l": 0.5888989704553399,
        "judge_overall": 4.082,
        "judge_halluc": 3.22,
        "judge_safety": 4.58,
        "bertscore_f1": 0.9351249849796295,
        "bleu4": 0.5283468725297154,
        "medcon_f1": 0.8398555247993441
      },
      "ft_rag": {
        "rouge_l": 0.5905961473630911,
        "judge_overall": 4.0992,
        "judge_halluc": 3.12,
        "judge_safety": 4.56,
        "bertscore_f1": 0.9348841392993927,
        "bleu4": 0.527331005825467,
        "medcon_f1": 0.8595572634957148
      },
      "teacher": {
        "rouge_l": 0.4163045907134565,
        "judge_overall": 4.773999999999998,
        "judge_halluc": 5.0,
        "judge_safety": 5.0,
        "bertscore_f1": 0.8754586970806122,
        "bleu4": 0.28427573243824483,
        "medcon_f1": 0.7743170428810052
      }
    },
    "rag_backends": {
      "dense_only": {
        "rouge_l": 0.5885772133982974,
        "judge_overall": 4.125800000000001,
        "bertscore_f1": 0.9352712273597718,
        "bleu4": 0.5294751279367066,
        "medcon_f1": 0.8407520490422246
      },
      "dense_rerank": {
        "rouge_l": 0.595052738503184,
        "judge_overall": 4.115599999999999,
        "bertscore_f1": 0.9357434976100921,
        "bleu4": 0.5289074242652317,
        "medcon_f1": 0.8509850451996362
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
