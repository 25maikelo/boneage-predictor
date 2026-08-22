import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
OUT_DIR = os.path.join(PROJECT_ROOT, "review2", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data/training/boneage-training-dataset.csv"))
bal = pd.read_csv(os.path.join(PROJECT_ROOT, "data/training/dataset_analysis/balanced_dataset.csv"))


def plot_age_hist(ax, ages, title, color, bins):
    ax.hist(ages, bins=bins, color=color, edgecolor="k", alpha=0.85, density=False)
    kde = gaussian_kde(ages)
    xs = np.linspace(ages.min(), ages.max(), 300)
    bin_width = (ages.max() - ages.min()) / bins
    ax2 = ax.twinx()
    ax2.plot(xs, kde(xs), color="#3a6bbf", lw=1.3, alpha=0.7)
    ax2.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Age (Months)")
    ax.set_ylabel("Count")


def plot_gender_pie(ax, male_col, title):
    counts = male_col.map({True: "Male", False: "Female"}).value_counts()
    colors = {"Male": "#e8746b", "Female": "#6fa9d8"}
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=[colors[k] for k in counts.index], textprops={"fontsize": 11})
    ax.set_title(title, fontsize=12)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_age_hist(axes[0, 0], raw["boneage"],
              f"Raw Training Dataset: Age Distribution (n={len(raw):,})", "#a8c8ec", bins=40)
plot_gender_pie(axes[0, 1], raw["male"],
                f"Raw Training Dataset: Gender Percentage (n={len(raw):,})")

plot_age_hist(axes[1, 0], bal["boneage"],
              f"Balanced Training Dataset: Age Distribution (n={len(bal):,})", "#f5b895", bins=36)
plot_gender_pie(axes[1, 1], bal["male"],
                f"Balanced Training Dataset: Gender Percentage (n={len(bal):,})")

axes[1, 0].set_xlim(axes[0, 0].get_xlim())

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "distribucion_balanceo_antes_despues.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

print(f"\nRaw: n={len(raw)}, unique ages={raw['boneage'].nunique()}, "
      f"mean={raw['boneage'].mean():.1f}, median={raw['boneage'].median():.1f}")
print(f"Balanced: n={len(bal)}, unique ages={bal['boneage'].nunique()}, "
      f"mean={bal['boneage'].mean():.1f}, median={bal['boneage'].median():.1f}")
