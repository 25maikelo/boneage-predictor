import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
OUT_DIR = os.path.join(PROJECT_ROOT, "review2", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location("mex_mod", f"{PROJECT_ROOT}/src/08_mex_validation.py")
mex_mod = importlib.util.module_from_spec(spec)
sys.modules["mex_mod"] = mex_mod
spec.loader.exec_module(mex_mod)

raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data/training/boneage-training-dataset.csv"))
bal = pd.read_csv(os.path.join(PROJECT_ROOT, "data/training/dataset_analysis/balanced_dataset.csv"))
mex = pd.read_csv(os.path.join(PROJECT_ROOT, "data/mex-validation/mex_dataset.csv"))
mex["boneage"] = mex["bone_age"].apply(mex_mod.parse_age_to_months)
mex["male"] = mex["gender"] == "M"


def plot_age_hist(ax, ages, title, color, kde_color, bins):
    ax.hist(ages, bins=bins, color=color, edgecolor="k", linewidth=0.6, alpha=0.9, density=False)
    kde = gaussian_kde(ages)
    xs = np.linspace(ages.min(), ages.max(), 300)
    ax2 = ax.twinx()
    ax2.plot(xs, kde(xs), color=kde_color, lw=1.3, alpha=0.75)
    ax2.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Age (Months)")
    ax.set_ylabel("Count")


def plot_gender_pie(ax, male_col, title):
    """Colorea por orden de frecuencia (mayoría=salmón, minoría=azul),
    igual que la Figura 2 original (no es un color fijo por sexo)."""
    counts = male_col.map({True: "Male", False: "Female"}).value_counts()
    order_colors = ["#e8746b", "#6fa9d8"]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=order_colors[:len(counts)], textprops={"fontsize": 11})
    ax.set_title(title, fontsize=12)


fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# (a)(b) — RSNA raw training dataset (idéntico a la Figura 2 original)
plot_age_hist(axes[0, 0], raw["boneage"], "RSNA Database: Age Distribution",
              "#a8c8ec", "#3a6bbf", bins=40)
plot_gender_pie(axes[0, 1], raw["male"], "RSNA Database: Gender Percentage")

# (c)(d) — Mexican dataset (idéntico a la Figura 2 original; edad = bone_age TW3)
plot_age_hist(axes[1, 0], mex["boneage"], "Mexican Database: Age Distribution",
              "#f5b8b0", "#c0392b", bins=10)
plot_gender_pie(axes[1, 1], mex["male"], "Mexican Database: Gender Percentage")

# (e)(f) — NUEVO: RSNA balanced training dataset
plot_age_hist(axes[2, 0], bal["boneage"], "Balanced RSNA Training Dataset: Age Distribution",
              "#f5d59a", "#c9820a", bins=36)
plot_gender_pie(axes[2, 1], bal["male"], "Balanced RSNA Training Dataset: Gender Percentage")

axes[2, 0].set_xlim(axes[0, 0].get_xlim())

for ax, letter in zip(axes.flat, "abcdef"):
    ax.text(0.5, -0.22, f"({letter})", transform=ax.transAxes, ha="center",
            fontsize=12, weight="bold")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "figura2_extendida_6paneles.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

print(f"\nRSNA raw:      n={len(raw):,}, mean={raw['boneage'].mean():.1f}, "
      f"male={100*raw['male'].mean():.1f}%")
print(f"Mexican:       n={len(mex):,}, mean={mex['boneage'].mean():.1f}, "
      f"male={100*mex['male'].mean():.1f}%")
print(f"RSNA balanced: n={len(bal):,}, mean={bal['boneage'].mean():.1f}, "
      f"male={100*bal['male'].mean():.1f}%")
