import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
EXPS = {"F-ResNet50": 23, "F-VGG16": 26, "F-DenseNet121": 27, "F-InceptionV3": 28}
OUT_DIR = os.path.join(PROJECT_ROOT, "review2", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MINUS = "−"


def fmt(x, decimals=1, sign=False):
    """Format a number using a proper Unicode minus sign (U+2212) instead of
    the ASCII hyphen-minus that f-strings produce for negative values."""
    s = f"{x:+.{decimals}f}" if sign else f"{x:.{decimals}f}"
    return s.replace("-", MINUS)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()
calib_results = {}
for ax, (name, exp) in zip(axes, EXPS.items()):
    with open(os.path.join(PROJECT_ROOT, f"experiments/{exp}/mex-validation/plot_data.json")) as f:
        d = json.load(f)
    trues = np.array(d["scatter"]["trues"])
    preds = np.array(d["scatter"]["preds"])

    ax.scatter(trues, preds, alpha=0.6, s=25, color="#3b6ea5", edgecolor="none",
               label="Individual patient")
    lims = [min(trues.min(), preds.min()) - 5, max(trues.max(), preds.max()) + 5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Identity (y=x)")

    slope, intercept, r, p, se = stats.linregress(trues, preds)
    xs = np.linspace(lims[0], lims[1], 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, color="#c0392b", lw=1.5,
            label=f"Regression (y={fmt(slope, 2)}x{'+' if intercept >= 0 else MINUS}{fmt(abs(intercept), 1)})")

    n = len(trues)
    resid = preds - (slope * trues + intercept)
    resid_std = np.std(resid)
    x_mean = trues.mean()
    ss_x = np.sum((trues - x_mean) ** 2)
    se_fit = resid_std * np.sqrt(1 / n + (xs - x_mean) ** 2 / ss_x)
    tval = stats.t.ppf(0.975, n - 2)
    ax.fill_between(xs, ys - tval * se_fit, ys + tval * se_fit, color="#c0392b", alpha=0.15,
                     label="95% CI")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("TW3 Bone Age (months)")
    ax.set_ylabel("Predicted Age (months)")
    ax.set_title(f"{name}  (n={n}, r={fmt(r, 2)}, slope={fmt(slope, 2)})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    calib_results[name] = {"n": n, "slope": float(slope), "intercept": float(intercept), "r": float(r)}

plt.suptitle("Dispersion: TW3 Bone Age vs Prediction (Mexican external validation)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figura8_mejorada_identidad_regresion.png"), dpi=150)
plt.close(fig)
print("Figure 8 (identity+regression) saved.")
print(json.dumps(calib_results, indent=2))

# ---------- Bland-Altman ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()
ba_results = {}
for ax, (name, exp) in zip(axes, EXPS.items()):
    with open(os.path.join(PROJECT_ROOT, f"experiments/{exp}/mex-validation/plot_data.json")) as f:
        d = json.load(f)
    trues = np.array(d["scatter"]["trues"])
    preds = np.array(d["scatter"]["preds"])

    mean_vals = (trues + preds) / 2
    diff_vals = preds - trues
    md = np.mean(diff_vals)
    sd = np.std(diff_vals)
    loa_hi, loa_lo = md + 1.96 * sd, md - 1.96 * sd

    ax.scatter(mean_vals, diff_vals, alpha=0.6, s=25, color="#3b6ea5", edgecolor="none",
               label="Individual patient (mean vs. difference)")
    ax.axhline(md, color="#c0392b", lw=1.5, label=f"Mean bias = {fmt(md, 2, sign=True)} m")
    ax.axhline(loa_hi, color="#c0392b", ls="--", lw=1,
               label=f"95% LoA = [{fmt(loa_lo, 1)}, {fmt(loa_hi, 1)}]")
    ax.axhline(loa_lo, color="#c0392b", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=0.8, label="Zero difference (no bias)")

    ax.set_xlabel("Mean of TW3 bone age and prediction (months)")
    ax.set_ylabel(f"Difference (prediction {MINUS} TW3 bone age) (months)")
    ax.set_title(f"{name}  Bland-Altman (n={len(trues)})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ba_results[name] = {"mean_diff": float(md), "sd_diff": float(sd),
                         "loa_lower": float(loa_lo), "loa_upper": float(loa_hi)}

plt.suptitle("Bland-Altman: Mexican external validation (TW3 bone age reference)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figura8_bland_altman.png"), dpi=150)
plt.close(fig)
print("Bland-Altman saved.")
print(json.dumps(ba_results, indent=2))

with open(os.path.join(OUT_DIR, "..", "results_json", "figure8_stats.json"), "w") as f:
    json.dump({"calibration": calib_results, "bland_altman": ba_results}, f, indent=2)
