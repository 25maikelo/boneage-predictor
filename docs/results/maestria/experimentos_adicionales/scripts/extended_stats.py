import json, os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
EXPS = {"F-ResNet50": 23, "F-VGG16": 26, "F-DenseNet121": 27, "F-InceptionV3": 28}

rsna_csv = pd.read_csv(os.path.join(PROJECT_ROOT, "data/validation/validation_dataset.csv"))
rsna_csv["id"] = rsna_csv["id"].astype(str)
rsna_csv["sex"] = rsna_csv["male"].map({True: "M", False: "F", "TRUE": "M", "FALSE": "F"})

mex_csv = pd.read_csv(os.path.join(PROJECT_ROOT, "data/mex-validation/mex_dataset.csv"))
mex_csv["ID"] = mex_csv["ID"].astype(str)
mex_csv["sex"] = mex_csv["gender"].map({"M": "M", "F": "F"})


def age_bin(age_months):
    years = age_months / 12.0
    if years < 6:
        return "0-6a"
    elif years < 12:
        return "6-12a"
    else:
        return "12-19a"


def compute_stats(trues, preds):
    trues = np.array(trues, dtype=float)
    preds = np.array(preds, dtype=float)
    err = preds - trues
    ae = np.abs(err)
    return {
        "n": int(len(trues)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(ae)),
        "bias_mean_signed_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "pct_within_6m": float(np.mean(ae <= 6) * 100),
        "pct_within_12m": float(np.mean(ae <= 12) * 100),
    }


def rank_biserial(a, b):
    diffs = np.array(a) - np.array(b)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    return (n_pos - n_neg) / len(diffs)


def bootstrap_paired(err_a, err_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(err_a)
    diffs = np.array(err_a) - np.array(err_b)
    obs = np.mean(diffs)
    boots = np.empty(n_boot)
    idx_all = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boots[i] = np.mean(diffs[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p = float(np.mean(boots <= 0) * 2) if obs > 0 else float(np.mean(boots >= 0) * 2)
    p = min(p, 1.0)
    return float(obs), float(ci_lo), float(ci_hi), p


def holm_correction(pvals, alpha=0.05):
    n = len(pvals)
    arr = np.array([p if not np.isnan(p) else 1.0 for p in pvals])
    order = np.argsort(arr)
    adjusted = np.minimum(1.0, np.maximum.accumulate(arr[order] * np.arange(n, 0, -1)))
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist(), (result < alpha).tolist()


results = {"RSNA": {}, "MEX": {}}
abs_errors = {"RSNA": {}, "MEX": {}}

for name, exp in EXPS.items():
    # ---- RSNA ----
    with open(os.path.join(PROJECT_ROOT, f"experiments/{exp}/validation/plot_data.json")) as f:
        d = json.load(f)
    ids = [str(i) for i in d["scatter"]["ids"]]
    trues = d["scatter"]["trues"]
    preds = d["scatter"]["preds"]
    df = pd.DataFrame({"id": ids, "true": trues, "pred": preds})
    df = df.merge(rsna_csv[["id", "sex"]], on="id", how="left")
    df["age_bin"] = df["true"].apply(age_bin)

    overall = compute_stats(df["true"], df["pred"])
    by_sex = {s: compute_stats(g["true"], g["pred"]) for s, g in df.groupby("sex") if pd.notna(s)}
    by_age = {a: compute_stats(g["true"], g["pred"]) for a, g in df.groupby("age_bin")}

    results["RSNA"][name] = {"overall": overall, "by_sex": by_sex, "by_age_group": by_age}
    abs_errors["RSNA"][name] = {"ids": df["id"].tolist(),
                                  "abs_err": (df["pred"] - df["true"]).abs().tolist()}

    # ---- MEX (corregido, bone_age) ----
    with open(os.path.join(PROJECT_ROOT, f"experiments/{exp}/mex-validation/plot_data.json")) as f:
        d = json.load(f)
    ids = [str(i) for i in d["scatter"]["ids"]]
    trues = d["scatter"]["trues"]
    preds = d["scatter"]["preds"]
    df = pd.DataFrame({"id": ids, "true": trues, "pred": preds})
    df = df.merge(mex_csv[["ID", "sex"]], left_on="id", right_on="ID", how="left")
    df["age_bin"] = df["true"].apply(age_bin)

    overall = compute_stats(df["true"], df["pred"])
    by_sex = {s: compute_stats(g["true"], g["pred"]) for s, g in df.groupby("sex") if pd.notna(s)}
    by_age = {a: compute_stats(g["true"], g["pred"]) for a, g in df.groupby("age_bin")}

    results["MEX"][name] = {"overall": overall, "by_sex": by_sex, "by_age_group": by_age}
    abs_errors["MEX"][name] = {"ids": df["id"].tolist(),
                                 "abs_err": (df["pred"] - df["true"]).abs().tolist()}

# ---- Pruebas pareadas (Wilcoxon + bootstrap, Holm) sobre MEX corregido ----
paired = {"RSNA": [], "MEX": []}
for dataset in ["RSNA", "MEX"]:
    pairs = list(combinations(EXPS.keys(), 2))
    entries = []
    for a, b in pairs:
        ids_a = abs_errors[dataset][a]["ids"]
        ids_b = abs_errors[dataset][b]["ids"]
        common = sorted(set(ids_a) & set(ids_b))
        map_a = dict(zip(ids_a, abs_errors[dataset][a]["abs_err"]))
        map_b = dict(zip(ids_b, abs_errors[dataset][b]["abs_err"]))
        err_a = np.array([map_a[i] for i in common])
        err_b = np.array([map_b[i] for i in common])
        stat, p_w = wilcoxon(err_a, err_b) if np.any(err_a != err_b) else (np.nan, 1.0)
        r = rank_biserial(err_a, err_b)
        delta, ci_lo, ci_hi, p_b = bootstrap_paired(err_a, err_b)
        entries.append({"pair": f"{a} vs {b}", "n_common": len(common),
                         "delta_mae": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
                         "wilcoxon_p": float(p_w), "bootstrap_p": p_b, "rank_biserial_r": r})
    w_adj, w_sig = holm_correction([e["wilcoxon_p"] for e in entries])
    b_adj, b_sig = holm_correction([e["bootstrap_p"] for e in entries])
    for e, wa, ws, ba, bs in zip(entries, w_adj, w_sig, b_adj, b_sig):
        e["wilcoxon_p_holm"] = wa
        e["wilcoxon_sig"] = ws
        e["bootstrap_p_holm"] = ba
        e["bootstrap_sig"] = bs
    paired[dataset] = entries

out = {"extended_stats": results, "paired_tests": paired}
out_path = os.path.join(PROJECT_ROOT, "review2/results_json/extended_stats_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

for ds in ["RSNA", "MEX"]:
    print(f"=== {ds} ===")
    for name, v in results[ds].items():
        o = v["overall"]
        print(f"{name:15s} n={o['n']:4d} MAE={o['mae']:.2f} RMSE={o['rmse']:.2f} "
              f"medianAE={o['median_ae']:.2f} bias={o['bias_mean_signed_error']:+.2f} "
              f"std={o['std_error']:.2f} <=6m={o['pct_within_6m']:.1f}% <=12m={o['pct_within_12m']:.1f}%")
    print(f"--- Pruebas pareadas {ds} ---")
    for e in paired[ds]:
        print(f"  {e['pair']}: dMAE={e['delta_mae']:+.2f} IC95=[{e['ci_lo']:.2f},{e['ci_hi']:.2f}] "
              f"p_wilcoxon(Holm)={e['wilcoxon_p_holm']:.4f} p_bootstrap(Holm)={e['bootstrap_p_holm']:.4f} "
              f"sig={e['wilcoxon_sig'] and e['bootstrap_sig']}")

print(f"\nGuardado en {out_path}")
