"""
Prueba pareada de validación + ablación entre experimentos.

Uso:
  python src/11_paired_validation.py --experiments 23 26 27 28
  python src/11_paired_validation.py --experiments 23 26 27 28 --source mex
  python src/11_paired_validation.py --experiments 23 26 27 28 --test bootstrap
  python src/11_paired_validation.py --experiments 23 26 27 28 --test both
  python src/11_paired_validation.py --experiments 23 26 27 28 --age-bins 0 72 144 228
  python src/11_paired_validation.py --experiments 23 26 27 28 --output results/paired_23_28.json
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon


# ── helpers ──────────────────────────────────────────────────────────────────

def load_experiment(exp_id: int, base_dir: str, source: str = "rsna") -> dict:
    """Carga plot_data.json de validación y devuelve {id: (true, pred)}.
    source: 'rsna' → validation/, 'mex' → mex-validation/
    """
    folder = "mex-validation" if source == "mex" else "validation"
    path = os.path.join(base_dir, str(exp_id), folder, "plot_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontrado: {path}")
    with open(path) as f:
        data = json.load(f)
    scatter = data.get("scatter", {})
    ids = scatter.get("ids")
    trues = scatter.get("trues")
    preds = scatter.get("preds")
    if ids is None:
        raise ValueError(
            f"Exp {exp_id}: plot_data.json no contiene 'ids'. "
            "Re-corre 07_validation.py con la versión actualizada del script."
        )
    if not (len(ids) == len(trues) == len(preds)):
        raise ValueError(f"Exp {exp_id}: ids/trues/preds tienen distinto largo.")
    return {sid: (float(t), float(p)) for sid, t, p in zip(ids, trues, preds)}


def paired_errors(data_a: dict, data_b: dict):
    """Errores absolutos sobre la intersección de IDs comunes."""
    common = sorted(set(data_a) & set(data_b))
    err_a = np.array([abs(data_a[sid][0] - data_a[sid][1]) for sid in common])
    err_b = np.array([abs(data_b[sid][0] - data_b[sid][1]) for sid in common])
    return err_a, err_b, common


def wilcoxon_test(err_a, err_b):
    """
    Wilcoxon signed-rank test. Devuelve (statistic, p_value).
    Usa zero_method='wilcox' (descarta empates exactos).
    """
    diff = err_a - err_b
    if np.all(diff == 0):
        return np.nan, 1.0
    stat, p = wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def rank_biserial(err_a, err_b):
    """Tamaño de efecto: rank-biserial correlation r = 1 - 2W/n(n+1)."""
    n = len(err_a)
    stat, _ = wilcoxon(err_a, err_b, zero_method="wilcox", alternative="two-sided")
    r = 1 - (2 * stat) / (n * (n + 1))
    return float(r)


def bootstrap_paired_test(err_a, err_b, n_bootstrap: int = 10_000, seed: int = 42):
    """
    Bootstrap pareado para diferencia de MAE (MAE_A - MAE_B).

    Devuelve:
      delta_obs  : diferencia observada MAE_A - MAE_B
      ci_lo      : percentil 2.5 de la distribución bootstrap
      ci_hi      : percentil 97.5
      p_value    : fracción de muestras bootstrap donde el signo se invierte
                   (prueba dos colas basada en la distribución bajo H0 centrada en 0)
    """
    rng = np.random.default_rng(seed)
    n = len(err_a)
    delta_obs = float(np.mean(err_a) - np.mean(err_b))

    # distribución bootstrap de ΔMAE
    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = np.mean(err_a[idx]) - np.mean(err_b[idx])

    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    # p-valor dos colas: centramos la distribución en 0 (H0) y medimos cuántas
    # muestras son tan extremas como delta_obs
    centered = boot_deltas - np.mean(boot_deltas)
    p_value = float(np.mean(np.abs(centered) >= abs(delta_obs)))

    return delta_obs, ci_lo, ci_hi, p_value


def age_bin_label(lo, hi):
    return f"{lo//12}-{hi//12}a ({lo}-{hi}m)"


# ── ablation table ────────────────────────────────────────────────────────────

def ablation_table(datasets: dict[int, dict], age_bins: list[int], exp_configs: dict):
    """
    Tabla de MAE por grupo de edad para cada experimento.
    datasets: {exp_id: {sid: (true, pred)}}
    age_bins: lista de límites, ej. [0, 72, 144, 228]
    """
    header = ["Rango"] + [f"Exp {e}\n{exp_configs[e]}" for e in sorted(datasets)]
    rows = []

    all_ids = set.intersection(*[set(d) for d in datasets.values()])

    for lo, hi in zip(age_bins[:-1], age_bins[1:]):
        row = [age_bin_label(lo, hi)]
        for exp_id in sorted(datasets):
            d = datasets[exp_id]
            errs = [abs(d[sid][0] - d[sid][1]) for sid in all_ids
                    if lo <= d[sid][0] < hi]
            row.append(f"{np.mean(errs):.1f}" if errs else "—")
        rows.append(row)

    # Global MAE (intersección)
    row_global = ["Global"]
    for exp_id in sorted(datasets):
        d = datasets[exp_id]
        errs = [abs(d[sid][0] - d[sid][1]) for sid in all_ids]
        row_global.append(f"{np.mean(errs):.1f}")
    rows.append(row_global)

    return header, rows


def print_table(header, rows, col_width=20):
    fmt = "".join(f"{{:<{col_width}}}" for _ in header)
    sep = "-" * (col_width * len(header))
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c).replace("\n", " ") for c in row]))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prueba pareada de validación entre experimentos")
    parser.add_argument("--experiments", nargs="+", type=int, required=True,
                        help="Lista de IDs de experimento, ej: 23 26 27 28")
    parser.add_argument("--base-dir", default="experiments",
                        help="Directorio raíz de experimentos (default: experiments)")
    parser.add_argument("--age-bins", nargs="+", type=int, default=[0, 72, 144, 228],
                        help="Límites de grupos de edad en meses (default: 0 72 144 228)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Nivel de significancia antes de corrección (default: 0.05)")
    parser.add_argument("--source", choices=["rsna", "mex"], default="rsna",
                        help="Conjunto de validación: rsna (default) o mex")
    parser.add_argument("--test", choices=["wilcoxon", "bootstrap", "both"], default="both",
                        help="Prueba estadística a usar (default: both)")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Número de remuestreos bootstrap (default: 10000)")
    parser.add_argument("--output", default=None,
                        help="Guardar resultados JSON en este archivo (opcional)")
    args = parser.parse_args()

    exp_ids = args.experiments
    base_dir = args.base_dir
    age_bins = sorted(args.age_bins)
    alpha = args.alpha
    test_mode = args.test
    n_bootstrap = args.n_bootstrap
    source = args.source

    # ── cargar datos ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Cargando datos de validación ({source.upper()})...")
    datasets = {}
    exp_configs = {}
    for eid in exp_ids:
        try:
            datasets[eid] = load_experiment(eid, base_dir, source)
            cfg_path = os.path.join(base_dir, str(eid), "config.py")
            backbone = "?"
            if os.path.exists(cfg_path):
                for line in open(cfg_path):
                    if line.startswith("BASE_MODEL_CHOICE"):
                        backbone = line.split('"')[1]
                        break
            exp_configs[eid] = backbone
            print(f"  Exp {eid} ({backbone}): {len(datasets[eid])} muestras")
        except Exception as e:
            print(f"  Exp {eid}: ERROR — {e}")
            sys.exit(1)

    # ── intersección global ───────────────────────────────────────────────────
    all_ids = set.intersection(*[set(d) for d in datasets.values()])
    print(f"\nMuestras comunes (intersección): {len(all_ids)}")

    # ── ablación por grupo de edad ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLACIÓN — MAE por grupo de edad (meses)\n")
    header, rows = ablation_table(datasets, age_bins, exp_configs)
    print_table(header, rows, col_width=22)

    # ── prueba pareada ─────────────────────────────────────────────────────────
    pairs = list(combinations(sorted(exp_ids), 2))
    n_pairs = len(pairs)
    alpha_corrected = alpha / n_pairs  # Bonferroni

    results = []
    for (a, b) in pairs:
        err_a, err_b, common = paired_errors(datasets[a], datasets[b])
        mae_a = float(np.mean(err_a))
        mae_b = float(np.mean(err_b))
        label = f"Exp{a}({exp_configs[a]}) vs Exp{b}({exp_configs[b]})"
        entry = {
            "exp_a": a, "backbone_a": exp_configs[a], "mae_a": mae_a,
            "exp_b": b, "backbone_b": exp_configs[b], "mae_b": mae_b,
            "n_common": len(common),
        }

        if test_mode in ("wilcoxon", "both"):
            stat, p_w = wilcoxon_test(err_a, err_b)
            r = rank_biserial(err_a, err_b) if not np.isnan(stat) else np.nan
            entry.update({"wilcoxon_W": stat, "wilcoxon_p": p_w,
                          "wilcoxon_p_corrected": float(p_w * n_pairs) if not np.isnan(p_w) else None,
                          "wilcoxon_r": r,
                          "wilcoxon_significant": bool(p_w < alpha_corrected)})

        if test_mode in ("bootstrap", "both"):
            delta, ci_lo, ci_hi, p_b = bootstrap_paired_test(err_a, err_b, n_bootstrap)
            entry.update({"bootstrap_delta": delta, "bootstrap_ci_lo": ci_lo,
                          "bootstrap_ci_hi": ci_hi, "bootstrap_p": p_b,
                          "bootstrap_p_corrected": float(p_b * n_pairs),
                          "bootstrap_significant": bool(p_b < alpha_corrected)})

        results.append(entry)

    # ── imprimir Wilcoxon ──────────────────────────────────────────────────────
    if test_mode in ("wilcoxon", "both"):
        print(f"\n{'='*60}")
        print(f"WILCOXON signed-rank (Bonferroni α={alpha}/{n_pairs}={alpha_corrected:.4f})\n")
        col_w = 16
        hdr = ["Par", "n", "MAE_A", "MAE_B", "W", "p", "p-corr", "r", "Sig?"]
        fmt = f"{{:<30}}{''.join(f'{{:>{col_w}}}' for _ in hdr[1:])}"
        print(fmt.format(*hdr))
        print("-" * (30 + col_w * (len(hdr) - 1)))
        for e in results:
            label = f"Exp{e['exp_a']}({e['backbone_a']}) vs Exp{e['exp_b']}({e['backbone_b']})"
            p_w = e.get("wilcoxon_p", float("nan"))
            sig = "✓" if e.get("wilcoxon_significant") else ""
            print(fmt.format(
                label, str(e["n_common"]),
                f"{e['mae_a']:.2f}m", f"{e['mae_b']:.2f}m",
                f"{e.get('wilcoxon_W', float('nan')):.1f}",
                f"{p_w:.4f}" if not np.isnan(p_w) else "—",
                f"{e.get('wilcoxon_p_corrected') or 0:.4f}",
                f"{e.get('wilcoxon_r', float('nan')):.3f}",
                sig,
            ))
        print("\nNota r: |r|>0.1 pequeño, |r|>0.3 mediano, |r|>0.5 grande")

    # ── imprimir Bootstrap ─────────────────────────────────────────────────────
    if test_mode in ("bootstrap", "both"):
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP pareado (n={n_bootstrap:,}, Bonferroni α={alpha}/{n_pairs}={alpha_corrected:.4f})\n")
        col_w = 16
        hdr = ["Par", "n", "ΔMAE", "IC 95% lo", "IC 95% hi", "p", "p-corr", "Sig?"]
        fmt = f"{{:<30}}{''.join(f'{{:>{col_w}}}' for _ in hdr[1:])}"
        print(fmt.format(*hdr))
        print("-" * (30 + col_w * (len(hdr) - 1)))
        for e in results:
            label = f"Exp{e['exp_a']}({e['backbone_a']}) vs Exp{e['exp_b']}({e['backbone_b']})"
            sig = "✓" if e.get("bootstrap_significant") else ""
            print(fmt.format(
                label, str(e["n_common"]),
                f"{e.get('bootstrap_delta', 0):+.2f}m",
                f"{e.get('bootstrap_ci_lo', 0):.2f}m",
                f"{e.get('bootstrap_ci_hi', 0):.2f}m",
                f"{e.get('bootstrap_p', 1):.4f}",
                f"{e.get('bootstrap_p_corrected', 1):.4f}",
                sig,
            ))
        print("\nNota: ΔMAE = MAE_A - MAE_B. IC no incluye 0 → diferencia significativa.")

    # ── guardar JSON ──────────────────────────────────────────────────────────
    if args.output:
        out = {
            "experiments": {str(e): exp_configs[e] for e in exp_ids},
            "n_common": len(all_ids),
            "alpha": alpha,
            "alpha_corrected": alpha_corrected,
            "source": source,
            "test_mode": test_mode,
            "n_bootstrap": n_bootstrap if test_mode in ("bootstrap", "both") else None,
            "paired_tests": results,
        }
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
