#!/usr/bin/env python3
"""
Regenera todos los gráficos de entrenamiento a partir de los datos guardados en JSON.

Uso:
    python scripts/generate_plots.py --experiment 30 --lang es
    python scripts/generate_plots.py --experiment 30 --lang en
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.experiment import get_experiment_output_dir

# ── Traducciones ─────────────────────────────────────────────
LABELS = {
    "es": {
        # Entrenamiento (06)
        "epoch":        "Época",
        "loss":         "Loss",
        "mae":          "MAE (meses)",
        "train_loss":   "Train Loss",
        "val_loss":     "Val Loss",
        "train_mae":    "Train MAE",
        "val_mae":      "Val MAE",
        "early_stop":   "Early stop (época {n})",
        "fold":         "Fold",
        "mean":         "Media",
        "segment":      "Segmento",
        "cv_title":     "Resumen Cross-Validation — val MAE por segmento",
        "cv_boxplot":   "Distribución val MAE por segmento (CV)",
        "cv_heatmap":   "Heatmap val MAE — Fold × Segmento",
        "cv_bars_mae":  "{seg} — val MAE por fold",
        "cv_bars_loss": "{seg} — val Loss por fold",
        "cv_suptitle":  "Cross-Validation: {seg}",
        "best_fold":    "Mejor fold",
        "mean_std":     "Media ± Std",
        "seg_fold":     "Segmento {seg} — Fold {fold}",
        "fusion":       "Fusión",
        "finetune":     "Fine-Tuning",
        # Dataset (05)
        "age_months":   "Edad (meses)",
        "frequency":    "Frecuencia",
        "age_dist":     "Distribución de Edad — {title}",
        "ds_original":  "Dataset Original",
        "ds_filtered":  "Dataset Filtrado",
        "ds_balanced":  "Dataset Balanceado",
        "complete":     "Completas",
        "missing":      "Con faltantes",
        "proportion":   "Proporción de Imágenes Originales vs Usables",
        # Validación (07/08)
        "age_dist_val": "Distribución de Edad (Dataset de Validación)",
        "gender_dist":  "Distribución de Sexo (Validación)",
        "male":         "Masculino",
        "female":       "Femenino",
        "val_summary":  "Resumen de Validación",
        "mex_summary":  "Resumen de Validación (MEX)",
        "metric":       "Métrica",
        "value":        "Valor",
        "processed":    "Imágenes procesadas",
        "failed":       "Imágenes fallidas",
        "mae_months":   "MAE (meses)",
        "time_pre":     "Tiempo medio preprocess (s)",
        "time_seg":     "Tiempo medio segmentación (s)",
        "time_pred":    "Tiempo medio predicción (s)",
        "real_age":     "Edad real (meses)",
        "pred_age":     "Predicción (meses)",
        "scatter":      "Dispersión: Edad Real vs Predicción",
        "scatter_mex":  "Dispersión: Edad Real vs Predicción (MEX)",
        # Performance (09)
        "model":        "Modelo",
        "params":       "Parámetros",
        "loss_train":   "Loss Train",
        "mae_train":    "MAE Train",
        "loss_val":     "Loss Val",
        "mae_val":      "MAE Val",
        "comp_table":   "Tabla Comparativa de Modelos",
        "seg_names":    {"pinky": "Meñique", "middle": "Medio",
                         "thumb": "Pulgar", "wrist": "Muñeca"},
    },
    "en": {
        # Entrenamiento (06)
        "epoch":        "Epoch",
        "loss":         "Loss",
        "mae":          "MAE (months)",
        "train_loss":   "Train Loss",
        "val_loss":     "Val Loss",
        "train_mae":    "Train MAE",
        "val_mae":      "Val MAE",
        "early_stop":   "Early stop (epoch {n})",
        "fold":         "Fold",
        "mean":         "Mean",
        "segment":      "Segment",
        "cv_title":     "Cross-Validation Summary — val MAE per segment",
        "cv_boxplot":   "val MAE distribution per segment (CV)",
        "cv_heatmap":   "Heatmap val MAE — Fold × Segment",
        "cv_bars_mae":  "{seg} — val MAE per fold",
        "cv_bars_loss": "{seg} — val Loss per fold",
        "cv_suptitle":  "Cross-Validation: {seg}",
        "best_fold":    "Best fold",
        "mean_std":     "Mean ± Std",
        "seg_fold":     "Segment {seg} — Fold {fold}",
        "fusion":       "Fusion",
        "finetune":     "Fine-Tuning",
        # Dataset (05)
        "age_months":   "Age (months)",
        "frequency":    "Frequency",
        "age_dist":     "Age Distribution — {title}",
        "ds_original":  "Original Dataset",
        "ds_filtered":  "Filtered Dataset",
        "ds_balanced":  "Balanced Dataset",
        "complete":     "Complete",
        "missing":      "With missing",
        "proportion":   "Proportion of Original vs Usable Images",
        # Validación (07/08)
        "age_dist_val": "Age Distribution (Validation Dataset)",
        "gender_dist":  "Gender Distribution (Validation)",
        "male":         "Male",
        "female":       "Female",
        "val_summary":  "Validation Summary",
        "mex_summary":  "Validation Summary (MEX)",
        "metric":       "Metric",
        "value":        "Value",
        "processed":    "Processed images",
        "failed":       "Failed images",
        "mae_months":   "MAE (months)",
        "time_pre":     "Mean preprocess time (s)",
        "time_seg":     "Mean segmentation time (s)",
        "time_pred":    "Mean prediction time (s)",
        "real_age":     "Real age (months)",
        "pred_age":     "Prediction (months)",
        "scatter":      "Scatter: Real Age vs Prediction",
        "scatter_mex":  "Scatter: Real Age vs Prediction (MEX)",
        # Performance (09)
        "model":        "Model",
        "params":       "Parameters",
        "loss_train":   "Train Loss",
        "mae_train":    "Train MAE",
        "loss_val":     "Val Loss",
        "mae_val":      "Val MAE",
        "comp_table":   "Model Comparison Table",
        "seg_names":    {"pinky": "Pinky", "middle": "Middle",
                         "thumb": "Thumb", "wrist": "Wrist"},
    },
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#CCB974", "#64B5CD", "#D65F5F"]


# ── Utilidades ───────────────────────────────────────────────
def _xticks(x_max):
    step = max(1, x_max // 10)
    ticks = list(range(1, x_max + 1, step))
    if x_max not in ticks:
        ticks.append(x_max)
    return ticks


def plot_history_from_data(hist_data, title, save_path, L):
    """Genera gráfico loss+mae a partir de un dict con 'history' y 'total_epochs'."""
    h = hist_data["history"]
    total = hist_data.get("total_epochs", len(h.get("loss", [])))
    actual = len(h.get("loss", []))
    epochs = range(1, actual + 1)
    x_max = max(total, actual)
    x_ticks = _xticks(x_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for key, label, style, ax in [
        ("loss",    L["train_loss"], "-",  axes[0]),
        ("val_loss",L["val_loss"],   "--", axes[0]),
        ("mae",     L["train_mae"],  "-",  axes[1]),
        ("val_mae", L["val_mae"],    "--", axes[1]),
    ]:
        if key in h:
            values = [max(v, 0) for v in h[key]]
            ax.plot(epochs, values, marker="o", linestyle=style, label=label)

    for ax, ylabel, panel_title in [
        (axes[0], L["loss"], L["loss"]),
        (axes[1], L["mae"],  L["mae"]),
    ]:
        ax.set_title(panel_title)
        ax.set_xlabel(L["epoch"])
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.5, x_max + 0.5)
        ax.set_xticks(x_ticks)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    if actual < x_max:
        for ax in axes:
            ax.axvline(actual, color="gray", linestyle=":", alpha=0.6,
                       label=L["early_stop"].format(n=actual))
        axes[0].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_segment_plots(exp_dir, segments, L, suffix):
    history_dir = os.path.join(exp_dir, "training_history")

    # Plots por fold (CV)
    for seg in segments:
        cv_path = os.path.join(history_dir, f"{seg}_cv_metrics.json")
        if not os.path.exists(cv_path):
            continue
        with open(cv_path) as f:
            cv = json.load(f)
        for fold_data in cv["folds"]:
            if "history" not in fold_data:
                print(f"  Sin datos de historial para {seg} fold {fold_data['fold']} — omitiendo")
                continue
            title = L["seg_fold"].format(seg=seg.capitalize(),
                                         fold=fold_data["fold"] + 1)
            out = os.path.join(history_dir,
                               f"{seg}_fold{fold_data['fold']}_history_{suffix}.png")
            plot_history_from_data(fold_data, title, out, L)
            print(f"  {os.path.basename(out)}")

        # Plot sin CV
        hist_path = os.path.join(history_dir, f"{seg}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist_data = json.load(f)
            title = f"{L['segment']} {seg.capitalize()}"
            out = os.path.join(history_dir, f"{seg}_history_{suffix}.png")
            plot_history_from_data(hist_data, title, out, L)
            print(f"  {os.path.basename(out)}")


def generate_fusion_plots(exp_dir, L, suffix):
    history_dir = os.path.join(exp_dir, "training_history")
    for fname, title_key in [("fusion_history.json", "fusion"),
                              ("fusion_ft.json",     "finetune")]:
        path = os.path.join(history_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            hist_data = json.load(f)
        out = os.path.join(history_dir, fname.replace(".json", f"_{suffix}.png"))
        plot_history_from_data(hist_data, L[title_key], out, L)
        print(f"  {os.path.basename(out)}")


def generate_cv_report(exp_dir, segments, L, suffix):
    history_dir = os.path.join(exp_dir, "training_history")
    all_metrics = {}
    for seg in segments:
        path = os.path.join(history_dir, f"{seg}_cv_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[seg] = json.load(f)

    if not all_metrics:
        print("  Sin datos CV, omitiendo reporte.")
        return

    segs = [s for s in segments if s in all_metrics]
    n_folds = len(next(iter(all_metrics.values()))["folds"])

    # 1. Tabla resumen
    fig, ax = plt.subplots(figsize=(10, len(segs) * 0.6 + 2))
    ax.axis("off")
    headers = [L["segment"]] + [f"{L['fold']} {i+1} MAE" for i in range(n_folds)] \
              + [L["mean_std"], L["best_fold"]]
    rows = []
    for seg, m in all_metrics.items():
        fold_maes = [f"{r['val_mae']:.2f}" for r in m["folds"]]
        summary = f"{m['mean_val_mae']:.2f} ± {m['std_val_mae']:.2f}"
        rows.append([seg.capitalize()] + fold_maes + [summary, str(m["best_fold"] + 1)])
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(L["cv_title"], pad=12, fontsize=11)
    plt.tight_layout()
    out = os.path.join(history_dir, f"cv_summary_table_{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {os.path.basename(out)}")

    # 2. Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = [[r["val_mae"] for r in all_metrics[seg]["folds"]] for seg in segs]
    labels = [seg.capitalize() for seg in segs]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS[:len(data)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(L["cv_boxplot"])
    ax.set_ylabel(L["val_mae"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(history_dir, f"cv_boxplot_{suffix}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {os.path.basename(out)}")

    # 3. Heatmap
    matrix = np.array([[r["val_mae"] for r in all_metrics[seg]["folds"]]
                        for seg in segs]).T
    fig, ax = plt.subplots(figsize=(max(6, len(segs) * 1.5), max(4, n_folds * 0.8 + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label=L["val_mae"])
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([s.capitalize() for s in segs])
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"{L['fold']} {i+1}" for i in range(n_folds)])
    for i in range(n_folds):
        for j in range(len(segs)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] < matrix.max() * 0.8 else "white")
    ax.set_title(L["cv_heatmap"])
    plt.tight_layout()
    out = os.path.join(history_dir, f"cv_heatmap_{suffix}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {os.path.basename(out)}")

    # 4. Barras por segmento
    fold_labels    = [f"{L['fold']} {i+1}" for i in range(n_folds)]
    fold_positions = list(range(1, n_folds + 1))
    for seg in segs:
        m = all_metrics[seg]
        fig, axes = plt.subplots(1, 2, figsize=(max(8, n_folds * 1.5), 4))
        for fi in range(n_folds):
            axes[0].bar(fold_positions[fi], m["folds"][fi]["val_mae"],
                        color=COLORS[fi % len(COLORS)], alpha=0.8,
                        label=fold_labels[fi])
            axes[1].bar(fold_positions[fi], max(m["folds"][fi]["val_loss"], 0),
                        color=COLORS[fi % len(COLORS)], alpha=0.8)
        mean_loss = float(np.mean([r["val_loss"] for r in m["folds"]]))
        axes[0].axhline(m["mean_val_mae"], color="black", linestyle="--",
                        label=f"{L['mean']}={m['mean_val_mae']:.2f}")
        axes[1].axhline(mean_loss, color="black", linestyle="--",
                        label=f"{L['mean']}={mean_loss:.2f}")
        for ax in axes:
            ax.set_xticks(fold_positions)
            ax.set_xticklabels(fold_labels)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(L["fold"])
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
        axes[0].set_title(L["cv_bars_mae"].format(seg=seg.capitalize()))
        axes[0].set_ylabel(L["val_mae"])
        axes[1].set_title(L["cv_bars_loss"].format(seg=seg.capitalize()))
        axes[1].set_ylabel(L["val_loss"])
        plt.suptitle(L["cv_suptitle"].format(seg=seg.capitalize()), fontsize=12)
        plt.tight_layout()
        out = os.path.join(history_dir, f"{seg}_cv_bars_{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {os.path.basename(out)}")


# ── 05 Dataset Analysis ──────────────────────────────────────
def generate_dataset_plots(exp_dir, L, suffix):
    from config.paths import DATASET_ANALYSIS_DIR
    data_path = os.path.join(DATASET_ANALYSIS_DIR, "plot_data.json")
    if not os.path.exists(data_path):
        print("  Sin plot_data.json de dataset analysis, omitiendo.")
        return

    with open(data_path) as f:
        data = json.load(f)

    colors = {"original": "#66b3ff", "filtered": "#99cc99", "balanced": "#ffcc66"}
    title_keys = {"original": "ds_original", "filtered": "ds_filtered", "balanced": "ds_balanced"}
    fnames = {"original": "age_distribution", "filtered": "filtered_age_distribution",
              "balanced": "balanced_age_distribution"}

    for h in data.get("histograms", []):
        key = h["key"]
        ages = np.array(h["ages"])
        title = L["age_dist"].format(title=L[title_keys[key]])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ages, bins=20, color=colors.get(key, "#66b3ff"), edgecolor="black")
        ax.set_xlabel(L["age_months"]); ax.set_ylabel(L["frequency"])
        ax.set_title(title)
        plt.tight_layout()
        out = os.path.join(DATASET_ANALYSIS_DIR, f"{fnames[key]}_{suffix}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  {os.path.basename(out)}")

    prop = data.get("proportion", {})
    if prop:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([prop["complete"], prop["missing"]],
               labels=[L["complete"], L["missing"]],
               autopct="%1.1f%%", startangle=90, colors=["#66b3ff", "#ff9999"])
        ax.set_title(L["proportion"])
        plt.tight_layout()
        out = os.path.join(DATASET_ANALYSIS_DIR, f"dataset_proportion_{suffix}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  {os.path.basename(out)}")


# ── 07/08 Validation ─────────────────────────────────────────
def _generate_validation_plots(folder, L, suffix, is_mex=False):
    data_path = os.path.join(folder, "plot_data.json")
    if not os.path.exists(data_path):
        print(f"  Sin plot_data.json en {folder}, omitiendo.")
        return

    with open(data_path) as f:
        data = json.load(f)

    # Histograma de edades
    ages = np.array(data.get("ages", []))
    if len(ages):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ages, bins=20, color="#5b8ff9", edgecolor="k", alpha=0.8)
        ax.set_xlabel(L["age_months"], fontsize=14)
        ax.set_ylabel(L["frequency"], fontsize=14)
        ax.set_title(L["age_dist_val"], fontsize=16)
        ax.grid(alpha=0.3); plt.tight_layout()
        out = os.path.join(folder, f"histograma_edad_validacion_{suffix}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  {os.path.basename(out)}")

    # Pastel de sexo
    gender = data.get("gender", {})
    if gender:
        # Remapear claves Masculino/Femenino al idioma actual
        remap = {"Masculino": L["male"], "Femenino": L["female"],
                 "Male": L["male"], "Female": L["female"]}
        labels = [remap.get(k, k) for k in gender.keys()]
        values = list(gender.values())
        import matplotlib.cm as cm
        fig, ax = plt.subplots(figsize=(5, 5))
        patches, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=[cm.tab10(0), cm.tab10(1)], textprops={"fontsize": 14}
        )
        for t in texts: t.set_fontsize(16)
        for at in autotexts:
            at.set_fontsize(15); at.set_color("white"); at.set_weight("bold")
        ax.set_title(L["gender_dist"], fontsize=16, weight="bold")
        plt.tight_layout()
        out = os.path.join(folder, f"sexo_pastel_validacion_{suffix}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  {os.path.basename(out)}")

    # Tabla resumen
    summary = data.get("summary", {})
    if summary:
        title_key = "mex_summary" if is_mex else "val_summary"
        rows = [
            [L["metric"], L["value"]],
            [L["processed"], str(summary.get("processed", "N/A"))],
            [L["failed"],    str(summary.get("failed", "N/A"))],
            [L["mae_months"], f"{summary['mae']:.2f}" if summary.get("mae") is not None else "N/A"],
        ]
        if not is_mex:
            for key, lkey in [("time_preprocess", "time_pre"),
                               ("time_segment", "time_seg"),
                               ("time_predict", "time_pred")]:
                v = summary.get(key)
                rows.append([L[lkey], f"{v:.2f}" if v is not None else "N/A"])

        fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(rows)))
        ax.axis("off")
        tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
                       colWidths=[0.36, 0.24])
        tbl.auto_set_font_size(False); tbl.set_fontsize(16)
        for i in range(len(rows)):
            for j in range(2):
                cell = tbl[(i, j)]
                cell.set_facecolor("#4062bb" if i == 0 else
                                   ("#dbe2ef" if i % 2 else "#f9f7f7"))
                if i == 0:
                    cell.set_text_props(weight="bold", color="white")
                cell.set_edgecolor("#112d4e")
        tbl.scale(1.3, 1.7)
        plt.title(L[title_key], fontsize=19, weight="bold", color="#112d4e", pad=18)
        plt.tight_layout()
        out = os.path.join(folder, f"validation_summary_{suffix}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  {os.path.basename(out)}")

    # Scatter
    scatter = data.get("scatter", {})
    trues = scatter.get("trues", [])
    preds = scatter.get("preds", [])
    if trues and preds:
        scatter_title = L["scatter_mex"] if is_mex else L["scatter"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(trues, preds, c="#325288", alpha=0.7, edgecolors="k", s=80)
        ax.set_xlabel(L["real_age"]); ax.set_ylabel(L["pred_age"])
        ax.set_title(scatter_title)
        ax.grid(alpha=0.3); plt.tight_layout()
        out = os.path.join(folder, f"scatter_pred_vs_real_{suffix}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  {os.path.basename(out)}")


def generate_validation_plots(exp_dir, L, suffix):
    _generate_validation_plots(os.path.join(exp_dir, "validation"), L, suffix, is_mex=False)


def generate_mex_validation_plots(exp_dir, L, suffix):
    _generate_validation_plots(os.path.join(exp_dir, "mex-validation"), L, suffix, is_mex=True)


# ── 09 Performance Analysis ───────────────────────────────────
def generate_performance_plots(exp_dir, L, suffix):
    data_path = os.path.join(exp_dir, "evaluation", "comparative_table_data.json")
    if not os.path.exists(data_path):
        print("  Sin comparative_table_data.json, omitiendo.")
        return

    with open(data_path) as f:
        data = json.load(f)

    seg_names = L["seg_names"]
    col_keys = ["model", "params", "loss_train", "mae_train", "loss_val", "mae_val"]
    cols = [L[k] for k in col_keys]

    # Traducir nombres de segmentos en la primera columna
    rows = []
    for row in data["rows"]:
        translated = list(row)
        translated[0] = seg_names.get(translated[0].lower(), translated[0])
        rows.append(translated)

    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.4 * len(rows)))
    fig.patch.set_visible(False); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.scale(1, 2)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f1f2" if i % 2 == 0 else "#e0e0e0")
        cell.set_edgecolor("black")
    out = os.path.join(exp_dir, "evaluation", f"tabla_comparativa_{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    print(f"  {os.path.basename(out)}")


# ── Main ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Regenerar gráficos de entrenamiento")
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--lang", choices=["es", "en"], default="es",
                        help="Idioma de los gráficos (es | en)")
    return parser.parse_args()


def main():
    args = parse_args()
    L = LABELS[args.lang]
    suffix = args.lang

    exp_dir  = get_experiment_output_dir(args.experiment)
    cfg_path = os.path.join(exp_dir, "config.py")
    if not os.path.exists(cfg_path):
        print(f"No se encontró el experimento {args.experiment}")
        sys.exit(1)

    from config.experiment import load_experiment_config
    cfg = load_experiment_config(args.experiment)
    segments = cfg.SEGMENTS_ORDER

    print(f"\nExperimento {args.experiment} — idioma: {args.lang}")
    print(f"Directorio: {exp_dir}\n")

    print("── Dataset Analysis (05) ──────────────────────")
    generate_dataset_plots(exp_dir, L, suffix)

    print("\n── Segmentos (06) ─────────────────────────────")
    generate_segment_plots(exp_dir, segments, L, suffix)

    print("\n── Fusión (06) ────────────────────────────────")
    generate_fusion_plots(exp_dir, L, suffix)

    print("\n── Reporte CV (06) ────────────────────────────")
    generate_cv_report(exp_dir, segments, L, suffix)

    print("\n── Validación (07) ────────────────────────────")
    generate_validation_plots(exp_dir, L, suffix)

    print("\n── Validación MEX (08) ────────────────────────")
    generate_mex_validation_plots(exp_dir, L, suffix)

    print("\n── Performance (09) ───────────────────────────")
    generate_performance_plots(exp_dir, L, suffix)

    print(f"\nListo. Gráficos guardados en: {exp_dir}/")


if __name__ == "__main__":
    main()
