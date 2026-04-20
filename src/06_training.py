#!/usr/bin/env python3
"""
Script principal de entrenamiento: modelos de segmento + modelo de fusión.

Uso:
    python src/training.py --experiment 26
"""
import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.models import load_model
import json
import shutil
from sklearn.model_selection import train_test_split, KFold

from config.paths import SEGMENTED_IMAGES_DIR, EXPERIMENTS_DIR, TRAINING_CSV
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.utils.losses import LOSS_MAP, dynamic_attention_loss
from src.utils.timing import report_timing, setup_logging, timer

START_TIME = time.time()


# ============================================================
# GPU
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {[g.name for g in gpus]}")
    else:
        print("ADVERTENCIA: No se detectó GPU. Ejecutando en CPU.")


# ============================================================
# CALLBACKS
# ============================================================
class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model.save(self.path)
            print(f"Modelo guardado en {self.path} (época {epoch + 1})")
        except Exception as e:
            print(f"Error guardando modelo: {e}")


class WarmupLR(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, init_lr, target_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.init_lr + (self.target_lr - self.init_lr) * ((epoch + 1) / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(lr)


# ============================================================
# UTILIDADES
# ============================================================
def get_optimizer(choice, lr):
    c = choice.lower()
    if c == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if c == "adamw":
        try:
            from tensorflow_addons.optimizers import AdamW
            return AdamW(learning_rate=lr, weight_decay=1e-4)
        except ImportError:
            print("AdamW no disponible, usando Adam")
    return tf.keras.optimizers.Adam(learning_rate=lr)


def plot_history(history, title, save_path, total_epochs=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual = len(history.history.get("loss", []))
    epochs = range(1, actual + 1)
    x_max = total_epochs if total_epochs and total_epochs >= actual else actual
    step = max(1, x_max // 10)
    x_ticks = list(range(1, x_max + 1, step))
    if x_max not in x_ticks:
        x_ticks.append(x_max)

    # Panel izquierdo: Loss
    for key, label, style in [("loss", "Train Loss", "-"), ("val_loss", "Val Loss", "--")]:
        if key in history.history:
            values = [max(v, 0) for v in history.history[key]]
            axes[0].plot(epochs, values, marker="o", linestyle=style, label=label)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].set_xlim(0.5, x_max + 0.5)
    axes[0].set_xticks(x_ticks); axes[0].set_ylim(bottom=0)
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel derecho: MAE
    for key, label, style in [("mae", "Train MAE", "-"), ("val_mae", "Val MAE", "--")]:
        if key in history.history:
            values = [max(v, 0) for v in history.history[key]]
            axes[1].plot(epochs, values, marker="o", linestyle=style, label=label)
    axes[1].set_title("MAE"); axes[1].set_xlabel("Época"); axes[1].set_ylabel("MAE (meses)")
    axes[1].set_xlim(0.5, x_max + 0.5)
    axes[1].set_xticks(x_ticks); axes[1].set_ylim(bottom=0)
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    if actual < x_max:
        for ax in axes:
            ax.axvline(actual, color="gray", linestyle=":", alpha=0.6,
                       label=f"Early stop (época {actual})")
        axes[0].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# ============================================================
# MODELOS
# ============================================================
def create_segment_model(cfg):
    IMAGE_SIZE = cfg.IMAGE_SIZE
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name="input_layer")

    choice = (cfg.BASE_MODEL_CHOICE or "").lower()
    weights = cfg.WEIGHTS
    bases = {
        "vgg16": tf.keras.applications.VGG16,
        "densenet121": tf.keras.applications.DenseNet121,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "resnet50": tf.keras.applications.ResNet50,
    }
    base_cls = bases.get(choice, tf.keras.applications.ResNet50)
    base = base_cls(include_top=False, weights=weights, input_tensor=inp)

    for layer in base.layers[-cfg.NUM_LAYERS_UNFREEZE:]:
        layer.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    inputs = [inp]
    if cfg.USE_GENDER:
        g = tf.keras.layers.Input(shape=(1,), name="gender_input")
        x = tf.keras.layers.Concatenate()([x, g])
        inputs.append(g)

    x = tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.DROPOUT_RATE)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="boneage_output")(x)
    return tf.keras.models.Model(inputs=inputs, outputs=out)


def create_simple_cnn_segment_model(cfg):
    """CNN simple sin backbone. Flatten preserva info espacial para fusión."""
    filters     = getattr(cfg, "CNN_FILTERS", [32, 64, 128, 256])
    kernel_size = getattr(cfg, "CNN_KERNEL_SIZE", 3)
    dropout     = getattr(cfg, "CNN_DROPOUT", 0.3)

    inp = tf.keras.layers.Input(shape=(*cfg.IMAGE_SIZE, 3), name="image_input")
    x = inp
    for f in filters:
        x = tf.keras.layers.Conv2D(f, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)

    x = tf.keras.layers.Flatten(name="flatten_features")(x)

    inputs = [inp]
    if cfg.USE_GENDER:
        g = tf.keras.layers.Input(shape=(1,), name="gender_input")
        x = tf.keras.layers.Concatenate()([x, g])
        inputs.append(g)

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="boneage_output")(x)
    return tf.keras.models.Model(inputs, out)


def create_fusion_model_cnn(segment_paths, cfg, loss_fn):
    """Fusión basada en vectores Flatten de cada CNN de segmento."""
    inputs, feature_outputs = [], []

    for seg in cfg.SEGMENTS_ORDER:
        seg_model = load_model(f"{segment_paths[seg]}", custom_objects=LOSS_MAP)
        # flatten_features only depends on image input (gender is added after flatten)
        feature_extractor = tf.keras.models.Model(
            inputs=seg_model.inputs[0],
            outputs=seg_model.get_layer("flatten_features").output,
            name=f"feature_extractor_{seg}",
        )
        feature_extractor.trainable = False

        inp = tf.keras.layers.Input(shape=(*cfg.IMAGE_SIZE, 3), name=f"input_{seg}")
        inputs.append(inp)
        feature_outputs.append(feature_extractor(inp))

    x = tf.keras.layers.Concatenate()(feature_outputs)

    if cfg.USE_GENDER:
        gender_in = tf.keras.layers.Input(shape=(1,), name="gender_input")
        inputs.append(gender_in)
        x = tf.keras.layers.Concatenate()([x, gender_in])

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="boneage_output")(x)

    fusion = tf.keras.models.Model(inputs=inputs, outputs=out, name="fusion_model_cnn")
    fusion.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_fn, metrics=["mae"],
    )
    return fusion


def clone_and_rename(model, prefix):
    def clone_fn(layer):
        c = layer.get_config()
        c["name"] = f"{prefix}_{c['name']}"
        return layer.__class__.from_config(c)
    cloned = tf.keras.models.clone_model(model, clone_function=clone_fn)
    cloned.set_weights(model.get_weights())
    named = tf.keras.models.Model(inputs=cloned.inputs, outputs=cloned.outputs,
                                   name=f"{prefix}_segment_model")
    return named


def create_fusion_model(segment_paths, cfg, loss_fn):
    inputs, outputs = [], []
    gender_in = None
    if cfg.USE_GENDER:
        gender_in = tf.keras.layers.Input(shape=(1,), name="gender_input")

    for seg in cfg.SEGMENTS_ORDER:
        inp = tf.keras.layers.Input(shape=(*cfg.IMAGE_SIZE, 3), name=f"input_{seg}")
        orig = load_model(f"{segment_paths[seg]}", custom_objects=LOSS_MAP)
        seg_model = clone_and_rename(orig, seg)
        out = seg_model([inp, gender_in]) if cfg.USE_GENDER else seg_model(inp)
        inputs.append(inp)
        outputs.append(out)

    if cfg.USE_GENDER:
        combined = tf.keras.layers.Concatenate()(outputs + [gender_in])
        inputs.append(gender_in)
    else:
        combined = tf.keras.layers.Concatenate()(outputs)

    x = tf.keras.layers.Dense(128, activation="relu")(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_out = tf.keras.layers.Dense(1, activation="linear", name="boneage_output")(x)

    fusion = tf.keras.models.Model(inputs=inputs, outputs=final_out, name="fusion_model")
    fusion.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_fn, metrics=["mae"]
    )
    return fusion


# ============================================================
# GENERADORES
# ============================================================
def custom_data_generator(df, cfg, augment=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=cfg.AUG_RESCALE,
        horizontal_flip=cfg.AUG_HORIZONTAL_FLIP if augment else False,
        rotation_range=cfg.AUG_ROTATION_RANGE if augment else 0,
        brightness_range=cfg.AUG_BRIGHTNESS_RANGE if augment else None,
        zoom_range=cfg.AUG_ZOOM_RANGE if augment else 0,
    )

    def gen():
        while True:
            for i in range(0, len(df), cfg.BATCH_SIZE):
                batch = df.iloc[i: i + cfg.BATCH_SIZE]
                imgs, ages = [], batch["boneage"].to_numpy(dtype="float32")
                for pid in batch["id"]:
                    seg = batch.loc[batch["id"] == pid, "segment"].iat[0]
                    path = os.path.join(cfg.SEGMENTS_FOLDER, seg, f"{pid}.png")
                    img = cv2.imread(path)
                    if img is None:
                        img = np.zeros((*cfg.IMAGE_SIZE, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, cfg.IMAGE_SIZE)
                    imgs.append(img)
                imgs = next(datagen.flow(np.array(imgs, dtype="float32"),
                                        batch_size=cfg.BATCH_SIZE, shuffle=False))
                if cfg.USE_GENDER:
                    genders = batch["gender"].to_numpy(dtype="float32").reshape(-1, 1)
                    yield (imgs, genders), ages
                else:
                    yield imgs, ages

    if cfg.USE_GENDER:
        sig = ((tf.TensorSpec((None, *cfg.IMAGE_SIZE, 3), tf.float32),
                tf.TensorSpec((None, 1), tf.float32)),
               tf.TensorSpec((None,), tf.float32))
    else:
        sig = (tf.TensorSpec((None, *cfg.IMAGE_SIZE, 3), tf.float32),
               tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig)


def fusion_data_generator(df, cfg, augment=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=cfg.AUG_RESCALE) if augment else None

    while True:
        df_sh = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(df_sh), cfg.BATCH_SIZE):
            batch = df_sh.iloc[i: i + cfg.BATCH_SIZE]
            inputs = []
            for seg in cfg.SEGMENTS_ORDER:
                arr = []
                for pid in batch["id"]:
                    img = cv2.imread(os.path.join(cfg.SEGMENTS_FOLDER, seg, f"{pid}.png"))
                    if img is None:
                        img = np.zeros((*cfg.IMAGE_SIZE, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, cfg.IMAGE_SIZE)
                    if datagen:
                        img = datagen.random_transform(img)
                    arr.append(img / 255.0)
                inputs.append(np.array(arr, dtype="float32"))
            if cfg.USE_GENDER:
                inputs.append(batch["male"].fillna(0).to_numpy().reshape(-1, 1))
            yield (*inputs,), batch["boneage"].to_numpy(dtype="float32")


# ============================================================
# ENTRENAMIENTO DE SEGMENTOS
# ============================================================
def _build_segment_model(cfg, loss_fn):
    model_type = getattr(cfg, "MODEL_TYPE", "backbone")
    if model_type == "simple_cnn":
        model = create_simple_cnn_segment_model(cfg)
    else:
        model = create_segment_model(cfg)
    model.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                  loss=loss_fn, metrics=["mae"])
    return model


def train_one_segment(args_tuple):
    segment, experiment_idx = args_tuple
    cfg = load_experiment_config(experiment_idx)
    cfg._experiment_idx = experiment_idx
    if not hasattr(cfg, "SEGMENTS_FOLDER") or not os.path.isabs(cfg.SEGMENTS_FOLDER):
        cfg.SEGMENTS_FOLDER = SEGMENTED_IMAGES_DIR

    exp_dir = get_experiment_output_dir(experiment_idx)
    models_dir = os.path.join(exp_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "training_history"), exist_ok=True)
    model_path = os.path.join(models_dir, f"{segment}_model")

    if os.path.exists(model_path):
        print(f"Modelo '{segment}' ya existe, omitiendo.")
        return

    df = pd.read_csv(getattr(cfg, "DATASET_PATH", TRAINING_CSV))
    df = df[(df["boneage"] >= cfg.AGE_RANGE[0]) & (df["boneage"] <= cfg.AGE_RANGE[1])]
    max_samples = getattr(cfg, "MAX_SAMPLES", None)
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    if cfg.USE_GENDER:
        df["gender"] = df["male"].astype(float)
    df["segment"] = segment
    df = df.reset_index(drop=True)

    loss_fn = LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss)
    use_cv = getattr(cfg, "USE_CROSS_VALIDATION", False)
    n_folds = getattr(cfg, "N_FOLDS", 5)

    if use_cv:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        best_val_loss = float("inf")
        best_fold_path = None

        with timer(f"Segmento {segment} (CV {n_folds} folds)"):
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                train_df = df.iloc[train_idx].copy()
                val_df   = df.iloc[val_idx].copy()
                fold_path = os.path.join(models_dir, f"{segment}_fold{fold}")

                print(f"Entrenando segmento: {segment} — Fold {fold + 1}/{n_folds}")
                model = _build_segment_model(cfg, loss_fn)
                cbs = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                      restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                          patience=3, min_lr=1e-7, verbose=1),
                    SaveModelCallback(fold_path),
                ]
                history = model.fit(
                    custom_data_generator(train_df, cfg, augment=cfg.USE_AUGMENTATION),
                    validation_data=custom_data_generator(val_df, cfg, augment=False),
                    epochs=cfg.EPOCHS_SEGMENT,
                    steps_per_epoch=len(train_df) // cfg.BATCH_SIZE,
                    validation_steps=len(val_df) // cfg.BATCH_SIZE,
                    verbose=2, callbacks=cbs,
                )

                best_epoch_val_loss = min(history.history["val_loss"])
                best_epoch_val_mae  = history.history["val_mae"][
                    history.history["val_loss"].index(best_epoch_val_loss)]
                fold_results.append({
                    "fold": fold,
                    "val_loss": best_epoch_val_loss,
                    "val_mae": best_epoch_val_mae,
                    "history": {k: [float(v) for v in vals]
                                for k, vals in history.history.items()},
                    "total_epochs": cfg.EPOCHS_SEGMENT,
                })
                plot_history(history, f"Segmento {segment} Fold {fold + 1}",
                             os.path.join(exp_dir, "training_history",
                                          f"{segment}_fold{fold}_history.png"),
                             total_epochs=cfg.EPOCHS_SEGMENT)

                if best_epoch_val_loss < best_val_loss:
                    best_val_loss = best_epoch_val_loss
                    best_fold_path = fold_path

        mean_mae = float(np.mean([r["val_mae"] for r in fold_results]))
        std_mae  = float(np.std([r["val_mae"] for r in fold_results]))
        best_fold_idx = int(min(range(len(fold_results)),
                                key=lambda i: fold_results[i]["val_loss"]))
        cv_metrics = {"folds": fold_results, "mean_val_mae": mean_mae,
                      "std_val_mae": std_mae, "best_fold": best_fold_idx}
        with open(os.path.join(exp_dir, "training_history",
                               f"{segment}_cv_metrics.json"), "w") as f:
            json.dump(cv_metrics, f, indent=2)

        if os.path.isdir(best_fold_path):
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            shutil.copytree(best_fold_path, model_path)
        else:
            shutil.copy2(best_fold_path, model_path)
        print(f"CV {segment}: MAE={mean_mae:.2f}±{std_mae:.2f} | "
              f"Mejor fold={best_fold_idx} (val_loss={best_val_loss:.4f})")
        print(f"Segmento {segment} completado (CV).")

    else:
        train_df, val_df = train_test_split(df, test_size=cfg.TEST_SPLIT, random_state=42)
        model = _build_segment_model(cfg, loss_fn)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                              restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                  patience=3, min_lr=1e-7, verbose=1),
            SaveModelCallback(model_path),
        ]
        print(f"Entrenando segmento: {segment}")
        with timer(f"Segmento {segment}"):
            history = model.fit(
                custom_data_generator(train_df, cfg, augment=cfg.USE_AUGMENTATION),
                validation_data=custom_data_generator(val_df, cfg, augment=False),
                epochs=cfg.EPOCHS_SEGMENT,
                steps_per_epoch=len(train_df) // cfg.BATCH_SIZE,
                validation_steps=len(val_df) // cfg.BATCH_SIZE,
                verbose=2, callbacks=cbs,
            )
        plot_history(history, f"Segmento {segment}",
                     os.path.join(exp_dir, "training_history", f"{segment}_history.png"),
                     total_epochs=cfg.EPOCHS_SEGMENT)
        with open(os.path.join(exp_dir, "training_history", f"{segment}_history.json"), "w") as f:
            json.dump({"history": {k: [float(v) for v in vals]
                                   for k, vals in history.history.items()},
                       "total_epochs": cfg.EPOCHS_SEGMENT}, f, indent=2)
        print(f"Segmento {segment} completado.")


# ============================================================
# REPORTE DE CROSS-VALIDATION
# ============================================================
def report_cv_results(cfg, exp_dir):
    history_dir = os.path.join(exp_dir, "training_history")
    segments = cfg.SEGMENTS_ORDER

    # Cargar métricas de cada segmento
    all_metrics = {}
    for seg in segments:
        path = os.path.join(history_dir, f"{seg}_cv_metrics.json")
        if not os.path.exists(path):
            print(f"[CV Report] No se encontró {path}, omitiendo.")
            continue
        with open(path) as f:
            all_metrics[seg] = json.load(f)

    if not all_metrics:
        return

    n_folds = len(next(iter(all_metrics.values()))["folds"])

    # ── 1. Tabla resumen ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, len(segments) * 0.6 + 2))
    ax.axis("off")
    headers = ["Segmento"] + [f"Fold {i+1} MAE" for i in range(n_folds)] + ["Media ± Std", "Mejor fold"]
    rows = []
    for seg, m in all_metrics.items():
        fold_maes = [f"{r['val_mae']:.2f}" for r in m["folds"]]
        summary = f"{m['mean_val_mae']:.2f} ± {m['std_val_mae']:.2f}"
        rows.append([seg.capitalize()] + fold_maes + [summary, str(m["best_fold"] + 1)])
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Resumen Cross-Validation — val MAE por segmento", pad=12, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, "cv_summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Boxplot MAE por segmento ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[r["val_mae"] for r in all_metrics[seg]["folds"]] for seg in segments if seg in all_metrics]
    labels = [seg.capitalize() for seg in segments if seg in all_metrics]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Distribución val MAE por segmento (CV)")
    ax.set_ylabel("val MAE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, "cv_boxplot.png"), dpi=150)
    plt.close()

    # ── 3. Heatmap fold × segmento ──────────────────────────────
    segs_available = [seg for seg in segments if seg in all_metrics]
    matrix = np.array([[r["val_mae"] for r in all_metrics[seg]["folds"]]
                        for seg in segs_available]).T  # shape: (folds, segments)
    fig, ax = plt.subplots(figsize=(max(6, len(segs_available) * 1.5), max(4, n_folds * 0.8 + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="val MAE")
    ax.set_xticks(range(len(segs_available)))
    ax.set_xticklabels([s.capitalize() for s in segs_available])
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_folds)])
    for i in range(n_folds):
        for j in range(len(segs_available)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] < matrix.max() * 0.8 else "white")
    ax.set_title("Heatmap val MAE — Fold × Segmento")
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, "cv_heatmap.png"), dpi=150)
    plt.close()

    # ── 4. Barras de val_MAE y val_Loss por fold por segmento ──
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]
    fold_positions = list(range(1, n_folds + 1))
    for seg in segs_available:
        m = all_metrics[seg]
        fig, axes = plt.subplots(1, 2, figsize=(max(8, n_folds * 1.5), 4))
        for fold_idx in range(n_folds):
            axes[0].bar(fold_positions[fold_idx], m["folds"][fold_idx]["val_mae"],
                        color=colors[fold_idx % len(colors)], alpha=0.8,
                        label=fold_labels[fold_idx])
            axes[1].bar(fold_positions[fold_idx], max(m["folds"][fold_idx]["val_loss"], 0),
                        color=colors[fold_idx % len(colors)], alpha=0.8)
        mean_loss = float(np.mean([r["val_loss"] for r in m["folds"]]))
        axes[0].axhline(m["mean_val_mae"], color="black", linestyle="--",
                        label=f"Media={m['mean_val_mae']:.2f}")
        axes[1].axhline(mean_loss, color="black", linestyle="--",
                        label=f"Media={mean_loss:.2f}")
        for ax in axes:
            ax.set_xticks(fold_positions)
            ax.set_xticklabels(fold_labels)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Fold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
        axes[0].set_title(f"{seg.capitalize()} — val MAE por fold")
        axes[0].set_ylabel("val MAE (meses)")
        axes[1].set_title(f"{seg.capitalize()} — val Loss por fold")
        axes[1].set_ylabel("val Loss")
        plt.suptitle(f"Cross-Validation: {seg.capitalize()}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(history_dir, f"{seg}_cv_bars.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    # ── 5. JSON resumen global ───────────────────────────────────
    summary = {seg: {"mean_val_mae": m["mean_val_mae"], "std_val_mae": m["std_val_mae"],
                     "best_fold": m["best_fold"]}
               for seg, m in all_metrics.items()}
    with open(os.path.join(history_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== RESUMEN CROSS-VALIDATION =====")
    for seg, m in all_metrics.items():
        print(f"  {seg:10s}  MAE = {m['mean_val_mae']:.2f} ± {m['std_val_mae']:.2f}"
              f"  (mejor fold: {m['best_fold'] + 1})")
    print("=====================================\n")
    print(f"Gráficos guardados en: {history_dir}")


# ============================================================
# ENTRENAMIENTO DE FUSIÓN
# ============================================================
def train_fusion(cfg, exp_dir):
    models_dir = os.path.join(exp_dir, "models")
    fusion_path = os.path.join(models_dir, "fusion_model")
    loss_fn = LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss)

    model_type = getattr(cfg, "MODEL_TYPE", "backbone")

    if os.path.exists(fusion_path):
        print("Fusión ya entrenada, omitiendo.")
        return
    else:
        seg_paths = {seg: os.path.join(models_dir, f"{seg}_model")
                     for seg in cfg.SEGMENTS_ORDER}
        if model_type == "simple_cnn":
            fusion = create_fusion_model_cnn(seg_paths, cfg, loss_fn)
        else:
            fusion = create_fusion_model(seg_paths, cfg, loss_fn)

    df = pd.read_csv(getattr(cfg, "DATASET_PATH", TRAINING_CSV))
    df = df[(df["boneage"] >= cfg.AGE_RANGE[0]) & (df["boneage"] <= cfg.AGE_RANGE[1])]
    max_samples = getattr(cfg, "MAX_SAMPLES", None)
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    train_f, val_f = train_test_split(df, test_size=cfg.TEST_SPLIT, random_state=42)

    sig = tuple([tf.TensorSpec((None, *cfg.IMAGE_SIZE, 3), tf.float32)
                 for _ in cfg.SEGMENTS_ORDER])
    if cfg.USE_GENDER:
        sig += (tf.TensorSpec((None, 1), tf.float32),)

    train_ds = tf.data.Dataset.from_generator(
        lambda: fusion_data_generator(train_f, cfg, augment=cfg.USE_AUGMENTATION),
        output_signature=(sig, tf.TensorSpec((None,), tf.float32))
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: fusion_data_generator(val_f, cfg, augment=False),
        output_signature=(sig, tf.TensorSpec((None,), tf.float32))
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, min_lr=1e-7),
        SaveModelCallback(fusion_path),
    ]
    if cfg.USE_WARMUP:
        cbs.insert(0, WarmupLR(cfg.WARMUP_EPOCHS, cfg.WARMUP_INITIAL_LR, cfg.LEARNING_RATE))

    print("Entrenando modelo de fusión...")
    with timer("Fusión"):
        hist_f = fusion.fit(
            train_ds, validation_data=val_ds,
            epochs=cfg.FUSION_EPOCHS,
            steps_per_epoch=len(train_f) // cfg.BATCH_SIZE,
            validation_steps=len(val_f) // cfg.BATCH_SIZE,
            callbacks=cbs, verbose=2,
        )
    plot_history(hist_f, "Fusión",
                 os.path.join(exp_dir, "training_history", "fusion_history.png"),
                 total_epochs=cfg.FUSION_EPOCHS)
    with open(os.path.join(exp_dir, "training_history", "fusion_history.json"), "w") as f:
        json.dump({"history": {k: [float(v) for v in vals]
                               for k, vals in hist_f.history.items()},
                   "total_epochs": cfg.FUSION_EPOCHS}, f, indent=2)

    if cfg.FINE_TUNING_EPOCHS > 0:
        if model_type == "simple_cnn":
            for layer in fusion.layers:
                if layer.name.startswith("feature_extractor_"):
                    layer.trainable = True
        fusion.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE / 10),
                       loss=loss_fn, metrics=["mae"])
        print("Fine-tuning fusión...")
        with timer("Fine-tuning"):
            hist_ft = fusion.fit(
                train_ds, validation_data=val_ds,
                epochs=cfg.FINE_TUNING_EPOCHS,
                steps_per_epoch=len(train_f) // cfg.BATCH_SIZE,
                validation_steps=len(val_f) // cfg.BATCH_SIZE,
                callbacks=cbs, verbose=2,
            )
        plot_history(hist_ft, "Fine-Tuning",
                     os.path.join(exp_dir, "training_history", "fusion_ft.png"),
                     total_epochs=cfg.FINE_TUNING_EPOCHS)
        with open(os.path.join(exp_dir, "training_history", "fusion_ft.json"), "w") as f:
            json.dump({"history": {k: [float(v) for v in vals]
                                   for k, vals in hist_ft.history.items()},
                       "total_epochs": cfg.FINE_TUNING_EPOCHS}, f, indent=2)

    print("Fusión completada.")


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento bone age predictor")
    parser.add_argument("--experiment", type=int, required=True,
                        help="Número de experimento (ej: 26)")
    return parser.parse_args()


def main():
    global _experiment_idx_global
    args = parse_args()
    setup_gpu()

    cfg = load_experiment_config(args.experiment)
    cfg._experiment_idx = args.experiment
    _experiment_idx_global = args.experiment

    # Usar SEGMENTS_FOLDER del config; si no tiene ruta absoluta, usar paths.py
    if not hasattr(cfg, "SEGMENTS_FOLDER") or not os.path.isabs(cfg.SEGMENTS_FOLDER):
        cfg.SEGMENTS_FOLDER = SEGMENTED_IMAGES_DIR

    exp_dir = get_experiment_output_dir(args.experiment)
    print(f"Experimento: {args.experiment} → {exp_dir}")

    models_dir = os.path.join(exp_dir, "models")
    pendientes = [seg for seg in cfg.SEGMENTS_ORDER
                  if not os.path.exists(os.path.join(models_dir, f"{seg}_model"))]

    if pendientes:
        with timer("Entrenamiento de segmentos"):
            for seg in pendientes:
                train_one_segment((seg, args.experiment))
    else:
        print("Todos los modelos de segmento ya existen.")

    if getattr(cfg, "USE_CROSS_VALIDATION", False):
        report_cv_results(cfg, exp_dir)

    train_fusion(cfg, exp_dir)
    print("===== PROCESO COMPLETO FINALIZADO =====")


if __name__ == "__main__":
    _args = parse_args()
    _exp_dir = get_experiment_output_dir(_args.experiment)
    setup_logging("06_training.py", log_dir=_exp_dir)
    main()
    report_timing(START_TIME, "06_training.py")
