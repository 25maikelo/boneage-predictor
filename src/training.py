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
from sklearn.model_selection import train_test_split

from config.paths import SEGMENTED_IMAGES_DIR, EXPERIMENTS_DIR
from config.experiment import load_experiment_config, get_experiment_output_dir
from src.models.losses import LOSS_MAP, dynamic_attention_loss
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


def plot_history(history, title, save_path):
    plt.figure(figsize=(8, 6))
    for key in ("loss", "val_loss", "mae", "val_mae"):
        if key in history.history:
            plt.plot(history.history[key], marker="o", label=key)
    plt.title(title); plt.xlabel("Época"); plt.ylabel("Valor")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path); plt.close()


# ============================================================
# MODELOS
# ============================================================
def create_segment_model(cfg):
    IMAGE_SIZE = cfg.IMAGE_SIZE
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name="input_layer")

    choice = cfg.BASE_MODEL_CHOICE.lower()
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


def clone_and_rename(model, prefix):
    def clone_fn(layer):
        c = layer.get_config()
        c["name"] = f"{prefix}_{c['name']}"
        return layer.__class__.from_config(c)
    cloned = tf.keras.models.clone_model(model, clone_function=clone_fn)
    cloned.set_weights(model.get_weights())
    cloned.name = f"{prefix}_model"
    return cloned


def create_fusion_model(segment_paths, cfg, loss_fn):
    inputs, outputs = [], []
    gender_in = None
    if cfg.USE_GENDER:
        gender_in = tf.keras.layers.Input(shape=(1,), name="gender_input")

    for seg in cfg.SEGMENTS_ORDER:
        inp = tf.keras.layers.Input(shape=(*cfg.IMAGE_SIZE, 3), name=f"input_{seg}")
        orig = load_model(f"{segment_paths[seg]}.keras", custom_objects=LOSS_MAP)
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
_cfg_global = None  # necesario para ProcessPoolExecutor


def train_one_segment(segment: str):
    global _cfg_global
    cfg = _cfg_global

    exp_dir = get_experiment_output_dir(cfg._experiment_idx)
    models_dir = os.path.join(exp_dir, "models")
    model_path = os.path.join(models_dir, f"{segment}_model.keras")

    if os.path.exists(model_path):
        print(f"Modelo '{segment}' ya existe, omitiendo.")
        return

    df = pd.read_csv(cfg.DATASET_PATH)
    df = df[(df["boneage"] >= cfg.AGE_RANGE[0]) & (df["boneage"] <= cfg.AGE_RANGE[1])]
    if cfg.USE_GENDER:
        df["gender"] = df["male"].astype(float)
    df["segment"] = segment

    train_df, val_df = train_test_split(df, test_size=cfg.TEST_SPLIT, random_state=42)

    loss_fn = LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss)
    model = create_segment_model(cfg)
    model.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                  loss=loss_fn, metrics=["mae"])

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
                 os.path.join(exp_dir, "training_history", f"{segment}_history.png"))
    print(f"Segmento {segment} completado.")


# ============================================================
# ENTRENAMIENTO DE FUSIÓN
# ============================================================
def train_fusion(cfg, exp_dir):
    models_dir = os.path.join(exp_dir, "models")
    fusion_path = os.path.join(models_dir, "fusion_model.keras")
    loss_fn = LOSS_MAP.get(cfg.LOSS_FUNCTION_NAME, dynamic_attention_loss)

    if os.path.exists(fusion_path):
        print("Fusión ya entrenada, cargando modelo existente.")
        fusion = load_model(fusion_path, custom_objects=LOSS_MAP)
        fusion.compile(optimizer=get_optimizer(cfg.OPTIMIZER_CHOICE, cfg.LEARNING_RATE),
                       loss=loss_fn, metrics=["mae"])
    else:
        seg_paths = {seg: os.path.join(models_dir, f"{seg}_model")
                     for seg in cfg.SEGMENTS_ORDER}
        fusion = create_fusion_model(seg_paths, cfg, loss_fn)

    df = pd.read_csv(cfg.DATASET_PATH)
    df = df[(df["boneage"] >= cfg.AGE_RANGE[0]) & (df["boneage"] <= cfg.AGE_RANGE[1])]
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
                 os.path.join(exp_dir, "training_history", "fusion_history.png"))

    if cfg.FINE_TUNING_EPOCHS > 0:
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
                     os.path.join(exp_dir, "training_history", "fusion_ft.png"))

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
    global _cfg_global
    args = parse_args()
    setup_gpu()

    cfg = load_experiment_config(args.experiment)
    cfg._experiment_idx = args.experiment
    _cfg_global = cfg

    # Usar SEGMENTS_FOLDER del config; si no tiene ruta absoluta, usar paths.py
    if not hasattr(cfg, "SEGMENTS_FOLDER") or not os.path.isabs(cfg.SEGMENTS_FOLDER):
        cfg.SEGMENTS_FOLDER = SEGMENTED_IMAGES_DIR

    exp_dir = get_experiment_output_dir(args.experiment)
    print(f"Experimento: {args.experiment} → {exp_dir}")

    models_dir = os.path.join(exp_dir, "models")
    pendientes = [seg for seg in cfg.SEGMENTS_ORDER
                  if not os.path.exists(os.path.join(models_dir, f"{seg}_model.keras"))]

    if pendientes:
        with timer("Entrenamiento de segmentos"):
            with ProcessPoolExecutor(max_workers=len(pendientes)) as ex:
                ex.map(train_one_segment, pendientes)
    else:
        print("Todos los modelos de segmento ya existen.")

    train_fusion(cfg, exp_dir)
    print("===== PROCESO COMPLETO FINALIZADO =====")


if __name__ == "__main__":
    setup_logging("training.py")
    main()
    report_timing(START_TIME, "training.py")
