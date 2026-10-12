"""
Diagnostico de fallos de segmentacion del modelo en produccion (hand-detector_00).

A diferencia de per_class_dice.py (que solo guarda el Dice/IoU agregado), este script
conserva el Dice/IoU por imagen y por clase, identifica los peores casos por clase, y
genera comparativas visuales (imagen, mascara real, mascara predicha) para esos casos.

Uso:
    python docs/results/segmentacion_diagnostico/scripts/diagnostico_fallos.py
"""
import os, sys, json, glob

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)
OUT_DIR = os.path.join(PROJECT_ROOT, "docs/results/segmentacion_diagnostico")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from config.paths import HAND_DETECTOR_IMAGES_DIR, HAND_DETECTOR_ANNOTATIONS_DIR

IMAGE_SIZE = (224, 224)
CLASS_NAMES = {0: "background", 1: "pinky", 2: "middle", 3: "thumb", 4: "wrist"}
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
}
N_WORST_PER_CLASS = 5


def load_data(image_size=IMAGE_SIZE):
    images, masks, ids = [], [], []
    image_paths = glob.glob(os.path.join(HAND_DETECTOR_IMAGES_DIR, "*.png"))
    for image_path in sorted(image_paths):
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(HAND_DETECTOR_ANNOTATIONS_DIR, f"{img_id}.json")
        if not os.path.exists(json_path):
            continue
        original_img = cv2.imread(image_path)
        if original_img is None:
            continue
        original_size = original_img.shape[:2]
        img = cv2.resize(original_img, image_size)
        images.append(img)

        mask = np.zeros(image_size, dtype=np.uint8)
        with open(json_path, encoding="utf-8") as f:
            annotations = json.load(f)
        for shape in annotations.get("shapes", []):
            label = shape["label"].lower().strip()
            points = np.array(shape["points"], dtype=np.float32)
            scale_x = image_size[1] / original_size[1]
            scale_y = image_size[0] / original_size[0]
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points = points.astype(np.int32)
            class_id = {"pinky": 1, "middle": 2, "thumb": 3, "wrist": 4}.get(label, 0)
            cv2.fillPoly(mask, [points], class_id)
        masks.append(mask)
        ids.append(img_id)

    images_arr = np.array(images) / 255.0
    masks_arr = np.expand_dims(np.array(masks), axis=-1)
    return images_arr, masks_arr, ids


def apply_color_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def iou_metric(y_true, y_pred):
    y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 5)
    scores = []
    for c in range(5):
        yt, yp = y_true_oh[..., c], y_pred[..., c]
        inter = tf.reduce_sum(yt * yp, axis=[1, 2])
        union = tf.reduce_sum(yt, axis=[1, 2]) + tf.reduce_sum(yp, axis=[1, 2]) - inter
        scores.append(inter / (union + K.epsilon()))
    return K.mean(tf.stack(scores))


def dice_metric(y_true, y_pred):
    y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 5)
    scores = []
    for c in range(5):
        yt, yp = y_true_oh[..., c], y_pred[..., c]
        inter = tf.reduce_sum(yt * yp, axis=[1, 2])
        dice_c = (2.0 * inter) / (tf.reduce_sum(yt, axis=[1, 2]) + tf.reduce_sum(yp, axis=[1, 2]) + K.epsilon())
        scores.append(dice_c)
    return K.mean(tf.stack(scores))


def main():
    print("Cargando dataset anotado (LabelMe)...")
    images, masks, ids = load_data()
    print(f"Total imagenes anotadas: {len(images)}")

    x_train, x_val, y_train, y_val, ids_train, ids_val = train_test_split(
        images, masks, ids, test_size=0.2, random_state=42
    )
    print(f"Split reproducido (mismo que per_class_dice.py): train={len(x_train)} val={len(x_val)}")

    model_path = os.path.join(
        PROJECT_ROOT, "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
    )
    print(f"Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"iou_metric": iou_metric, "dice_metric": dice_metric}
    )

    print("Prediciendo sobre el subconjunto de validacion...")
    y_pred_probs = model.predict(x_val, batch_size=4, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)
    y_true_classes = y_val[..., 0].astype(np.int32)

    eps = 1e-8
    n = len(x_val)

    # ---- Dice/IoU por imagen y por clase ----
    per_image = []
    for i in range(n):
        yt_img, yp_img = y_true_classes[i], y_pred_classes[i]
        entry = {"id": ids_val[i]}
        for c, name in CLASS_NAMES.items():
            yt = (yt_img == c).astype(np.float32)
            yp = (yp_img == c).astype(np.float32)
            present = yt.sum() > 0
            inter = float((yt * yp).sum())
            union = float(yt.sum() + yp.sum() - inter)
            dice = (2 * inter) / (yt.sum() + yp.sum() + eps)
            iou = inter / (union + eps)
            entry[name] = {
                "present": bool(present),
                "dice": float(dice) if present else None,
                "iou": float(iou) if present else None,
            }
        per_image.append(entry)

    with open(os.path.join(OUT_DIR, "results_json", "per_image_dice.json"), "w") as f:
        json.dump(per_image, f, indent=2)
    print("Guardado: results_json/per_image_dice.json")

    # ---- Matriz de confusion a nivel de pixel (clase real vs predicha) ----
    confusion = np.zeros((5, 5), dtype=np.int64)
    yt_flat = y_true_classes.ravel()
    yp_flat = y_pred_classes.ravel()
    for c_true in range(5):
        mask_true = yt_flat == c_true
        for c_pred in range(5):
            confusion[c_true, c_pred] = int(np.sum(yp_flat[mask_true] == c_pred))
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)

    with open(os.path.join(OUT_DIR, "results_json", "pixel_confusion_matrix.json"), "w") as f:
        json.dump({
            "classes": list(CLASS_NAMES.values()),
            "counts": confusion.tolist(),
            "row_normalized": confusion_norm.tolist(),
        }, f, indent=2)
    print("Guardado: results_json/pixel_confusion_matrix.json")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASS_NAMES.values(), rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASS_NAMES.values())
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion a nivel de pixel (normalizada por fila)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{confusion_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if confusion_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figures", "matriz_confusion_pixel.png"), dpi=150)
    plt.close()
    print("Guardado: figures/matriz_confusion_pixel.png")

    # ---- Peores casos por clase (solo donde la clase esta presente en GT) ----
    worst_cases = {}
    for c, name in CLASS_NAMES.items():
        if name == "background":
            continue
        scored = [
            (entry["id"], entry[name]["dice"])
            for entry in per_image
            if entry[name]["present"]
        ]
        scored.sort(key=lambda t: t[1])
        worst_cases[name] = scored[:N_WORST_PER_CLASS]

    with open(os.path.join(OUT_DIR, "results_json", "peores_casos.json"), "w") as f:
        json.dump(worst_cases, f, indent=2)
    print("Guardado: results_json/peores_casos.json")
    for name, cases in worst_cases.items():
        print(f"  Peores {name}: {cases}")

    # ---- Visualizacion de los peores casos ----
    id_to_idx = {vid: i for i, vid in enumerate(ids_val)}
    for name, cases in worst_cases.items():
        for rank, (img_id, dice) in enumerate(cases, 1):
            i = id_to_idx[img_id]
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(cv2.cvtColor((x_val[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax[0].set_title(f"Entrada (id={img_id})"); ax[0].axis("off")
            ax[1].imshow(apply_color_mask(y_true_classes[i]))
            ax[1].set_title("Mascara real"); ax[1].axis("off")
            ax[2].imshow(apply_color_mask(y_pred_classes[i]))
            ax[2].set_title(f"Prediccion (dice {name}={dice:.3f})"); ax[2].axis("off")
            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, "figures", f"peor_{name}_{rank}_{img_id}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
    print(f"Visualizaciones guardadas en {OUT_DIR}/figures/")


if __name__ == "__main__":
    main()
