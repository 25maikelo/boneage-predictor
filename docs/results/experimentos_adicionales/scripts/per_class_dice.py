import os, sys, json, glob
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = "/lustre/home/mlozano/boneage-predictor"
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from config.paths import HAND_DETECTOR_IMAGES_DIR, HAND_DETECTOR_ANNOTATIONS_DIR

IMAGE_SIZE = (224, 224)
CLASS_NAMES = {0: "background", 1: "pinky", 2: "middle", 3: "thumb", 4: "wrist"}


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

    images = np.array(images) / 255.0
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    return images, masks, ids


def main():
    print("Cargando dataset anotado (LabelMe)...")
    images, masks, ids = load_data()
    print(f"Total imagenes anotadas: {len(images)}")

    x_train, x_val, y_train, y_val, ids_train, ids_val = train_test_split(
        images, masks, ids, test_size=0.2, random_state=42
    )
    print(f"Split reproducido: train={len(x_train)}  val={len(x_val)}")

    model_path = os.path.join(
        PROJECT_ROOT, "models/hand-detector/hand-detector_00/models/modelo_segmentacion.h5"
    )
    print(f"Cargando modelo: {model_path}")

    def iou_metric(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 5)
        iou_scores = []
        for c in range(5):
            yt = y_true_oh[..., c]
            yp = y_pred[..., c]
            inter = tf.reduce_sum(yt * yp, axis=[1, 2])
            union = tf.reduce_sum(yt, axis=[1, 2]) + tf.reduce_sum(yp, axis=[1, 2]) - inter
            iou_scores.append(inter / (union + K.epsilon()))
        return K.mean(tf.stack(iou_scores))

    def dice_metric(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), 5)
        dice_scores = []
        for c in range(5):
            yt = y_true_oh[..., c]
            yp = y_pred[..., c]
            inter = tf.reduce_sum(yt * yp, axis=[1, 2])
            dice_c = (2.0 * inter) / (
                tf.reduce_sum(yt, axis=[1, 2]) + tf.reduce_sum(yp, axis=[1, 2]) + K.epsilon()
            )
            dice_scores.append(dice_c)
        return K.mean(tf.stack(dice_scores))

    model = tf.keras.models.load_model(
        model_path, custom_objects={"iou_metric": iou_metric, "dice_metric": dice_metric}
    )
    print("Modelo cargado. Prediciendo sobre el subconjunto de validacion...")

    y_pred_probs = model.predict(x_val, batch_size=4, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)
    y_true_classes = y_val[..., 0]

    eps = 1e-8
    results = {}
    for c, name in CLASS_NAMES.items():
        yt = (y_true_classes == c).astype(np.float32)
        yp = (y_pred_classes == c).astype(np.float32)
        inter = np.sum(yt * yp, axis=(1, 2))
        union = np.sum(yt, axis=(1, 2)) + np.sum(yp, axis=(1, 2)) - inter
        dice = (2 * inter) / (np.sum(yt, axis=(1, 2)) + np.sum(yp, axis=(1, 2)) + eps)
        iou = inter / (union + eps)

        present = np.sum(yt, axis=(1, 2)) > 0
        dice_present = dice[present]
        iou_present = iou[present]

        results[name] = {
            "class_id": c,
            "n_images_with_class": int(present.sum()),
            "mean_dice": float(np.mean(dice_present)) if len(dice_present) else None,
            "std_dice": float(np.std(dice_present)) if len(dice_present) else None,
            "mean_iou": float(np.mean(iou_present)) if len(iou_present) else None,
            "std_iou": float(np.std(iou_present)) if len(iou_present) else None,
        }

    pixel_acc = float(np.mean(y_pred_classes == y_true_classes))

    macro_dice_no_bg = float(np.mean([results[n]["mean_dice"] for n in ["pinky", "middle", "thumb", "wrist"]]))
    macro_iou_no_bg = float(np.mean([results[n]["mean_iou"] for n in ["pinky", "middle", "thumb", "wrist"]]))
    macro_dice_all = float(np.mean([results[n]["mean_dice"] for n in CLASS_NAMES.values()]))
    macro_iou_all = float(np.mean([results[n]["mean_iou"] for n in CLASS_NAMES.values()]))

    out = {
        "n_total_annotated": len(images),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "pixel_accuracy": pixel_acc,
        "per_class": results,
        "macro_dice_no_background": macro_dice_no_bg,
        "macro_iou_no_background": macro_iou_no_bg,
        "macro_dice_all_classes": macro_dice_all,
        "macro_iou_all_classes": macro_iou_all,
    }

    out_path = os.path.join(PROJECT_ROOT, "review2/results_json/per_class_dice_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
