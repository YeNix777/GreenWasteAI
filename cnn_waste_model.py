from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from wastewise import Recognition


CNN_MODEL_PATH = Path("models") / "waste_cnn.keras"
CNN_LABELS_PATH = Path("models") / "waste_cnn_labels.txt"
IMAGE_SIZE = (160, 160)


CATEGORY_NAMES = {
    "battery": "Batterie oder Akku",
    "electronics": "Elektrogerät",
    "hazardous": "Schadstoff",
    "residual": "Restmüllähnlicher Abfall",
    "packaging": "Verpackung oder Wertstoff",
    "organic": "Bioabfall",
    "glass": "Glasverpackung",
    "paper": "Papierprodukt",
}


def load_cnn_model(
    model_path: Path = CNN_MODEL_PATH, labels_path: Path = CNN_LABELS_PATH
):
    if not model_path.exists() or not labels_path.exists():
        return None
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return model, labels


def image_array(image_bytes: bytes) -> np.ndarray:
    import io

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def analyze_with_cnn(image_bytes: bytes, loaded_model) -> Recognition:
    model, labels = loaded_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    probabilities = model.predict(preprocess_input(image_array(image_bytes)), verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    category = labels[top_index]
    confidence = float(probabilities[top_index])

    return Recognition(
        item=CATEGORY_NAMES.get(category, "Unbekannter Abfall"),
        material="durch CNN-Bildmodell geschätzt",
        category=category,
        condition="nicht sicher bestimmbar",
        hazardous=category in {"battery", "electronics", "hazardous"},
        confidence=confidence,
        explanation=(
            "Dieses Ergebnis stammt aus einem MobileNetV2-CNN, das mit dem "
            "gelabelten Kaggle-Waste-Datensatz nachtrainiert wurde."
        ),
    )
