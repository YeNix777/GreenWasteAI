from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from wastewise import Recognition


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_PATH = Path("models") / "waste_classifier.json"

LABEL_TO_CATEGORY = {
    "batteries": "battery",
    "cans_all_type": "packaging",
    "ceramic_product": "residual",
    "coffee_tea_bags": "organic",
    "diapers": "residual",
    "e-waste": "electronics",
    "egg_shells": "organic",
    "food_scraps": "organic",
    "glass_containers": "glass",
    "kitchen_waste": "organic",
    "paints": "hazardous",
    "paper_products": "paper",
    "pesticides": "hazardous",
    "plastic_bottles": "packaging",
    "platics_bags_wrappers": "packaging",
    "sanitary_napkin": "residual",
    "stroform_product": "residual",
    "yard_trimmings": "organic",
}

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


def image_feature_vector(image_bytes: bytes) -> list[float]:
    with Image.open(PathBytes(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize((96, 96))
        array = np.asarray(image, dtype=np.float32) / 255.0

    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(array[:, :, channel], bins=16, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        features.extend(hist.tolist())

    gray = array.mean(axis=2)
    max_channel = array.max(axis=2)
    min_channel = array.min(axis=2)
    saturation = max_channel - min_channel
    features.extend(
        [
            float(array[:, :, 0].mean()),
            float(array[:, :, 1].mean()),
            float(array[:, :, 2].mean()),
            float(gray.mean()),
            float(gray.std()),
            float(saturation.mean()),
            float(saturation.std()),
        ]
    )

    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    features.extend([float(vertical_edges), float(horizontal_edges)])
    return features


class PathBytes:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = len(self.data) - self.offset
        chunk = self.data[self.offset : self.offset + size]
        self.offset += size
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.data) + offset
        return self.offset

    def tell(self) -> int:
        return self.offset


def iter_labeled_images(dataset_root: Path):
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.parent.name
        category = LABEL_TO_CATEGORY.get(label)
        if category:
            yield path, label, category


def train_centroid_model(dataset_root: Path) -> dict:
    grouped: dict[str, list[np.ndarray]] = {}
    label_counts: dict[str, int] = {}
    for path, label, category in iter_labeled_images(dataset_root):
        try:
            vector = np.array(image_feature_vector(path.read_bytes()), dtype=np.float32)
        except Exception:
            continue
        grouped.setdefault(category, []).append(vector)
        label_counts[label] = label_counts.get(label, 0) + 1

    if not grouped:
        raise ValueError(f"No labeled images found in {dataset_root}")

    centroids = {}
    for category, vectors in grouped.items():
        matrix = np.vstack(vectors)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        centroids[category] = centroid.tolist()

    return {
        "version": 1,
        "feature_count": len(next(iter(centroids.values()))),
        "centroids": centroids,
        "label_counts": label_counts,
        "category_counts": {category: len(vectors) for category, vectors in grouped.items()},
    }


def save_model(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_category(image_bytes: bytes, model: dict) -> tuple[str, float]:
    vector = np.array(image_feature_vector(image_bytes), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm

    scores = []
    for category, centroid_values in model["centroids"].items():
        centroid = np.array(centroid_values, dtype=np.float32)
        score = float(np.dot(vector, centroid))
        scores.append((category, score))
    scores.sort(key=lambda item: item[1], reverse=True)

    best_category, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    confidence = 1 / (1 + math.exp(-12 * (best_score - second_score)))
    confidence = max(0.35, min(0.92, confidence))
    return best_category, confidence


def analyze_with_local_model(image_bytes: bytes, model: dict) -> Recognition:
    category, confidence = predict_category(image_bytes, model)
    item = CATEGORY_NAMES.get(category, "Unbekannter Abfall")
    return Recognition(
        item=item,
        material="lokal aus Bildmerkmalen geschätzt",
        category=category,
        condition="nicht sicher bestimmbar",
        hazardous=category in {"battery", "electronics", "hazardous"},
        confidence=confidence,
        explanation=(
            "Dieses Ergebnis stammt aus einem lokal trainierten einfachen "
            "Bildklassifikationsmodell auf Basis des Kaggle-Datensatzes."
        ),
    )
