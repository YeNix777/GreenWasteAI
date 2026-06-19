from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cnn_waste_model import CNN_LABELS_PATH, CNN_MODEL_PATH, IMAGE_SIZE
from local_waste_model import IMAGE_EXTENSIONS, LABEL_TO_CATEGORY


def collect_images(dataset_root: Path):
    rows = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        category = LABEL_TO_CATEGORY.get(path.parent.name)
        if category:
            rows.append((str(path), category))
    if not rows:
        raise ValueError(f"No labeled images found in {dataset_root}")
    return rows


def build_dataset(filepaths, labels, class_names, batch_size, shuffle):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    label_lookup = {name: index for index, name in enumerate(class_names)}
    label_ids = [label_lookup[label] for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, label_ids))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=42)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = preprocess_input(tf.cast(image, tf.float32))
        return image, label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def train(dataset_root: Path, epochs: int, batch_size: int, limit_per_class: int | None):
    import tensorflow as tf

    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    rows = collect_images(dataset_root)
    if limit_per_class:
        sampled = []
        grouped = {}
        for path, category in rows:
            grouped.setdefault(category, []).append((path, category))
        for category_rows in grouped.values():
            random.shuffle(category_rows)
            sampled.extend(category_rows[:limit_per_class])
        rows = sampled

    filepaths = [path for path, _ in rows]
    categories = [category for _, category in rows]
    class_names = sorted(set(categories))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=categories,
    )

    train_ds = build_dataset(train_paths, train_labels, class_names, batch_size, True)
    val_ds = build_dataset(val_paths, val_labels, class_names, batch_size, False)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    loss, accuracy = model.evaluate(val_ds, verbose=0)

    CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(CNN_MODEL_PATH)
    CNN_LABELS_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    return {
        "classes": class_names,
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "validation_accuracy": float(accuracy),
        "history": history.history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 CNN on the Kaggle waste dataset."
    )
    parser.add_argument("--data", default="data/archive (1)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=250,
        help="Use fewer images per class for faster local training. Set 0 for all images.",
    )
    args = parser.parse_args()

    metrics = train(
        Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit_per_class=args.limit_per_class or None,
    )

    print(f"Saved CNN model to: {CNN_MODEL_PATH}")
    print(f"Saved labels to: {CNN_LABELS_PATH}")
    print(f"Training images: {metrics['train_images']}")
    print(f"Validation images: {metrics['validation_images']}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.1%}")
    print("Classes:")
    for name in metrics["classes"]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
