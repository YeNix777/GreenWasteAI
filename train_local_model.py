from __future__ import annotations

import argparse
from pathlib import Path

from local_waste_model import MODEL_PATH, save_model, train_centroid_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the free local waste image classifier from the Kaggle dataset."
    )
    parser.add_argument(
        "--data",
        default="data/archive (1)",
        help="Path to the extracted Kaggle dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_PATH),
        help="Where to write the trained model JSON.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.data)
    model = train_centroid_model(dataset_root)
    save_model(model, Path(args.output))

    print(f"Trained local model written to: {args.output}")
    print("Category counts:")
    for category, count in sorted(model["category_counts"].items()):
        print(f"- {category}: {count}")


if __name__ == "__main__":
    main()
