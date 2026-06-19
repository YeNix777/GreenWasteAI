from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from wastewise import analyze_image, disposal_advice


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOCAL_DATASET_ROOT = Path("data") / "archive (1)"


LABEL_TO_CATEGORY = {
    "batteries": "battery",
    "e-waste": "electronics",
    "paints": "hazardous",
    "pesticides": "hazardous",
    "ceramic_product": "residual",
    "diapers": "residual",
    "platics_bags_wrappers": "packaging",
    "plastic_bags_wrappers": "packaging",
    "sanitary_napkin": "residual",
    "stroform_product": "residual",
    "styrofoam_product": "residual",
    "coffee_tea_bags": "organic",
    "cans_all_type": "packaging",
    "egg_shells": "organic",
    "food_scraps": "organic",
    "kitchen_waste": "organic",
    "yard_trimmings": "organic",
    "glass_containers": "glass",
    "paper_products": "paper",
    "plastic_bottles": "packaging",
}


def find_dataset_root() -> Path:
    if LOCAL_DATASET_ROOT.exists():
        return LOCAL_DATASET_ROOT

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Dataset not found locally. Either extract it into data/archive (1), "
            "or install kagglehub with: python -m pip install kagglehub"
        ) from exc

    return Path(kagglehub.dataset_download("phenomsg/waste-classification"))


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = path.parent.name
            expected_category = LABEL_TO_CATEGORY.get(label, "unknown")
            yield path, label, expected_category


def write_manifest(root: Path, output_path: Path) -> list[dict[str, str]]:
    rows = []
    for path, label, expected_category in iter_images(root):
        rows.append(
            {
                "image": str(path),
                "dataset_label": label,
                "expected_category": expected_category,
                "expected_bin": disposal_advice(
                    type(
                        "ExpectedRecognition",
                        (),
                        {
                            "category": expected_category,
                            "hazardous": expected_category == "hazardous",
                        },
                    )()
                ).bin_name,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate_sample(rows: list[dict[str, str]], sample_size: int, model: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("No OPENAI_API_KEY set. Manifest created, image evaluation skipped.")
        return

    checked = rows[:sample_size]
    correct = 0
    for index, row in enumerate(checked, start=1):
        path = Path(row["image"])
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        recognition = analyze_image(path.read_bytes(), mime_type, api_key, model)
        advice = disposal_advice(recognition)
        is_correct = advice.bin_name == row["expected_bin"]
        correct += int(is_correct)
        status = "OK" if is_correct else "WRONG"
        print(
            f"{index:03d}/{len(checked)} {status}: {path.name} | "
            f"expected={row['expected_bin']} predicted={advice.bin_name} "
            f"item={recognition.item}"
        )

    accuracy = correct / len(checked) if checked else 0
    print(f"\nSample accuracy: {accuracy:.1%} ({correct}/{len(checked)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manifest for the Kaggle waste dataset and optionally evaluate a sample."
    )
    parser.add_argument("--sample", type=int, default=0, help="Number of images to evaluate with the API.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument(
        "--output",
        default="data/waste_dataset_manifest.csv",
        help="CSV file to write.",
    )
    args = parser.parse_args()

    root = find_dataset_root()
    rows = write_manifest(root, Path(args.output))
    labels = {}
    for row in rows:
        labels[row["dataset_label"]] = labels.get(row["dataset_label"], 0) + 1

    print(f"Dataset root: {root}")
    print(f"Images found: {len(rows)}")
    print("Labels:")
    for label, count in sorted(labels.items()):
        mapped = LABEL_TO_CATEGORY.get(label, "unknown")
        print(f"- {label}: {count} images -> {mapped}")
    print(f"Manifest written to: {args.output}")

    if args.sample:
        evaluate_sample(rows, args.sample, args.model)


if __name__ == "__main__":
    main()
