from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_PATH = PROJECT_ROOT / "dataset" / "sample_claims.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_image_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [path.strip() for path in str(value).split(";") if path.strip()]


def count_existing_sample_images(frame: pd.DataFrame) -> int:
    image_paths = {
        path
        for paths in frame["image_paths"].map(parse_image_paths)
        for path in paths
    }
    return sum(1 for path in image_paths if (DATASET_ROOT / path).is_file())


def print_distribution(title: str, series: pd.Series) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    table = (
        series.value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    print(table.to_string(index=False))


def main() -> None:
    sample = pd.read_csv(DATASET_PATH)
    images_per_case = sample["image_paths"].map(parse_image_paths).map(len)

    summary = pd.DataFrame(
        [
            ("total_cases", len(sample)),
            ("sample_images_referenced", int(images_per_case.sum())),
            ("sample_images_existing", count_existing_sample_images(sample)),
            ("average_images_per_case", round(float(images_per_case.mean()), 2)),
        ],
        columns=["metric", "value"],
    )

    print("\nSample Dataset Summary")
    print("======================")
    print(summary.to_string(index=False))

    print_distribution("Object Type Distribution", sample["claim_object"])
    print_distribution("Issue Type Distribution", sample["issue_type"])
    print_distribution("Object Part Distribution", sample["object_part"])


if __name__ == "__main__":
    main()
