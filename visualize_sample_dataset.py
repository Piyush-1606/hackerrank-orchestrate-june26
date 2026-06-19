from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_PATH = PROJECT_ROOT / "dataset" / "sample_claims.csv"


def parse_image_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [path.strip() for path in str(value).split(";") if path.strip()]


def load_image(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def show_case(case_number: int, row: pd.Series) -> None:
    image_paths = parse_image_paths(row["image_paths"])
    images = [(path, load_image(DATASET_ROOT / path)) for path in image_paths]
    column_count = max(1, len(images))

    fig, axes = plt.subplots(1, column_count, figsize=(5 * column_count, 5))
    if column_count == 1:
        axes = [axes]

    fig.suptitle(
        "\n".join(
            [
                f"CASE {case_number:03d}",
                f"object_type={row['claim_object']} | issue_type={row['issue_type']} | object_part={row['object_part']}",
                f"status={row['claim_status']} | severity={row['severity']}",
            ]
        ),
        fontsize=12,
    )

    for axis, (relative_path, image) in zip(axes, images, strict=False):
        axis.axis("off")
        axis.set_title(Path(relative_path).name)
        if image is None:
            axis.text(0.5, 0.5, f"Missing image\n{relative_path}", ha="center", va="center")
        else:
            axis.imshow(image)

    for axis in axes[len(images):]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    sample = pd.read_csv(DATASET_PATH)
    print("Close each matplotlib window to continue to the next sample case.")

    for index, row in sample.iterrows():
        show_case(index + 1, row)


if __name__ == "__main__":
    main()
