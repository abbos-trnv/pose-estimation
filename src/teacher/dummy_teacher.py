"""Generate deterministic dummy keypoints for pipeline integration tests.

These labels are not valid training targets. They only verify that keypoint QA
and COCO conversion work before a real Teacher model is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.teacher.mmpose_teacher import COCO_17_KEYPOINTS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dummy_coco17_keypoints(width: float, height: float) -> list[list[float]]:
    points = [
        (0.50, 0.12),
        (0.46, 0.10),
        (0.54, 0.10),
        (0.42, 0.12),
        (0.58, 0.12),
        (0.38, 0.30),
        (0.62, 0.30),
        (0.32, 0.48),
        (0.68, 0.48),
        (0.30, 0.64),
        (0.70, 0.64),
        (0.43, 0.58),
        (0.57, 0.58),
        (0.40, 0.76),
        (0.60, 0.76),
        (0.38, 0.94),
        (0.62, 0.94),
    ]
    return [[round(x * width, 2), round(y * height, 2), 0.99] for x, y in points]


def generate_dummy_labels(dataset_dir: Path, output_dir: Path, limit: int | None) -> None:
    records = read_jsonl(dataset_dir / "manifest.jsonl")
    if limit is not None:
        records = records[:limit]

    pseudo_records: list[dict[str, Any]] = []
    for record in records:
        keypoints = dummy_coco17_keypoints(float(record["crop_width"]), float(record["crop_height"]))
        pseudo_records.append(
            {
                **record,
                "teacher": {
                    "framework": "dummy",
                    "keypoint_schema": "coco_17",
                    "warning": "Synthetic labels for integration testing only. Do not train on these labels.",
                },
                "keypoints": keypoints,
                "keypoint_names": COCO_17_KEYPOINTS,
                "mean_keypoint_score": 0.99,
                "min_keypoint_score": 0.99,
                "num_keypoints": 17,
                "num_keypoints_above_threshold": 17,
                "score_threshold": 0.5,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pseudo_records, output_dir / "pseudo_labels.jsonl")
    (output_dir / "pseudo_labels.json").write_text(json.dumps(pseudo_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pseudo_records)} dummy pseudo-labels to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy keypoints for integration tests.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pseudo/dummy_pose"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dummy_labels(dataset_dir=args.dataset_dir, output_dir=args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
