"""Convert Teacher pseudo-labels into a COCO-like pose dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


COCO_17_CATEGORIES = [
    {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "skeleton": [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ],
    }
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def to_coco_keypoints(keypoints: list[list[float]], score_threshold: float) -> tuple[list[float], int]:
    coco_keypoints: list[float] = []
    num_visible = 0
    for x, y, score in keypoints:
        visibility = 2 if score >= score_threshold else 0
        if visibility > 0:
            num_visible += 1
        coco_keypoints.extend([float(x), float(y), visibility])
    return coco_keypoints, num_visible


def build_coco_dataset(
    dataset_dir: Path,
    labels_path: Path,
    output_dir: Path,
    score_threshold: float,
    min_mean_score: float,
    min_keypoints: int,
    copy_images: bool,
) -> None:
    records = read_jsonl(labels_path)
    output_images_dir = output_dir / "images"
    if copy_images:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for image_id, record in enumerate(records, start=1):
        if float(record["mean_keypoint_score"]) < min_mean_score:
            continue

        coco_keypoints, num_keypoints = to_coco_keypoints(record["keypoints"], score_threshold=score_threshold)
        if num_keypoints < min_keypoints:
            continue

        crop_source_path = dataset_dir / record["crop_path"]
        image_file_name = Path(record["crop_path"]).name
        if copy_images:
            shutil.copy2(crop_source_path, output_images_dir / image_file_name)

        images.append(
            {
                "id": image_id,
                "file_name": f"images/{image_file_name}" if copy_images else record["crop_path"],
                "width": int(record["crop_width"]),
                "height": int(record["crop_height"]),
                "source_crop_id": record["crop_id"],
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0.0, 0.0, float(record["crop_width"]), float(record["crop_height"])],
                "area": float(record["crop_width"] * record["crop_height"]),
                "iscrowd": 0,
                "keypoints": coco_keypoints,
                "num_keypoints": num_keypoints,
                "teacher_mean_score": float(record["mean_keypoint_score"]),
                "source_annotation_token": record["annotation_token"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "info": {
                    "description": "Pseudo-labeled nuScenes pedestrian pose dataset",
                    "keypoint_schema": "coco_17",
                },
                "images": images,
                "annotations": annotations,
                "categories": COCO_17_CATEGORIES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "num_input": len(records),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "score_threshold": score_threshold,
        "min_mean_score": min_mean_score,
        "min_keypoints": min_keypoints,
        "copy_images": copy_images,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved COCO-like dataset with {len(annotations)} annotations to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build COCO-like pose dataset from Teacher pseudo-labels.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pseudo/nuscenes_pose_coco"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-mean-score", type=float, default=0.45)
    parser.add_argument("--min-keypoints", type=int, default=8)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_coco_dataset(
        dataset_dir=args.dataset_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        min_mean_score=args.min_mean_score,
        min_keypoints=args.min_keypoints,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
