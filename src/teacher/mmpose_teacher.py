"""MMPose-based Teacher inference for pedestrian crops.

This module intentionally imports MMPose lazily. The base project can prepare
datasets without heavy OpenMMLab dependencies, while Teacher inference can be
enabled in a GPU environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_17_KEYPOINTS = [
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
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_mmpose_model(config_path: str, checkpoint_path: str, device: str):
    try:
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
    except ImportError as exc:
        raise RuntimeError(
            "MMPose is not installed. Install OpenMMLab/MMPose in the Teacher "
            "environment before running pseudo-label generation."
        ) from exc

    register_all_modules()
    return init_model(config_path, checkpoint_path, device=device)


def predict_crop(model: Any, crop_path: Path) -> dict[str, Any]:
    try:
        from mmpose.apis import inference_topdown
    except ImportError as exc:
        raise RuntimeError("MMPose is not installed.") from exc

    image = cv2.imread(str(crop_path))
    if image is None:
        raise RuntimeError(f"Failed to read crop: {crop_path}")

    height, width = image.shape[:2]
    bbox = np.array([[0, 0, width, height]], dtype=np.float32)
    result = inference_topdown(model, str(crop_path), bboxes=bbox, bbox_format="xyxy")[0]

    keypoints = np.asarray(result.pred_instances.keypoints)[0]
    scores = np.asarray(result.pred_instances.keypoint_scores)[0]
    keypoints_with_scores = [
        [float(x), float(y), float(score)]
        for (x, y), score in zip(keypoints, scores, strict=True)
    ]

    return {
        "keypoints": keypoints_with_scores,
        "keypoint_scores": [float(score) for score in scores],
        "mean_keypoint_score": float(scores.mean()),
        "min_keypoint_score": float(scores.min()),
        "num_keypoints": int(len(scores)),
    }


def run_teacher(
    dataset_dir: Path,
    output_dir: Path,
    mmpose_config: str,
    checkpoint: str,
    device: str,
    limit: int | None,
    score_threshold: float,
) -> None:
    records = read_jsonl(dataset_dir / "manifest.jsonl")
    if limit is not None:
        records = records[:limit]

    model = load_mmpose_model(config_path=mmpose_config, checkpoint_path=checkpoint, device=device)
    pseudo_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        crop_path = dataset_dir / record["crop_path"]
        prediction = predict_crop(model, crop_path)
        keypoint_scores = prediction["keypoint_scores"]
        pseudo_records.append(
            {
                **record,
                "teacher": {
                    "framework": "mmpose",
                    "config": mmpose_config,
                    "checkpoint": checkpoint,
                    "device": device,
                    "keypoint_schema": "coco_17",
                },
                "keypoints": prediction["keypoints"],
                "keypoint_names": COCO_17_KEYPOINTS,
                "mean_keypoint_score": prediction["mean_keypoint_score"],
                "min_keypoint_score": prediction["min_keypoint_score"],
                "num_keypoints": prediction["num_keypoints"],
                "num_keypoints_above_threshold": int(sum(score >= score_threshold for score in keypoint_scores)),
                "score_threshold": score_threshold,
            }
        )
        if index % 100 == 0:
            print(f"Teacher inference: {index}/{len(records)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pseudo_records, output_dir / "pseudo_labels.jsonl")
    (output_dir / "pseudo_labels.json").write_text(json.dumps(pseudo_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pseudo_records)} pseudo-labels to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMPose Teacher inference on pedestrian crops.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Filtered crop dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pseudo/nuscenes_pose_teacher"))
    parser.add_argument("--mmpose-config", required=True, help="MMPose config path or model config name.")
    parser.add_argument("--checkpoint", required=True, help="MMPose checkpoint path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_teacher(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        mmpose_config=args.mmpose_config,
        checkpoint=args.checkpoint,
        device=args.device,
        limit=args.limit,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
