"""HuggingFace ViTPose Teacher inference for pedestrian crops.

This avoids OpenMMLab/MMCV installation issues in Kaggle Python 3.12.
The model is still a top-down pose estimator: each crop is treated as a single
person image with one full-crop bounding box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.teacher.mmpose_teacher import COCO_17_KEYPOINTS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_hf_vitpose(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace ViTPose dependencies are not installed. "
            "Install transformers, torch and pillow before running Teacher inference."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model, torch


def extract_prediction(result: Any) -> tuple[list[list[float]], list[float]]:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("ViTPose returned an empty prediction list.")
        result = result[0]

    keypoints = result.get("keypoints")
    scores = result.get("scores", result.get("keypoint_scores"))
    if keypoints is None or scores is None:
        raise RuntimeError(f"Unexpected ViTPose post-process result keys: {list(result.keys())}")

    keypoints_array = np.asarray(keypoints, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    if keypoints_array.ndim == 3:
        keypoints_array = keypoints_array[0]
    if scores_array.ndim == 2:
        scores_array = scores_array[0]

    keypoints_with_scores = [
        [float(x), float(y), float(score)]
        for (x, y), score in zip(keypoints_array[:, :2], scores_array, strict=True)
    ]
    return keypoints_with_scores, [float(score) for score in scores_array]


def predict_crop(processor: Any, model: Any, torch: Any, crop_path: Path, device: str) -> dict[str, Any]:
    image = Image.open(crop_path).convert("RGB")
    width, height = image.size
    boxes = [[[0.0, 0.0, float(width), float(height)]]]

    inputs = processor(image, boxes=boxes, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(processor, "post_process_pose_estimation"):
        results = processor.post_process_pose_estimation(outputs, boxes=boxes)
    else:
        target_sizes = torch.tensor([[height, width]], device=device)
        results = processor.post_process_pose_estimation(outputs, target_sizes=target_sizes)

    keypoints, scores = extract_prediction(results[0])
    score_array = np.asarray(scores, dtype=np.float32)
    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "mean_keypoint_score": float(score_array.mean()),
        "min_keypoint_score": float(score_array.min()),
        "num_keypoints": int(len(scores)),
    }


def run_teacher(
    dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    limit: int | None,
    score_threshold: float,
) -> None:
    records = read_jsonl(dataset_dir / "manifest.jsonl")
    if limit is not None:
        records = records[:limit]

    processor, model, torch = load_hf_vitpose(model_name=model_name, device=device)
    pseudo_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        prediction = predict_crop(processor, model, torch, dataset_dir / record["crop_path"], device=device)
        keypoint_scores = prediction["keypoint_scores"]
        pseudo_records.append(
            {
                **record,
                "teacher": {
                    "framework": "huggingface-transformers",
                    "model_name": model_name,
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
        if index % 50 == 0:
            print(f"HF ViTPose inference: {index}/{len(records)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pseudo_records, output_dir / "pseudo_labels.jsonl")
    (output_dir / "pseudo_labels.json").write_text(json.dumps(pseudo_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(pseudo_records)} pseudo-labels to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HuggingFace ViTPose Teacher inference on pedestrian crops.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pseudo/nuscenes_pose_teacher"))
    parser.add_argument("--model-name", default="usyd-community/vitpose-plus-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_teacher(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        limit=args.limit,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
