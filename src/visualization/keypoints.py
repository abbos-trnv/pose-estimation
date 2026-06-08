"""Visual QA for Teacher pseudo-labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_17_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def draw_keypoints(image: np.ndarray, keypoints: list[list[float]], score_threshold: float) -> np.ndarray:
    output = image.copy()
    for start, end in COCO_17_SKELETON:
        if keypoints[start][2] < score_threshold or keypoints[end][2] < score_threshold:
            continue
        p1 = (int(round(keypoints[start][0])), int(round(keypoints[start][1])))
        p2 = (int(round(keypoints[end][0])), int(round(keypoints[end][1])))
        cv2.line(output, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)

    for x, y, score in keypoints:
        if score < score_threshold:
            continue
        cv2.circle(output, (int(round(x)), int(round(y))), 3, (30, 220, 30), -1, cv2.LINE_AA)
    return output


def resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def build_keypoint_preview(
    dataset_dir: Path,
    labels_path: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    score_threshold: float,
    tile_size: int,
    columns: int,
) -> None:
    records = read_jsonl(labels_path)
    if not records:
        raise RuntimeError(f"No pseudo-label records found: {labels_path}")

    rng = random.Random(seed)
    sampled_records = records if len(records) <= num_samples else rng.sample(records, num_samples)
    tiles: list[np.ndarray] = []

    for record in sampled_records:
        image = cv2.imread(str(dataset_dir / record["crop_path"]))
        if image is None:
            continue
        image = draw_keypoints(image, record["keypoints"], score_threshold)
        tile = resize_with_padding(image, tile_size)
        label = f"mean={record['mean_keypoint_score']:.2f}"
        cv2.putText(tile, label, (6, tile_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        raise RuntimeError("No readable crop images found for keypoint preview.")

    rows = math.ceil(len(tiles) / columns)
    grid = np.full((rows * tile_size, columns * tile_size, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        grid[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = tile

    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / "keypoint_grid.jpg"), grid):
        raise RuntimeError(f"Failed to write keypoint preview to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual QA preview for Teacher keypoints.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa/keypoints_preview"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--columns", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_keypoint_preview(
        dataset_dir=args.dataset_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        score_threshold=args.score_threshold,
        tile_size=args.tile_size,
        columns=args.columns,
    )
    print(f"Saved keypoint QA preview to {args.output_dir}")


if __name__ == "__main__":
    main()
