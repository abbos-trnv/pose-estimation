"""Visualize Student predictions against pseudo-label targets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from src.visualization.keypoints import COCO_17_SKELETON


def load_annotations(data_dir: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    annotations = json.loads((data_dir / "annotations.json").read_text(encoding="utf-8"))
    images = annotations["images"]
    images_by_id = {image["id"]: image for image in images}
    annotations_by_image_id = {annotation["image_id"]: annotation for annotation in annotations["annotations"]}
    return images, images_by_id, annotations_by_image_id


def preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return (tensor - 0.5) / 0.5


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int], radius: int = 3) -> None:
    for start, end in COCO_17_SKELETON:
        if keypoints[start, 2] <= 0 or keypoints[end, 2] <= 0:
            continue
        p1 = (int(round(keypoints[start, 0])), int(round(keypoints[start, 1])))
        p2 = (int(round(keypoints[end, 0])), int(round(keypoints[end, 1])))
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)

    for x, y, visibility in keypoints:
        if visibility <= 0:
            continue
        cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


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


def visualize_predictions(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    tile_size: int,
    columns: int,
) -> None:
    import torch

    from src.inference.student import decode_heatmaps_to_image_points, load_student_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_student_checkpoint(checkpoint_path, device=device)
    image_size = tuple(config["data"]["image_size"])

    images, _, annotations_by_image_id = load_annotations(data_dir)
    rng = random.Random(seed)
    sampled_images = images if len(images) <= num_samples else rng.sample(images, num_samples)

    tiles: list[np.ndarray] = []
    for image_info in sampled_images:
        image_path = data_dir / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        source_height, source_width = image.shape[:2]
        input_tensor = preprocess_image(image, image_size=(int(image_size[0]), int(image_size[1]))).unsqueeze(0).to(device)
        source_sizes = torch.tensor([[source_width, source_height]], dtype=torch.float32, device=device)
        with torch.no_grad():
            heatmaps = model(input_tensor)
            pred_points = decode_heatmaps_to_image_points(heatmaps, source_sizes=source_sizes)[0].cpu().numpy()

        annotation = annotations_by_image_id[image_info["id"]]
        target_keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        pred_keypoints = np.concatenate([pred_points, np.ones((pred_points.shape[0], 1), dtype=np.float32)], axis=1)

        canvas = image.copy()
        draw_skeleton(canvas, target_keypoints, color=(0, 210, 0), radius=3)
        draw_skeleton(canvas, pred_keypoints, color=(0, 80, 255), radius=2)
        tile = resize_with_padding(canvas, tile_size)
        cv2.putText(tile, "green=target red=student", (6, tile_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        raise RuntimeError("No readable images were found for Student prediction preview.")

    rows = math.ceil(len(tiles) / columns)
    grid = np.full((rows * tile_size, columns * tile_size, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        grid[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = tile

    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / "student_predictions_grid.jpg"), grid):
        raise RuntimeError(f"Failed to write preview to {output_dir}")
    print(f"Saved Student prediction preview to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Student predictions on COCO-like pose dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa/student_predictions"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--columns", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_predictions(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        tile_size=args.tile_size,
        columns=args.columns,
    )


if __name__ == "__main__":
    main()
