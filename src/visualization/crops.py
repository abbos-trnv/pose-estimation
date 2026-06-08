"""Visual QA for pedestrian crops and projected bounding boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resize_with_padding(image: np.ndarray, size: int, background: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), background, dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def put_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, image.shape[0] - 24), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, dst=output)
    cv2.putText(output, label, (6, image.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def make_crop_grid(records: list[dict[str, Any]], dataset_dir: Path, output_path: Path, tile_size: int, columns: int) -> None:
    tiles: list[np.ndarray] = []
    for record in records:
        crop_path = dataset_dir / record["crop_path"]
        image = cv2.imread(str(crop_path))
        if image is None:
            continue
        tile = resize_with_padding(image, tile_size)
        label = f"{record['channel']} {record['crop_width']}x{record['crop_height']} v{record['visibility_token']}"
        tiles.append(put_label(tile, label))

    if not tiles:
        raise RuntimeError("No readable crop images found for preview.")

    rows = math.ceil(len(tiles) / columns)
    grid = np.full((rows * tile_size, columns * tile_size, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        grid[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid):
        raise RuntimeError(f"Failed to write crop grid: {output_path}")


def make_bbox_previews(records: list[dict[str, Any]], dataroot: Path, output_dir: Path, max_images: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_image: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_image.setdefault(record["image_path"], []).append(record)

    for index, (image_rel_path, image_records) in enumerate(records_by_image.items()):
        if index >= max_images:
            break
        image_path = dataroot / image_rel_path
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for record in image_records:
            x1, y1, x2, y2 = [int(round(value)) for value in record["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
            label = f"{record['category_name'].replace('human.pedestrian.', '')} v{record['visibility_token']}"
            cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        safe_name = image_rel_path.replace("/", "__")
        output_path = output_dir / safe_name
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write bbox preview: {output_path}")


def write_report(records: list[dict[str, Any]], sampled_records: list[dict[str, Any]], output_path: Path) -> None:
    widths = np.array([record["crop_width"] for record in records], dtype=np.float32)
    heights = np.array([record["crop_height"] for record in records], dtype=np.float32)
    areas = widths * heights
    visibility_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    for record in records:
        visibility_counts[record["visibility_token"]] = visibility_counts.get(record["visibility_token"], 0) + 1
        channel_counts[record["channel"]] = channel_counts.get(record["channel"], 0) + 1

    report = {
        "num_records": len(records),
        "num_sampled_records": len(sampled_records),
        "crop_width": {
            "min": float(widths.min()),
            "median": float(np.median(widths)),
            "max": float(widths.max()),
        },
        "crop_height": {
            "min": float(heights.min()),
            "median": float(np.median(heights)),
            "max": float(heights.max()),
        },
        "crop_area": {
            "min": float(areas.min()),
            "median": float(np.median(areas)),
            "max": float(areas.max()),
        },
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_preview(
    dataset_dir: Path,
    dataroot: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    tile_size: int,
    columns: int,
    max_bbox_images: int,
) -> None:
    records = read_jsonl(dataset_dir / "manifest.jsonl")
    if not records:
        raise RuntimeError(f"Manifest is empty: {dataset_dir / 'manifest.jsonl'}")

    rng = random.Random(seed)
    sampled_records = records if len(records) <= num_samples else rng.sample(records, num_samples)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_crop_grid(
        records=sampled_records,
        dataset_dir=dataset_dir,
        output_path=output_dir / "crop_grid.jpg",
        tile_size=tile_size,
        columns=columns,
    )
    make_bbox_previews(
        records=sampled_records,
        dataroot=dataroot,
        output_dir=output_dir / "bbox_on_source",
        max_images=max_bbox_images,
    )
    write_report(records=records, sampled_records=sampled_records, output_path=output_dir / "report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual QA previews for pedestrian crop dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with crops/ and manifest.jsonl.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/raw/nuscenes"), help="nuScenes dataroot for source images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa/crops_preview"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--max-bbox-images", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_preview(
        dataset_dir=args.dataset_dir,
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        tile_size=args.tile_size,
        columns=args.columns,
        max_bbox_images=args.max_bbox_images,
    )
    print(f"Saved crop QA preview to {args.output_dir}")


if __name__ == "__main__":
    main()
