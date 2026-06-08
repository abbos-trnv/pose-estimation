"""Filter pedestrian crop manifests before Teacher pseudo-labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def reject_reasons(record: dict[str, Any], config: dict[str, Any]) -> list[str]:
    width = float(record["crop_width"])
    height = float(record["crop_height"])
    area = width * height
    aspect_ratio = width / max(height, 1.0)
    reasons: list[str] = []

    if width < float(config["min_width"]):
        reasons.append("width_too_small")
    if height < float(config["min_height"]):
        reasons.append("height_too_small")
    if area < float(config["min_area"]):
        reasons.append("area_too_small")
    if aspect_ratio < float(config["min_aspect_ratio"]):
        reasons.append("aspect_ratio_too_small")
    if aspect_ratio > float(config["max_aspect_ratio"]):
        reasons.append("aspect_ratio_too_large")
    if record["visibility_token"] not in set(config["allowed_visibility_tokens"]):
        reasons.append("visibility_not_allowed")

    return reasons


def filter_records(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in records:
        reasons = reject_reasons(record, config)
        if reasons:
            rejected_record = dict(record)
            rejected_record["reject_reasons"] = reasons
            rejected.append(rejected_record)
        else:
            kept.append(record)

    return kept, rejected


def summarize(records: list[dict[str, Any]], kept: list[dict[str, Any]], rejected: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    widths = np.array([record["crop_width"] for record in records], dtype=np.float32)
    heights = np.array([record["crop_height"] for record in records], dtype=np.float32)
    reason_counts: dict[str, int] = {}
    for record in rejected:
        for reason in record["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": 0.0, "median": 0.0, "max": 0.0}
        return {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }

    return {
        "config": config,
        "num_input": len(records),
        "num_kept": len(kept),
        "num_rejected": len(rejected),
        "kept_fraction": round(len(kept) / max(len(records), 1), 4),
        "input_width": stats(widths),
        "input_height": stats(heights),
        "reject_reason_counts": dict(sorted(reason_counts.items())),
    }


def copy_kept_crops(kept: list[dict[str, Any]], input_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    import shutil

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    copied_records: list[dict[str, Any]] = []
    for record in kept:
        source_path = input_dir / record["crop_path"]
        destination_rel_path = Path("crops") / Path(record["crop_path"]).name
        destination_path = output_dir / destination_rel_path
        shutil.copy2(source_path, destination_path)

        copied_record = dict(record)
        copied_record["crop_path"] = str(destination_rel_path)
        copied_records.append(copied_record)

    return copied_records


def run_filter(input_dir: Path, output_dir: Path, config_path: Path, copy_crops: bool) -> None:
    config = read_json(config_path)
    records = read_jsonl(input_dir / "manifest.jsonl")
    kept, rejected = filter_records(records, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_records = copy_kept_crops(kept, input_dir=input_dir, output_dir=output_dir) if copy_crops else kept

    write_jsonl(output_records, output_dir / "manifest.jsonl")
    write_jsonl(rejected, output_dir / "rejected.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(output_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(summarize(records, kept, rejected, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Kept {len(kept)}/{len(records)} records in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter pedestrian crop manifest before Teacher inference.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/filtering/crops_default.json"))
    parser.add_argument("--copy-crops", action="store_true", help="Copy kept crop images into the filtered dataset directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_filter(input_dir=args.input_dir, output_dir=args.output_dir, config_path=args.config, copy_crops=args.copy_crops)


if __name__ == "__main__":
    main()
