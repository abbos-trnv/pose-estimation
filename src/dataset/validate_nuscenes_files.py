"""Validate local nuScenes camera file coverage against metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CAMERA_CHANNELS = {
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
}


def validate_files(dataroot: Path, version: str, output_path: Path) -> None:
    sample_data_path = dataroot / version / "sample_data.json"
    records = json.loads(sample_data_path.read_text(encoding="utf-8"))

    total_keyframe_camera = 0
    present = 0
    missing_by_channel: dict[str, int] = {}
    present_by_channel: dict[str, int] = {}
    missing_examples: list[str] = []

    for record in records:
        filename = record["filename"]
        parts = filename.split("/")
        if len(parts) < 3 or parts[0] != "samples" or parts[1] not in CAMERA_CHANNELS:
            continue
        if not record["is_key_frame"]:
            continue

        total_keyframe_camera += 1
        channel = parts[1]
        if (dataroot / filename).exists():
            present += 1
            present_by_channel[channel] = present_by_channel.get(channel, 0) + 1
        else:
            missing_by_channel[channel] = missing_by_channel.get(channel, 0) + 1
            if len(missing_examples) < 1000:
                missing_examples.append(filename)

    report: dict[str, Any] = {
        "dataroot": str(dataroot),
        "version": version,
        "total_keyframe_camera_images": total_keyframe_camera,
        "present_keyframe_camera_images": present,
        "missing_keyframe_camera_images": total_keyframe_camera - present,
        "present_fraction": round(present / max(total_keyframe_camera, 1), 4),
        "present_by_channel": dict(sorted(present_by_channel.items())),
        "missing_by_channel": dict(sorted(missing_by_channel.items())),
        "missing_examples": missing_examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "missing_examples"}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local nuScenes camera file coverage.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/raw/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-path", type=Path, default=Path("data/qa/nuscenes_file_coverage.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_files(dataroot=args.dataroot, version=args.version, output_path=args.output_path)


if __name__ == "__main__":
    main()
