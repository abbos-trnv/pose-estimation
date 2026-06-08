"""Create a nuScenes devkit-compatible dataroot from split archives.

The official nuScenes trainval download is often extracted as separate folders:
metadata in one directory and keyframe samples in several directories. The
devkit expects a single dataroot containing `v1.0-trainval`, `maps`, `samples`.
This script creates that layout with symlinks and does not copy the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CAMERA_CHANNELS = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)


def force_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def link_tree_files(source_dir: Path, destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        destination = destination_dir / source.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source.resolve())
        linked += 1
    return linked


def prepare_dataroot(raw_dir: Path, output_dir: Path, version: str) -> None:
    meta_dirs = sorted(raw_dir.glob(f"*meta*/{version}"))
    if len(meta_dirs) != 1:
        found = ", ".join(str(path) for path in meta_dirs) or "none"
        raise RuntimeError(f"Expected exactly one metadata directory for {version}, found: {found}")

    metadata_dir = meta_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    force_symlink(metadata_dir, output_dir / version)

    maps_dir = metadata_dir.parent / "maps"
    if maps_dir.exists():
        force_symlink(maps_dir, output_dir / "maps")

    samples_root = output_dir / "samples"
    samples_root.mkdir(exist_ok=True)

    keyframe_dirs = sorted(path for path in raw_dir.glob("*keyframes*") if path.is_dir())
    if not keyframe_dirs:
        raise RuntimeError(f"No keyframe directories found in {raw_dir}")

    total_linked = 0
    for keyframe_dir in keyframe_dirs:
        for channel in CAMERA_CHANNELS:
            source_channel_dir = keyframe_dir / "samples" / channel
            if not source_channel_dir.exists():
                continue
            total_linked += link_tree_files(source_channel_dir, samples_root / channel)

    print(f"Prepared {output_dir}")
    print(f"Metadata: {output_dir / version}")
    print(f"Linked camera images: {total_linked}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare nuScenes dataroot from split trainval archives.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataroot(raw_dir=args.raw_dir, output_dir=args.output_dir, version=args.version)


if __name__ == "__main__":
    main()
