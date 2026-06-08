"""Extract pedestrian crops from nuScenes camera images.

The script assumes nuScenes is already downloaded locally. It uses nuScenes
3D annotations as the bbox source, projects boxes into camera frames via the
official devkit, and stores image crops plus a JSONL manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    crop_path: str
    image_path: str
    sample_token: str
    sample_data_token: str
    annotation_token: str
    scene_token: str
    channel: str
    category_name: str
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    crop_width: int
    crop_height: int
    visibility_token: str


@dataclass(frozen=True)
class ExtractionReport:
    num_samples: int
    num_crops: int
    num_missing_image_refs: int
    num_unique_missing_images: int
    missing_images: list[str]


def is_pedestrian(category_name: str, category_prefixes: tuple[str, ...]) -> bool:
    return any(category_name.startswith(prefix) for prefix in category_prefixes)


def box_to_clipped_bbox(box: Box, camera_intrinsic: list[list[float]], width: int, height: int) -> tuple[int, int, int, int] | None:
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    x1 = max(0, int(corners[0].min()))
    y1 = max(0, int(corners[1].min()))
    x2 = min(width, int(corners[0].max()))
    y2 = min(height, int(corners[1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def iter_camera_annotation_boxes(
    nusc: NuScenes,
    sample: dict,
    camera_channels: Iterable[str],
    category_prefixes: tuple[str, ...],
    box_visibility: BoxVisibility,
) -> Iterable[tuple[str, dict, str, Box, list[list[float]], str]]:
    for channel in camera_channels:
        sample_data_token = sample["data"].get(channel)
        if sample_data_token is None:
            continue

        sample_data = nusc.get("sample_data", sample_data_token)
        if not sample_data["is_key_frame"]:
            continue

        annotation_tokens = [
            token
            for token in sample["anns"]
            if is_pedestrian(nusc.get("sample_annotation", token)["category_name"], category_prefixes)
        ]
        if not annotation_tokens:
            continue

        image_path, boxes, camera_intrinsic = nusc.get_sample_data(
            sample_data_token,
            box_vis_level=box_visibility,
            selected_anntokens=annotation_tokens,
        )
        boxes_by_token = {box.token: box for box in boxes}

        for annotation_token in annotation_tokens:
            box = boxes_by_token.get(annotation_token)
            if box is None:
                continue
            annotation = nusc.get("sample_annotation", annotation_token)
            yield channel, sample_data, image_path, box, camera_intrinsic, annotation["category_name"]


def extract_crops(
    dataroot: Path,
    version: str,
    output_dir: Path,
    limit_samples: int | None,
    min_box_area: int,
    image_ext: str,
    category_prefixes: tuple[str, ...],
    missing_image_policy: str,
) -> tuple[list[CropRecord], ExtractionReport]:
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    records: list[CropRecord] = []
    missing_images: list[str] = []
    samples = nusc.sample[:limit_samples] if limit_samples is not None else nusc.sample

    for sample_index, sample in enumerate(samples):
        for channel, sample_data, image_path, box, camera_intrinsic, category_name in iter_camera_annotation_boxes(
            nusc=nusc,
            sample=sample,
            camera_channels=CAMERA_CHANNELS,
            category_prefixes=category_prefixes,
            box_visibility=BoxVisibility.ANY,
        ):
            image_rel_path = str(Path(image_path).relative_to(dataroot))
            if not Path(image_path).exists():
                if missing_image_policy == "error":
                    raise RuntimeError(f"Image file does not exist: {image_path}")
                missing_images.append(image_rel_path)
                continue

            image = cv2.imread(image_path)
            if image is None:
                if missing_image_policy == "error":
                    raise RuntimeError(f"Failed to read image: {image_path}")
                missing_images.append(image_rel_path)
                continue

            height, width = image.shape[:2]
            bbox = box_to_clipped_bbox(box, camera_intrinsic, width=width, height=height)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width * bbox_height < min_box_area:
                continue

            crop = image[y1:y2, x1:x2]
            crop_id = f"{sample['token']}_{channel}_{box.token}"
            crop_rel_path = Path("crops") / f"{crop_id}.{image_ext}"
            crop_path = output_dir / crop_rel_path
            ok = cv2.imwrite(str(crop_path), crop)
            if not ok:
                raise RuntimeError(f"Failed to write crop: {crop_path}")

            records.append(
                CropRecord(
                    crop_id=crop_id,
                    crop_path=str(crop_rel_path),
                    image_path=image_rel_path,
                    sample_token=sample["token"],
                    sample_data_token=sample_data["token"],
                    annotation_token=box.token,
                    scene_token=sample["scene_token"],
                    channel=channel,
                    category_name=category_name,
                    bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    bbox_xywh=[float(x1), float(y1), float(bbox_width), float(bbox_height)],
                    crop_width=int(bbox_width),
                    crop_height=int(bbox_height),
                    visibility_token=nusc.get("sample_annotation", box.token)["visibility_token"],
                )
            )

        if sample_index > 0 and sample_index % 100 == 0:
            print(f"Processed {sample_index}/{len(samples)} samples, crops={len(records)}")

    unique_missing_images = sorted(set(missing_images))
    report = ExtractionReport(
        num_samples=len(samples),
        num_crops=len(records),
        num_missing_image_refs=len(missing_images),
        num_unique_missing_images=len(unique_missing_images),
        missing_images=unique_missing_images[:1000],
    )
    return records, report


def write_manifest(records: list[CropRecord], report: ExtractionReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "manifest.jsonl"
    json_path = output_dir / "manifest.json"
    report_path = output_dir / "extraction_report.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(record) for record in records], file, ensure_ascii=False, indent=2)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(report), file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pedestrian crops from nuScenes camera frames.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/raw/nuscenes"), help="Path to nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version, e.g. v1.0-mini or v1.0-trainval.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/nuscenes_pedestrian_crops"))
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--min-box-area", type=int, default=32 * 32)
    parser.add_argument("--image-ext", choices=("jpg", "png"), default="jpg")
    parser.add_argument(
        "--missing-image-policy",
        choices=("skip", "error"),
        default="skip",
        help="What to do when metadata references an image that is not present in dataroot.",
    )
    parser.add_argument(
        "--category-prefix",
        action="append",
        default=["human.pedestrian"],
        help="Category prefix to keep. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, report = extract_crops(
        dataroot=args.dataroot,
        version=args.version,
        output_dir=args.output_dir,
        limit_samples=args.limit_samples,
        min_box_area=args.min_box_area,
        image_ext=args.image_ext,
        category_prefixes=tuple(args.category_prefix),
        missing_image_policy=args.missing_image_policy,
    )
    write_manifest(records, report, args.output_dir)
    print(f"Saved {len(records)} crops to {args.output_dir}")
    if report.num_missing_image_refs:
        print(
            "Skipped "
            f"{report.num_missing_image_refs} missing image references "
            f"({report.num_unique_missing_images} unique files). "
            "See extraction_report.json"
        )


if __name__ == "__main__":
    main()
