"""Filter Teacher pseudo-labels before building the Student dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    reasons: list[str] = []

    if float(record["mean_keypoint_score"]) < float(config["min_mean_keypoint_score"]):
        reasons.append("mean_keypoint_score_too_low")

    if int(record["num_keypoints_above_threshold"]) < int(config["min_keypoints_above_threshold"]):
        reasons.append("too_few_confident_keypoints")

    if int(record["crop_width"]) < int(config["min_crop_width"]):
        reasons.append("crop_width_too_small")

    if int(record["crop_height"]) < int(config["min_crop_height"]):
        reasons.append("crop_height_too_small")

    return reasons


def filter_pseudo_labels(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    reason_counts: dict[str, int] = {}
    for record in rejected:
        for reason in record["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "config": config,
        "num_input": len(records),
        "num_kept": len(kept),
        "num_rejected": len(rejected),
        "kept_fraction": round(len(kept) / max(len(records), 1), 4),
        "reject_reason_counts": dict(sorted(reason_counts.items())),
    }


def run_filter(labels_path: Path, output_dir: Path, config_path: Path) -> None:
    config = read_json(config_path)
    records = read_jsonl(labels_path)
    kept, rejected = filter_pseudo_labels(records, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept, output_dir / "pseudo_labels_filtered.jsonl")
    write_jsonl(rejected, output_dir / "pseudo_labels_rejected.jsonl")
    (output_dir / "pseudo_labels_filtered.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(summarize(records, kept, rejected, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Kept {len(kept)}/{len(records)} pseudo-labels in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Teacher pseudo-labels before Student training.")
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/filtering/pseudo_labels_default.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_filter(labels_path=args.labels_path, output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
