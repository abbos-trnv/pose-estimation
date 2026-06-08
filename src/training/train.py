"""Student training entry point.

The implementation is intentionally guarded for now: the training loop should
be finalized after Teacher pseudo-labels and the COCO-like dataset are verified.
This keeps the Kaggle notebook interface stable without pretending that the
Student baseline has already been trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def train_student(config_path: str | Path, data_dir: str | Path, output_dir: str | Path) -> None:
    config = load_config(config_path)
    data_path = Path(data_dir)
    annotations_path = data_path / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"COCO-like annotations were not found: {annotations_path}. "
            "Build data/pseudo/nuscenes_pose_coco before launching Kaggle training."
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Student training loop is not implemented yet. "
        f"Config '{config.get('experiment_name', config_path)}' and dataset '{data_path}' were validated."
    )
