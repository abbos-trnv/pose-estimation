"""Student checkpoint loading and heatmap decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.simple_pose import SimplePoseStudent
from src.training.metrics import heatmap_argmax


def load_student_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[SimplePoseStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SimplePoseStudent(
        num_keypoints=int(config["data"]["num_keypoints"]),
        width=int(config["model"].get("width", 48)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def decode_heatmaps_to_image_points(heatmaps: torch.Tensor, source_sizes: torch.Tensor) -> torch.Tensor:
    points = heatmap_argmax(heatmaps)
    _, _, heatmap_height, heatmap_width = heatmaps.shape
    scale = torch.stack(
        [
            source_sizes[:, 0].clamp(min=1) / heatmap_width,
            source_sizes[:, 1].clamp(min=1) / heatmap_height,
        ],
        dim=-1,
    )
    return points * scale[:, None, :]
