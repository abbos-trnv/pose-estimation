"""Training metrics for heatmap pose models."""

from __future__ import annotations

import torch


def heatmap_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    batch_size, num_keypoints, height, width = heatmaps.shape
    flat_indices = heatmaps.reshape(batch_size, num_keypoints, -1).argmax(dim=-1)
    xs = flat_indices % width
    ys = flat_indices // width
    return torch.stack([xs.float(), ys.float()], dim=-1)


def pck_from_heatmaps(pred_heatmaps: torch.Tensor, target_keypoints: torch.Tensor, source_sizes: torch.Tensor, threshold: float = 0.2) -> float:
    _, _, heatmap_height, heatmap_width = pred_heatmaps.shape
    pred_points = heatmap_argmax(pred_heatmaps)

    scale = torch.stack(
        [
            torch.full_like(source_sizes[:, 0], heatmap_width) / source_sizes[:, 0].clamp(min=1),
            torch.full_like(source_sizes[:, 1], heatmap_height) / source_sizes[:, 1].clamp(min=1),
        ],
        dim=-1,
    )
    target_points = target_keypoints[..., :2] * scale[:, None, :]
    visible = target_keypoints[..., 2] > 0
    distances = torch.linalg.norm(pred_points - target_points, dim=-1)
    norm = torch.minimum(source_sizes[:, 0], source_sizes[:, 1]).clamp(min=1) / 4.0
    norm = norm[:, None]
    correct = (distances <= threshold * norm) & visible
    return float(correct.sum().item() / visible.sum().clamp(min=1).item())
