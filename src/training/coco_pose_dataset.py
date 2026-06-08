"""COCO-like pose dataset for Student heatmap training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CocoPoseHeatmapDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        image_ids: list[int],
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        sigma: float,
    ) -> None:
        self.data_dir = Path(data_dir)
        annotations = json.loads((self.data_dir / "annotations.json").read_text(encoding="utf-8"))
        self.images_by_id = {image["id"]: image for image in annotations["images"]}
        self.annotations_by_image_id = {annotation["image_id"]: annotation for annotation in annotations["annotations"]}
        self.image_ids = image_ids
        self.image_width, self.image_height = image_size
        self.heatmap_width, self.heatmap_height = heatmap_size
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_id = self.image_ids[index]
        image_info = self.images_by_id[image_id]
        annotation = self.annotations_by_image_id[image_id]

        image_path = self.data_dir / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        source_height, source_width = image.shape[:2]
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5

        keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        scale_x = self.heatmap_width / max(float(source_width), 1.0)
        scale_y = self.heatmap_height / max(float(source_height), 1.0)
        heatmaps, weights = self._make_heatmaps(keypoints, scale_x=scale_x, scale_y=scale_y)

        sample_weight = float(annotation.get("teacher_mean_score", 1.0))
        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "weights": torch.from_numpy(weights) * sample_weight,
            "keypoints": torch.from_numpy(keypoints),
            "source_size": torch.tensor([source_width, source_height], dtype=torch.float32),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "file_name": image_info["file_name"],
        }

    def _make_heatmaps(self, keypoints: np.ndarray, scale_x: float, scale_y: float) -> tuple[np.ndarray, np.ndarray]:
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, self.heatmap_height, self.heatmap_width), dtype=np.float32)
        weights = np.zeros((num_keypoints, 1, 1), dtype=np.float32)

        for keypoint_index, (x, y, visibility) in enumerate(keypoints):
            if visibility <= 0:
                continue
            heatmap_x = x * scale_x
            heatmap_y = y * scale_y
            if heatmap_x < 0 or heatmap_y < 0 or heatmap_x >= self.heatmap_width or heatmap_y >= self.heatmap_height:
                continue
            self._draw_gaussian(heatmaps[keypoint_index], heatmap_x, heatmap_y)
            weights[keypoint_index, 0, 0] = 1.0

        return heatmaps, weights

    def _draw_gaussian(self, heatmap: np.ndarray, center_x: float, center_y: float) -> None:
        radius = int(math.ceil(self.sigma * 3))
        x0 = max(0, int(center_x) - radius)
        x1 = min(self.heatmap_width, int(center_x) + radius + 1)
        y0 = max(0, int(center_y) - radius)
        y1 = min(self.heatmap_height, int(center_y) + radius + 1)

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        patch = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * self.sigma**2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)


def load_image_ids(data_dir: str | Path) -> list[int]:
    annotations = json.loads((Path(data_dir) / "annotations.json").read_text(encoding="utf-8"))
    return [image["id"] for image in annotations["images"]]
