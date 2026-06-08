"""Student training entry point."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.simple_pose import SimplePoseStudent, count_parameters
from src.training.coco_pose_dataset import CocoPoseHeatmapDataset, load_image_ids
from src.training.metrics import pck_from_heatmaps


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(image_ids: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_size:], shuffled[:val_size]


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    weighted_error = squared_error * weights
    _, _, height, width = prediction.shape
    normalizer = weights.sum().clamp(min=1.0) * height * width
    return weighted_error.sum() / normalizer


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_pck = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, leave=False):
        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        weights = batch["weights"].to(device)
        target_keypoints = batch["keypoints"].to(device)
        source_sizes = batch["source_size"].to(device)

        with torch.set_grad_enabled(is_train):
            pred_heatmaps = model(images)
            loss = weighted_mse_loss(pred_heatmaps, target_heatmaps, weights)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_pck += pck_from_heatmaps(pred_heatmaps.detach(), target_keypoints, source_sizes)
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "pck": total_pck / max(num_batches, 1),
    }


def train_student(config_path: str | Path, data_dir: str | Path, output_dir: str | Path) -> None:
    config = load_config(config_path)
    data_path = Path(data_dir)
    annotations_path = data_path / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"COCO-like annotations were not found: {annotations_path}. "
            "Build data/pseudo/nuscenes_pose_coco before launching Kaggle training."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)

    image_size = tuple(config["data"]["image_size"])
    heatmap_size = tuple(config["data"]["heatmap_size"])
    image_ids = load_image_ids(data_path)
    train_ids, val_ids = split_ids(image_ids, val_fraction=float(config["training"].get("val_fraction", 0.1)), seed=seed)

    train_dataset = CocoPoseHeatmapDataset(
        data_dir=data_path,
        image_ids=train_ids,
        image_size=(int(image_size[0]), int(image_size[1])),
        heatmap_size=(int(heatmap_size[0]), int(heatmap_size[1])),
        sigma=float(config["data"].get("heatmap_sigma", 2.0)),
    )
    val_dataset = CocoPoseHeatmapDataset(
        data_dir=data_path,
        image_ids=val_ids,
        image_size=(int(image_size[0]), int(image_size[1])),
        heatmap_size=(int(heatmap_size[0]), int(heatmap_size[1])),
        sigma=float(config["data"].get("heatmap_sigma", 2.0)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseStudent(
        num_keypoints=int(config["data"]["num_keypoints"]),
        width=int(config["model"].get("width", 48)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_val_pck = -1.0
    print(f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Device: {device}")

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pck": train_metrics["pck"],
            "val_loss": val_metrics["loss"],
            "val_pck": val_metrics["pck"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, output_path / "last.pt")
        if val_metrics["pck"] > best_val_pck:
            best_val_pck = val_metrics["pck"]
            torch.save(checkpoint, output_path / "best.pt")

    (output_path / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
