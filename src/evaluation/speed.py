"""Benchmark Student model latency and FPS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def benchmark_speed(
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> None:
    import torch

    from src.inference.student import load_student_checkpoint
    from src.models.simple_pose import count_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_student_checkpoint(checkpoint_path, device=device)
    image_width, image_height = config["data"]["image_size"]
    inputs = torch.randn(batch_size, 3, image_height, image_width, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_images = batch_size * iterations
    latency_ms_per_image = elapsed / total_images * 1000.0
    fps = total_images / elapsed
    report = {
        "device": str(device),
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "num_parameters": count_parameters(model),
        "image_size": [image_width, image_height],
        "latency_ms_per_image": latency_ms_per_image,
        "fps": fps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Student model latency/FPS.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("logs/student_speed.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_speed(
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
