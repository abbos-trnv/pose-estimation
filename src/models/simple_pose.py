"""Small heatmap-based Student pose model."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimplePoseStudent(nn.Module):
    """Compact CNN that outputs keypoint heatmaps at 1/4 input resolution."""

    def __init__(self, num_keypoints: int = 17, width: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(width, width, stride=1),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
        )
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(width * 2, width * 2),
            DepthwiseSeparableBlock(width * 2, width * 3),
            DepthwiseSeparableBlock(width * 3, width * 3),
            DepthwiseSeparableBlock(width * 3, width * 2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 2, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
