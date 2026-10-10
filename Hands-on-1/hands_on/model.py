"""Small CNN used throughout the exercise."""

from __future__ import annotations

import torch
from torch import nn


class DigitsCNN(nn.Module):
    def __init__(self, channels: int = 16, dropout: float = 0.15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2 * 2 * 2, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)
