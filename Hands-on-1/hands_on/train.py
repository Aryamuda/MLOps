"""Training utilities with deterministic seeds and lightweight augmentation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hands_on.model import DigitsCNN


@dataclass(frozen=True)
class Config:
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    channels: int = 16
    dropout: float = 0.15
    batch_size: int = 64
    augmentation: bool = True
    epochs: int = 8

    def params(self) -> dict[str, int | float | bool]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def tensors(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(images[:, None].astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x, y)


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply mild per-image translation using zero-filled integer shifts."""
    batch_size, channels, height, width = batch.shape
    dy = torch.randint(-1, 2, (batch_size, 1, 1), generator=generator)
    dx = torch.randint(-1, 2, (batch_size, 1, 1), generator=generator)
    yy = torch.arange(height).view(1, height, 1) - dy
    xx = torch.arange(width).view(1, 1, width) - dx
    valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    flat_indices = (yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1))
    flat_indices = flat_indices.expand(batch_size, height, width).reshape(batch_size, -1)
    flat = batch.reshape(batch_size, channels, height * width)
    result = flat.gather(2, flat_indices[:, None, :].expand(-1, channels, -1))
    return result.reshape(batch_size, channels, height, width) * valid[:, None, :, :]


def evaluate(model: nn.Module, dataset: TensorDataset) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += float(criterion(logits, y).item()) * len(y)
            count += len(y)
            predictions.append(logits.argmax(1).numpy())
            targets.append(y.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": loss_sum / count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def fit(
    train_data: TensorDataset,
    validation_data: TensorDataset,
    config: Config,
    seed: int,
    on_epoch=None,
) -> tuple[DigitsCNN, dict[str, float]]:
    seed_everything(seed)
    model = DigitsCNN(config.channels, config.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    aug_generator = torch.Generator().manual_seed(seed + 17)
    best_state = None
    best_metrics: dict[str, float] | None = None
    for epoch in range(config.epochs):
        model.train()
        for x, y in loader:
            if config.augmentation:
                x = augment(x, aug_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_data)
        if on_epoch:
            on_epoch(epoch + 1, metrics)
        if best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics
