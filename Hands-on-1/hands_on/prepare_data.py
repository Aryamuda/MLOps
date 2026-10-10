"""Create the deterministic, versioned digits dataset and fixed data splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "digits.npz"
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPLIT_SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare() -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bunch = load_digits()
    images = bunch.images.astype(np.float32) / 16.0
    labels = bunch.target.astype(np.int64)
    indices = np.arange(labels.size)
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.30, random_state=SPLIT_SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=SPLIT_SEED,
        stratify=labels[remainder_idx],
    )
    np.savez_compressed(
        DATA_PATH,
        images=images,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    manifest = {
        "dataset": "scikit-learn load_digits",
        "source": "scikit-learn built-in dataset (8x8 grayscale, 10 classes)",
        "n_samples": int(labels.size),
        "image_shape": [8, 8],
        "normalization": "pixel values divided by 16 into [0, 1]",
        "split_seed": SPLIT_SEED,
        "split_counts": {
            "train": int(train_idx.size),
            "validation": int(val_idx.size),
            "test": int(test_idx.size),
        },
        "sha256": sha256_file(DATA_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
