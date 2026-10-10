"""Run the full CPU experiment plan when MLflow/DVC cannot be installed yet.

This runner produces real model metrics and CSV artifacts. It does not claim to
create an MLflow server, registry alias, or DVC cache; run the MLflow workflow
after installing project dependencies to complete those platform steps.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from hands_on.prepare_data import DATA_PATH, ROOT, prepare, sha256_file
from hands_on.train import Config, evaluate, fit, tensors

RESULTS = ROOT / "results"
SEEDS = (13, 42, 73)


def load_data():
    if not DATA_PATH.exists():
        prepare()
    data = np.load(DATA_PATH)
    images, labels = data["images"], data["labels"]
    return (
        tensors(images[data["train_idx"]], labels[data["train_idx"]]),
        tensors(images[data["val_idx"]], labels[data["val_idx"]]),
        tensors(images[data["test_idx"]], labels[data["test_idx"]]),
    )


def main():
    RESULTS.mkdir(exist_ok=True)
    train_data, val_data, test_data = load_data()
    baseline_cfg = Config(epochs=8)
    baseline_model, baseline_metrics = fit(train_data, val_data, baseline_cfg, 42)

    configs = list(itertools.product((0.003, 0.01, 0.03, 0.06), (8, 16), (0.0, 0.2), (32, 64)))
    rng = np.random.default_rng(2026)
    selected = sorted(rng.choice(len(configs), size=20, replace=False).tolist())
    trial_rows = []
    for trial, index in enumerate(selected, 1):
        lr, channels, dropout, batch_size = configs[index]
        config = Config(learning_rate=lr, channels=channels, dropout=dropout,
                        batch_size=batch_size, augmentation=True, epochs=8)
        _, metrics = fit(train_data, val_data, config, 2026)
        trial_rows.append({"trial": trial, **config.params(), **metrics})
        print(f"trial {trial:02d}/20 val_accuracy={metrics['accuracy']:.4f}", flush=True)
    trial_df = pd.DataFrame(trial_rows).sort_values(
        ["accuracy", "macro_f1", "loss"], ascending=[False, False, True]
    )
    trial_df.to_csv(RESULTS / "search_trials.csv", index=False)
    row = trial_df.iloc[0]
    best = Config(learning_rate=float(row.learning_rate), channels=int(row.channels),
                  dropout=float(row.dropout), batch_size=int(row.batch_size),
                  augmentation=bool(row.augmentation), epochs=int(row.epochs))

    seed_rows = []
    final_models = {}
    for seed in SEEDS:
        model, metrics = fit(train_data, val_data, best, seed)
        final_models[seed] = model
        seed_rows.append({"seed": seed, **metrics})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(RESULTS / "seed_repetitions.csv", index=False)

    ablation_rows = []
    dropout_levels = (0.0, best.dropout if best.dropout > 0 else 0.25)
    for augmentation, dropout in itertools.product((False, True), dropout_levels):
        config = Config(**{**best.params(), "augmentation": augmentation, "dropout": dropout})
        for seed in SEEDS:
            _, metrics = fit(train_data, val_data, config, seed)
            ablation_rows.append({"augmentation": augmentation, "dropout": dropout,
                                  "seed": seed, **metrics})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS / "ablation.csv", index=False)
    (ablation_df.groupby(["augmentation", "dropout"])[["accuracy", "macro_f1", "loss"]]
     .agg(["mean", "std"]).reset_index().to_csv(RESULTS / "ablation_summary.csv", index=False))

    # Seed 42 is the predeclared final fit; test metrics are computed once.
    final_model = final_models[42]
    test_metrics = evaluate(final_model, test_data)
    summary = {
        "execution_mode": "local-training-only",
        "dataset_sha256": sha256_file(DATA_PATH),
        "split_seed": 2026,
        "baseline_validation": baseline_metrics,
        "baseline_validation_accuracy": baseline_metrics["accuracy"],
        "search_trials": len(trial_rows),
        "best_config": best.params(),
        "best_search_validation": float(trial_df.iloc[0].accuracy),
        "seed_metrics": seed_rows,
        "seed_validation_mean": float(seed_df.accuracy.mean()),
        "seed_validation_std": float(seed_df.accuracy.std(ddof=1)),
        "ablation_rows": len(ablation_rows),
        "test_metrics": test_metrics,
        "registered_model": None,
        "registered_version": None,
        "alias": None,
        "final_run_id": None,
        "sample_prediction": int(final_model(test_data[0][0].unsqueeze(0)).argmax(1).item()),
        "environment": {"python": __import__("platform").python_version(),
                        "torch": torch.__version__, "scikit_learn": sklearn.__version__},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from hands_on.make_report import build_report
    build_report()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
