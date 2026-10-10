"""Finish report and inference verification after a run registered its model.

Use this when the training run completed but stopped at the PT2 model loading
step. It reuses the existing run data and does not retrain or re-score the test
split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient

from hands_on.prepare_data import DATA_PATH, ROOT, sha256_file
from hands_on.train import tensors

RESULTS = ROOT / "results"
EXPERIMENT_NAME = "digits-cnn"
MODEL_NAME = "digits-cnn"


def main() -> dict[str, object]:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT_NAME!r} was not found")

    seed_df = pd.read_csv(RESULTS / "seed_repetitions.csv")
    search_df = pd.read_csv(RESULTS / "search_trials.csv")
    final_run_id = str(seed_df.loc[seed_df.seed == 42, "run_id"].iloc[-1])
    final_run = client.get_run(final_run_id)
    metrics = final_run.data.metrics

    baseline_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.run_type = 'baseline'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not baseline_runs:
        raise RuntimeError("No baseline run was found in the MLflow experiment")
    baseline = baseline_runs[0].data.metrics

    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    loaded = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
    if isinstance(loaded, torch.export.ExportedProgram):
        loaded = loaded.module()
    if not isinstance(loaded, torch.nn.Module):
        raise TypeError(f"Expected a PyTorch module, received {type(loaded).__name__}")
    if not isinstance(loaded, torch.fx.GraphModule):
        loaded.eval()

    data = np.load(DATA_PATH)
    val = tensors(data["images"][data["val_idx"]], data["labels"][data["val_idx"]])
    with torch.no_grad():
        prediction = int(loaded(val[0][0].unsqueeze(0)).argmax(1).item())

    best = search_df.sort_values("accuracy", ascending=False).iloc[0]
    config = {
        "learning_rate": float(best.learning_rate),
        "weight_decay": float(best.weight_decay),
        "channels": int(best.channels),
        "dropout": float(best.dropout),
        "batch_size": int(best.batch_size),
        "augmentation": bool(best.augmentation),
        "epochs": int(best.epochs),
    }
    summary: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "dataset_sha256": sha256_file(DATA_PATH),
        "split_seed": 2026,
        "code_revision": final_run.data.tags.get("code_revision"),
        "baseline_validation": {
            key.removeprefix("best_validation_"): value
            for key, value in baseline.items()
            if key.startswith("best_validation_")
        },
        "baseline_validation_accuracy": baseline["best_validation_accuracy"],
        "search_trials": int(len(search_df)),
        "best_config": config,
        "best_search_validation": float(best.accuracy),
        "seed_metrics": seed_df.to_dict(orient="records"),
        "seed_validation_mean": float(seed_df.accuracy.mean()),
        "seed_validation_std": float(seed_df.accuracy.std(ddof=1)),
        "ablation_rows": int(len(pd.read_csv(RESULTS / "ablation.csv"))),
        "test_metrics": {
            key.removeprefix("test_"): value
            for key, value in metrics.items()
            if key.startswith("test_")
        },
        "registered_model": MODEL_NAME,
        "registered_version": str(version.version),
        "alias": "champion",
        "final_run_id": final_run_id,
        "sample_prediction": prediction,
        "sample_prediction_split": "validation",
    }
    if not summary["test_metrics"]:
        raise RuntimeError("The selected run has no test metrics; refusing to invent them")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from hands_on.make_report import build_report
    build_report()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
