"""Run the complete MLflow experiment-tracking practical."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import sklearn
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature

from hands_on.prepare_data import DATA_PATH, ROOT, prepare, sha256_file
from hands_on.train import Config, fit, tensors

EXPERIMENT_NAME = "digits-cnn"
MODEL_NAME = "digits-cnn"
SEARCH_SEED = 2026
SEEDS = (13, 42, 73)
BASE_CONFIG = Config(epochs=8)
RESULTS = ROOT / "results"


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-or-unavailable"


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


def common_tags(data_hash: str, split_seed: int) -> dict[str, str]:
    return {
        "dataset": "sklearn-load-digits",
        "dataset_sha256": data_hash,
        "split_seed": str(split_seed),
        "code_revision": code_revision(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "purpose": "teaching-experiment",
    }


def log_pytorch_model(model) -> str:
    example = np.zeros((1, 1, 8, 8), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(example)).numpy()
    model_info = mlflow.pytorch.log_model(
        model,
        name="model",
        signature=infer_signature(example, output),
        input_example=example,
        code_paths=[str(ROOT / "hands_on")],
    )
    return model_info.model_uri


def load_inference_model(model_uri: str):
    """Load either classic PyTorch pickle models or MLflow's PT2 export."""
    loaded = mlflow.pytorch.load_model(model_uri)
    if isinstance(loaded, torch.export.ExportedProgram):
        loaded = loaded.module()
    if not isinstance(loaded, torch.nn.Module):
        raise TypeError(f"Expected a PyTorch module, received {type(loaded).__name__}")
    # MLflow materializes PT2 as a GraphModule. Its exported graph already
    # captured eval-mode behavior; GraphModule.eval() is unsupported in torch.
    if not isinstance(loaded, torch.fx.GraphModule):
        loaded.eval()
    return loaded


def make_search_space() -> list[Config]:
    rows = list(itertools.product(
        (0.003, 0.01, 0.03, 0.06),
        (8, 16),
        (0.0, 0.2),
        (32, 64),
    ))
    # A fixed, seeded 20-point design across the 32 possible configurations.
    rng = np.random.default_rng(SEARCH_SEED)
    selected = sorted(rng.choice(len(rows), size=20, replace=False).tolist())
    return [
        Config(
            learning_rate=rows[i][0], channels=rows[i][1], dropout=rows[i][2],
            batch_size=rows[i][3], augmentation=True, epochs=8,
        )
        for i in selected
    ]


def run() -> dict[str, object]:
    RESULTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    train_data, val_data, test_data = load_data()
    data_hash = sha256_file(DATA_PATH)
    split_seed = 2026
    common = common_tags(data_hash, split_seed)

    baseline_cfg = Config(epochs=8)
    with mlflow.start_run(run_name="baseline-default"):
        mlflow.set_tags({**common, "run_type": "baseline"})
        mlflow.log_params(baseline_cfg.params())
        model, metrics = fit(
            train_data, val_data, baseline_cfg, 42,
            on_epoch=lambda epoch, values: [
                mlflow.log_metric(f"validation_{key}", value, step=epoch)
                for key, value in values.items()
            ],
        )
        log_pytorch_model(model)
        mlflow.log_metrics({f"best_validation_{k}": v for k, v in metrics.items()})
        baseline_metrics = metrics.copy()

    search_rows: list[dict[str, object]] = []
    space = make_search_space()
    with mlflow.start_run(run_name="random-search-20-trials") as parent:
        mlflow.set_tags({**common, "run_type": "search_parent", "trial_count": "20"})
        mlflow.log_dict([cfg.params() for cfg in space], "search_space.json")
        for trial, config in enumerate(space, 1):
            with mlflow.start_run(run_name=f"trial-{trial:02d}", nested=True):
                mlflow.set_tags({**common, "run_type": "search_trial", "trial": str(trial)})
                mlflow.log_params(config.params())
                model, metrics = fit(
                    train_data, val_data, config, seed=SEARCH_SEED,
                    on_epoch=lambda epoch, values: [
                        mlflow.log_metric(f"validation_{key}", value, step=epoch)
                        for key, value in values.items()
                    ],
                )
                log_pytorch_model(model)
                mlflow.log_metrics({f"best_validation_{k}": v for k, v in metrics.items()})
                search_rows.append({"trial": trial, **config.params(), **metrics,
                                    "run_id": mlflow.active_run().info.run_id})
    search_df = pd.DataFrame(search_rows).sort_values(
        ["accuracy", "macro_f1", "loss"], ascending=[False, False, True]
    )
    search_df.to_csv(RESULTS / "search_trials.csv", index=False)
    best_config_values = {
        key: search_df.iloc[0][key]
        for key in Config.__dataclass_fields__
    }
    best_config_values["augmentation"] = bool(best_config_values["augmentation"])
    best_config_values["epochs"] = int(best_config_values["epochs"])
    best_config_values["channels"] = int(best_config_values["channels"])
    best_config_values["batch_size"] = int(best_config_values["batch_size"])
    best_config = Config(**best_config_values)

    seed_rows = []
    seed_models = {}
    seed_model_uris = {}
    for seed in SEEDS:
        with mlflow.start_run(run_name=f"selected-config-seed-{seed}"):
            mlflow.set_tags({**common, "run_type": "seed_repetition", "seed": str(seed)})
            mlflow.log_params(best_config.params())
            model, metrics = fit(
                train_data, val_data, best_config, seed=seed,
                on_epoch=lambda epoch, values: [
                    mlflow.log_metric(f"validation_{key}", value, step=epoch)
                    for key, value in values.items()
                ],
            )
            seed_model_uris[seed] = log_pytorch_model(model)
            mlflow.log_metrics({f"best_validation_{k}": v for k, v in metrics.items()})
            seed_models[seed] = model
            seed_rows.append({"seed": seed, **metrics, "run_id": mlflow.active_run().info.run_id})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(RESULTS / "seed_repetitions.csv", index=False)

    # Matched 2x2 ablation: augmentation and dropout, each on/off.
    ablation_rows = []
    dropout_levels = (0.0, best_config.dropout if best_config.dropout > 0 else 0.25)
    for augmentation, dropout in itertools.product((False, True), dropout_levels):
        cfg = Config(**{
            **best_config.params(),
            "augmentation": augmentation,
            "dropout": dropout,
        })
        for seed in SEEDS:
            with mlflow.start_run(run_name=f"ablation-aug-{int(augmentation)}-drop-{dropout:g}-seed-{seed}"):
                mlflow.set_tags({**common, "run_type": "ablation", "seed": str(seed)})
                mlflow.log_params(cfg.params())
                model, metrics = fit(train_data, val_data, cfg, seed=seed)
                log_pytorch_model(model)
                mlflow.log_metrics({f"best_validation_{k}": v for k, v in metrics.items()})
                ablation_rows.append({"augmentation": augmentation, "dropout": dropout,
                                      "seed": seed, **metrics})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS / "ablation.csv", index=False)
    ablation_summary = ablation_df.groupby(["augmentation", "dropout"])[
        ["accuracy", "macro_f1", "loss"]
    ].agg(["mean", "std"]).reset_index()
    ablation_summary.to_csv(RESULTS / "ablation_summary.csv", index=False)

    # Lock the configuration from validation results before touching test data.
    final_seed = 42  # Predeclared before inspecting the three validation scores.
    final_model = seed_models[final_seed]
    test_metrics = __import__("hands_on.train", fromlist=["evaluate"]).evaluate(final_model, test_data)
    final_run_id = str(seed_df.loc[seed_df.seed == final_seed, "run_id"].iloc[0])
    with mlflow.start_run(run_id=final_run_id):
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.set_tag("test_evaluated_once", "true")

    model_uri = seed_model_uris[final_seed]
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    loaded = load_inference_model(f"models:/{MODEL_NAME}@champion")
    with torch.no_grad():
        sample_x, _ = val_data[0]
        prediction = int(loaded(sample_x.unsqueeze(0)).argmax(1).item())

    summary: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "dataset_sha256": data_hash,
        "split_seed": split_seed,
        "code_revision": code_revision(),
        "baseline_validation": baseline_metrics,
        "baseline_validation_accuracy": baseline_metrics["accuracy"],
        "search_trials": len(search_rows),
        "best_config": best_config.params(),
        "best_search_validation": float(search_df.iloc[0]["accuracy"]),
        "seed_metrics": seed_rows,
        "seed_validation_mean": float(seed_df.accuracy.mean()),
        "seed_validation_std": float(seed_df.accuracy.std(ddof=1)),
        "ablation_rows": len(ablation_rows),
        "test_metrics": test_metrics,
        "registered_model": MODEL_NAME,
        "registered_version": str(mv.version),
        "alias": "champion",
        "final_run_id": final_run_id,
        "sample_prediction": prediction,
        "sample_prediction_split": "validation",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from hands_on.make_report import build_report
    build_report()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
