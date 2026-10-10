# Hands-on 1 — MLflow Experiment Tracking

An end-to-end, CPU-friendly image classification exercise using scikit-learn's
built-in handwritten digits dataset (1,797 grayscale images, 8×8 pixels).
The dataset is materialized locally, split once with a fixed seed, and tracked
with DVC. Training uses a small PyTorch CNN and a local MLflow server with a
SQLite backend and file artifacts.

## Learning objective

Compare controlled CNN experiments, select a candidate using validation data,
estimate seed variability, inspect a two-factor ablation, then register and
load the chosen model through the `@champion` alias. The test split is reserved
for the final evaluation and is not used to choose hyperparameters.

## Setup

Python 3.11–3.14 is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Generate the fixed dataset (already included in this repository) and inspect
its DVC pointer:

```bash
python -m hands_on.prepare_data
export DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache
dvc status
dvc push
```

`data/digits.npz` is small and is included in this repository for easy first
use. The accompanying `.dvc` file records its content hash. A local DVC remote
is configured at `dvc-storage/`; `dvc push` copies the version into that local
remote. To version a changed dataset, regenerate it, run `dvc add data/digits.npz`,
and commit both the resulting pointer and updated ignore rule.

## Run the exercise

Terminal 1, start the tracking server:

```bash
./scripts/start_mlflow.sh
```

Terminal 2, run the complete experiment sequence:

```bash
source .venv/bin/activate
python -m hands_on.run_experiments
```

This logs the baseline, 20 search trials, three final seeds, four matched
ablation configurations, test evaluation, model registration and inference
through `models:/digits-cnn@champion`. Results are written to `results/`.
`python -m hands_on.make_report` builds `REPORT.md`. To create the Indonesian
submission PDF, run:

```bash
pandoc REPORT.md -o 123450063_Arya_Muda_Siregar_MLOps6.pdf --pdf-engine=xelatex -V lang=id -V papersize:a4 -V geometry:margin=0.7in -V fontsize=10pt
```

Markdown and PDF files other than `README.md` are ignored by Git, so generated
submission reports stay local.

If the run already registered version 1 but stopped during the alias inference
check with a PyTorch `GraphModule.eval()` error, finish from the existing run
data without retraining or reusing the test split:

```bash
python -m hands_on.finalize_existing
```

Open <http://127.0.0.1:5000> and use the **digits-cnn** experiment's Compare
view to capture `images/mlflow-compare.png`. Also capture the registered model
page showing the `champion` alias as `images/model-registry.png`. Add those
screenshots to this folder; the report also includes the supplied parameter,
metric, and tag views as `mlflow-compare-params-metrics.png` and
`mlflow-compare-tags.png`.

The server uses SQLite for the backend and a local artifact directory. MLflow's
registry requires a database-backed store; this setup follows that requirement
and uses a model alias for promotion. See the [MLflow backend store docs](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
and [model registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/).

Stop the server with Ctrl+C in Terminal 1. Tracking data and artifacts stay in
`mlflow.db` and `mlartifacts/`.

## Reproduce / inspect

```bash
python -m hands_on.run_experiments --help
dvc status
python -m hands_on.make_report
```

The split and search space are committed in code. Every run records the data
SHA-256, split seed, Python and library versions, configuration, and code
revision when available. Results are educational demonstrations on a tiny
dataset; they do not establish production readiness.

## Deliverables

- Git repository containing source, fixed dataset, DVC metadata, MLflow local
  state, and result tables.
- MLflow compare-view and model-registry screenshots in `images/`.
- Local `REPORT.md` and `123450063_Arya_Muda_Siregar_MLOps6.pdf` with the
  hypothesis, observed results, and conclusion.
