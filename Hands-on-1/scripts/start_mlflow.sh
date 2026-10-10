#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --default-artifact-root "file://$PWD/mlartifacts" \
  --host 127.0.0.1 --port 5000
