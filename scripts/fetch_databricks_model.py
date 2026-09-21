#!/usr/bin/env python3
"""Downloads a Laya model version from Unity Catalog into runs/laya_databricks.

The test rig can talk to a Databricks serving endpoint, but a GPU endpoint costs
money while it is warm and adds a network hop to every decision. For iterating on
thresholds and prompts against a freshly trained model, pulling the weights down
once and running them on local MPS is cheaper and faster.

After a successful download, select "Laya: Databricks Weights (local MPS)" in the
Decision Engine dropdown.

    uv run python scripts/fetch_databricks_model.py               # latest version
    uv run python scripts/fetch_databricks_model.py --version 4

Requires DATABRICKS_HOST and DATABRICKS_TOKEN in the environment or the repo-root .env.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "runs" / "laya_databricks"

# The pyfunc stores the Laya checkpoint under this artifact subdirectory.
CHECKPOINT_SUBPATH = "artifacts/checkpoint"


def load_env() -> None:
    """Fills missing Databricks credentials from the repo-root .env."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("DATABRICKS_") and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def resolve_version(client, model: str, requested: str | None) -> str:
    if requested:
        return requested
    versions = client.search_model_versions(f"name='{model}'")
    if not versions:
        sys.exit(f"No versions registered for {model}.")
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="muckrack_data.laya.laya_typed_decisions")
    parser.add_argument("--version", default=None, help="defaults to the latest version")
    parser.add_argument("--target", default=str(TARGET_DIR))
    args = parser.parse_args()

    load_env()
    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        sys.exit("Set DATABRICKS_HOST and DATABRICKS_TOKEN (environment or repo-root .env).")

    import mlflow
    from mlflow.tracking import MlflowClient

    # An ambient MLFLOW_TRACKING_URI (the repo has a local sqlite one) otherwise
    # wins and the registry lookup fails on an unsupported scheme.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")

    version = resolve_version(client, args.model, args.version)
    print(f"Downloading {args.model} version {version}...")

    local_path = Path(
        mlflow.artifacts.download_artifacts(f"models:/{args.model}/{version}")
    )
    # The pyfunc lays artifacts out under a directory MLflow chooses, so find
    # the checkpoint by its contents rather than assuming a path.
    marker = "model.safetensors"
    candidates = sorted(local_path.rglob(marker))
    if not candidates:
        listing = sorted(p.relative_to(local_path).as_posix() for p in local_path.rglob("*"))
        sys.exit(f"No {marker} in the downloaded artifacts. Contents:\n  " + "\n  ".join(listing[:60]))
    checkpoint = candidates[0].parent
    print(f"Found checkpoint at {checkpoint.relative_to(local_path)}")
    target = Path(args.target)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, target)

    files = sorted(p.name for p in target.rglob("*") if p.is_file())
    print(f"Wrote {len(files)} files to {target}")
    print("Select 'Laya: Databricks Weights (local MPS)' in the Decision Engine dropdown.")


if __name__ == "__main__":
    main()
