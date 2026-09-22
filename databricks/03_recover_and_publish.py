# Databricks notebook source
# MAGIC %md
# MAGIC # Recover and publish a trained Laya checkpoint
# MAGIC
# MAGIC Use this only when a qualifying training run saved its best-epoch checkpoint
# MAGIC but failed while packaging or registering it. The source run must contain the
# MAGIC complete four-kind metrics written by `02_laya_gpu_training`; this notebook
# MAGIC refuses incomplete or below-threshold runs rather than bypassing the training
# MAGIC registration gate.

# COMMAND ----------

# MAGIC %pip install laya>=0.3.4 transformers==4.57.6 safetensors --quiet
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import shutil
import sys
import time
from importlib.metadata import version as package_version

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, Schema

_NOTEBOOK_DIR = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
for _candidate in (f"/Workspace{_NOTEBOOK_DIR}", _NOTEBOOK_DIR):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import laya_questions

LAYA_QUESTIONS_PATH = laya_questions.__file__
PACKAGED_LAYA_QUESTIONS_PATH = "/local_disk0/laya_questions.py"
shutil.copyfile(LAYA_QUESTIONS_PATH, PACKAGED_LAYA_QUESTIONS_PATH)
SERVING_REQUIREMENTS = [
    f"{package}=={package_version(package)}"
    for package in (
        "cloudpickle",
        "laya",
        "torch",
        "transformers",
        "safetensors",
    )
]
print(f"Exact serving requirements: {SERVING_REQUIREMENTS}")

dbutils.widgets.text("source_run_id", "", "Qualifying training run ID")
dbutils.widgets.text("serving_endpoint", "laya-curation-engine", "Serving endpoint")

CHECKPOINT = "/local_disk0/laya_checkpoint"
SOURCE_RUN_ID = dbutils.widgets.get("source_run_id").strip()
REGISTERED_MODEL = "muckrack_data.laya.laya_typed_decisions"
SERVING_ENDPOINT = dbutils.widgets.get("serving_endpoint").strip()
EXPECTED_DECISION_KINDS = ("validation", "prominence", "sentiment", "tag")
MIN_ACCURACY = 83.0

if not SOURCE_RUN_ID:
    raise RuntimeError("source_run_id is required")
if not SERVING_ENDPOINT:
    raise RuntimeError("serving_endpoint is required")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
source_run = client.get_run(SOURCE_RUN_ID)
source_metrics = source_run.data.metrics
required_metrics = {
    "post_train_accuracy_pct",
    "best_epoch",
    "best_val_loss",
}
for kind in EXPECTED_DECISION_KINDS:
    required_metrics.add(f"post_train_{kind}_count")
    required_metrics.add(f"post_train_{kind}_accuracy_pct")

missing_metrics = sorted(required_metrics - source_metrics.keys())
zero_count_kinds = [
    kind
    for kind in EXPECTED_DECISION_KINDS
    if source_metrics.get(f"post_train_{kind}_count", 0) <= 0
]
if missing_metrics or zero_count_kinds:
    details = []
    if missing_metrics:
        details.append("missing metrics: " + ", ".join(missing_metrics))
    if zero_count_kinds:
        details.append("zero-count kinds: " + ", ".join(zero_count_kinds))
    raise RuntimeError(
        f"Source run {SOURCE_RUN_ID} has incomplete evaluation; " + "; ".join(details)
    )
source_accuracy = source_metrics["post_train_accuracy_pct"]
if source_accuracy < MIN_ACCURACY:
    raise RuntimeError(
        f"Source run {SOURCE_RUN_ID} accuracy {source_accuracy}% is below "
        f"the {MIN_ACCURACY}% registration gate"
    )


def _find_checkpoint(root: str) -> str | None:
    for dirpath, _dirnames, filenames in os.walk(root):
        if "rl_agent_config.json" in filenames:
            return dirpath
    return None


def _checkpoint_run_id(root: str) -> str | None:
    marker = os.path.join(root, "training_run_id.txt")
    if not os.path.isfile(marker):
        return None
    with open(marker, encoding="utf-8") as marker_file:
        return marker_file.read().strip()


if _checkpoint_run_id(CHECKPOINT) != SOURCE_RUN_ID:
    print(
        "Local checkpoint is absent or belongs to another run; "
        f"downloading artifacts from {SOURCE_RUN_ID}"
    )
    recovered_checkpoint = None
    download_errors = []
    for artifact_path in ("checkpoint", "laya_model"):
        try:
            local_artifact = mlflow.artifacts.download_artifacts(
                f"runs:/{SOURCE_RUN_ID}/{artifact_path}"
            )
        except MlflowException as error:
            download_errors.append(f"{artifact_path}: {error}")
            continue
        recovered_checkpoint = _find_checkpoint(local_artifact)
        if recovered_checkpoint:
            break

    if not recovered_checkpoint:
        raise RuntimeError(
            "No checkpoint found in source-run artifacts. "
            + " | ".join(download_errors)
        )
    CHECKPOINT = recovered_checkpoint

present = sorted(os.listdir(CHECKPOINT)) if os.path.isdir(CHECKPOINT) else []
print(f"checkpoint dir: {CHECKPOINT}")
print(f"contents: {present}")
checkpoint_run_id = _checkpoint_run_id(CHECKPOINT)
if checkpoint_run_id != SOURCE_RUN_ID:
    raise RuntimeError(
        f"Checkpoint belongs to run {checkpoint_run_id!r}, not {SOURCE_RUN_ID}"
    )

# COMMAND ----------

PASSING_PROMINENCE = {
    "type": "choice",
    "choice": "passing",
    "probabilities": {"primary": 0.02, "significant": 0.08, "passing": 0.90},
    "confidence": 0.92,
}
NEUTRAL_SENTIMENT = {
    "type": "choice",
    "choice": "neutral",
    "probabilities": {"positive": 0.05, "negative": 0.05, "neutral": 0.88, "balanced": 0.02},
    "confidence": 0.90,
}
FALSE_TAG = {"type": "noul", "noul": 0.0, "confidence": 1.0}

# Matches VALIDATION_THRESHOLD in backend/config.py. Gating on a different
# number here would make served and local answers disagree.
VALIDATION_GATE = 0.45


class LayaDecisionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import laya
        import torch
        from laya_questions import HEAD_MAX_LEN, MAX_LEN

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = laya.load(context.artifacts["checkpoint"], device=device)
        # Serve at the budget the sequences were built and trained at. This was
        # 2048/384, a truncation regime the fine-tune never saw.
        self.agent.cfg["max_len"] = MAX_LEN
        self.agent.cfg["head_max_len"] = HEAD_MAX_LEN

    def _decide(self, state, questions):
        val_questions = {k: v for k, v in questions.items() if k.endswith("_valid")}

        if not val_questions or len(val_questions) == len(questions):
            return self.agent.predict(state, questions)

        res_val = self.agent.predict(state, val_questions)
        answers = dict(res_val.get("answers", {}))
        input_tokens = int(res_val.get("usage", {}).get("input_tokens", 0))

        valid_subjects = {
            qid.replace("subj_", "").replace("_valid", "")
            for qid, ans in answers.items()
            if ans.get("noul", 0.0) >= VALIDATION_GATE
        }

        stage2 = {}
        for qid, qdef in questions.items():
            if qid.endswith("_valid"):
                continue
            subject_id = ""
            if qid.startswith(("subj_", "tag_")):
                parts = qid.split("_")
                if len(parts) > 1:
                    subject_id = parts[1]

            if subject_id in valid_subjects:
                stage2[qid] = qdef
            elif qid.endswith("_prominence"):
                answers[qid] = dict(PASSING_PROMINENCE)
            elif qid.endswith("_sentiment"):
                answers[qid] = dict(NEUTRAL_SENTIMENT)
            elif qid.startswith("tag_"):
                answers[qid] = dict(FALSE_TAG)

        if stage2:
            res2 = self.agent.predict(state, stage2)
            answers.update(res2.get("answers", {}))
            input_tokens += int(res2.get("usage", {}).get("input_tokens", 0))

        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    def predict(self, context, model_input, params=None):
        out = []
        for _, row in model_input.iterrows():
            questions = row["questions"]
            if isinstance(questions, str):
                questions = json.loads(questions)
            result = self._decide(row["state"], questions)
            out.append(
                json.dumps({
                    "model": REGISTERED_MODEL,
                    "answers": result.get("answers", {}),
                    "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
                })
            )
        return out

SERVING_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "state"), ColSpec("string", "questions")]),
    outputs=Schema([ColSpec("string")]),
)
SERVING_EXAMPLE = pd.DataFrame(
    [{
        "state": "Headline: Example Corp names a new CFO\n\n\nLead Paragraph:\nExample Corp announced...",
        "questions": json.dumps({
            "subj_1_valid": {
                "type": "noul",
                "instructions": "Is the provided content meaningfully relevant to 'Example Corp'?",
                "criteria": {"true": "Relevant to Example Corp.", "false": "Not relevant."},
            }
        }),
    }]
)

# COMMAND ----------

# Re-log only a checkpoint whose run has already cleared the complete evaluation
# and aggregate accuracy gates above.
with mlflow.start_run(run_name=f"recover-{SOURCE_RUN_ID}") as run:
    mlflow.log_params(
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_accuracy_pct": source_accuracy,
            "source_best_epoch": source_metrics["best_epoch"],
        }
    )
    mlflow.log_dict(
        {"pip_requirements": SERVING_REQUIREMENTS},
        "serving_requirements.json",
    )
    mlflow.pyfunc.log_model(
        artifact_path="laya_model",
        python_model=LayaDecisionModel(),
        artifacts={"checkpoint": CHECKPOINT},
        signature=SERVING_SIGNATURE,
        input_example=SERVING_EXAMPLE,
        pip_requirements=SERVING_REQUIREMENTS,
        code_paths=[PACKAGED_LAYA_QUESTIONS_PATH],
    )
    LOG_RUN_ID = run.info.run_id

versions_before = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
if "2" not in {str(version.version) for version in versions_before}:
    raise RuntimeError(
        f"Rollback version 2 is missing from {REGISTERED_MODEL}; refusing publication"
    )

registered = mlflow.register_model(
    model_uri=f"runs:/{LOG_RUN_ID}/laya_model",
    name=REGISTERED_MODEL,
)
NEW_VERSION = registered.version
print(f"registered {REGISTERED_MODEL} v{NEW_VERSION} from run {LOG_RUN_ID}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()
entity = ServedEntityInput(
    entity_name=REGISTERED_MODEL,
    entity_version=str(NEW_VERSION),
    workload_size="Small",
    workload_type=ServingModelWorkloadType.GPU_SMALL,
    scale_to_zero_enabled=True,
)
existing = [e.name for e in w.serving_endpoints.list()]
if SERVING_ENDPOINT in existing:
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[entity])
    action = "updated"
else:
    w.serving_endpoints.create(
        name=SERVING_ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=[entity]),
    )
    action = "created"

deadline = time.monotonic() + 45 * 60
while True:
    endpoint = w.serving_endpoints.get(SERVING_ENDPOINT)
    ready = getattr(endpoint.state.ready, "value", str(endpoint.state.ready))
    config_update = getattr(
        endpoint.state.config_update,
        "value",
        str(endpoint.state.config_update),
    )
    print(f"Endpoint state: ready={ready}, config_update={config_update}")
    if config_update == "NOT_UPDATING":
        if ready != "READY":
            raise RuntimeError(
                f"Endpoint {SERVING_ENDPOINT} stopped updating but is {ready}"
            )
        break
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"Endpoint {SERVING_ENDPOINT} did not become ready within 45 minutes"
        )
    time.sleep(15)

deployed_versions = {
    str(served.entity_version)
    for served in (endpoint.config.served_entities or [])
}
if deployed_versions != {str(NEW_VERSION)}:
    raise RuntimeError(
        f"Endpoint {SERVING_ENDPOINT} has unexpected deployed versions: "
        f"{sorted(deployed_versions)}"
    )

versions_after = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
if "2" not in {str(version.version) for version in versions_after}:
    raise RuntimeError(
        f"Rollback version 2 disappeared from {REGISTERED_MODEL}"
    )

dbutils.notebook.exit(json.dumps({
    "source_run_id": SOURCE_RUN_ID,
    "packaging_run_id": LOG_RUN_ID,
    "source_accuracy_pct": source_accuracy,
    "source_best_epoch": source_metrics["best_epoch"],
    "version": str(NEW_VERSION),
    "rollback_version": "2",
    "endpoint": SERVING_ENDPOINT,
    "endpoint_ready": True,
    "action": action,
    "pip_requirements": SERVING_REQUIREMENTS,
}))
