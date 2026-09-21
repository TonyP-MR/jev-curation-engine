# Databricks notebook source
# MAGIC %md
# MAGIC # Recover and publish a trained Laya checkpoint
# MAGIC
# MAGIC Training run `bb46c09dfc0a42308e27b003143ee8c1` reached 89.33% and saved its
# MAGIC checkpoint, then the driver REPL died during `mlflow.pyfunc.log_model` with
# MAGIC `Py4JException: Error while obtaining a new communication channel`. The weights
# MAGIC are still on the cluster's local disk, so this republishes them instead of
# MAGIC spending another GPU hour retraining.
# MAGIC
# MAGIC The wrapper class below is generated from `databricks/02_laya_gpu_training.py`
# MAGIC so the two cannot drift.

# COMMAND ----------

# MAGIC %pip install laya>=0.3.4 transformers>=5.17.0 safetensors --quiet
# MAGIC %restart_python

# COMMAND ----------

import json
import os

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

CHECKPOINT = "/local_disk0/laya_checkpoint"
SOURCE_RUN_ID = "bb46c09dfc0a42308e27b003143ee8c1"
REGISTERED_MODEL = "muckrack_data.laya.laya_typed_decisions"
SERVING_ENDPOINT = "laya-curation-engine"

present = sorted(os.listdir(CHECKPOINT)) if os.path.isdir(CHECKPOINT) else []
print(f"checkpoint dir: {CHECKPOINT}")
print(f"contents: {present}")
if not present:
    dbutils.notebook.exit(json.dumps({"error": "checkpoint missing", "dir": CHECKPOINT}))

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
VALIDATION_GATE = 0.50


class LayaDecisionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import laya
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = laya.load(context.artifacts["checkpoint"], device=device)
        self.agent.cfg["max_len"] = 2048
        self.agent.cfg["head_max_len"] = 384

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
    entity_version="1",
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

dbutils.notebook.exit(json.dumps({
    "version": "1",
    "endpoint": SERVING_ENDPOINT,
    "action": action,
}))
