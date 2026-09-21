# Databricks notebook source
# MAGIC %md
# MAGIC # Laya GPU training
# MAGIC
# MAGIC Fine-tunes ModernBERT-large typed-decision heads on the sequences written by
# MAGIC `01_laya_sequence_prep`, logs the run to MLflow, and registers the weights in
# MAGIC Unity Catalog.
# MAGIC
# MAGIC Runs single-GPU by default on `g5.xlarge` (A10G, 24 GB). Set `num_gpus` above 1
# MAGIC on a multi-GPU node and `TorchDistributor` handles DDP.
# MAGIC
# MAGIC Changes from the Azure VM run that produced the hedging model:
# MAGIC
# MAGIC | | Azure T4 | Here |
# MAGIC |---|---|---|
# MAGIC | Epochs | 1 | 3 |
# MAGIC | Trainable layers | top 6 | top 8 |
# MAGIC | Tag class balance | 85% negative | balanced in notebook 01 |
# MAGIC | Positive class weight | none | configurable, default 2.5 |
# MAGIC | Tracking | a log file on the VM | MLflow |

# COMMAND ----------

# MAGIC %md
# MAGIC `%restart_python` resets the interpreter, so it has to run before anything
# MAGIC else. Any variable defined above it is discarded.

# COMMAND ----------

# MAGIC %pip install laya>=0.3.4 transformers>=5.17.0 safetensors --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "muckrack_data", "Catalog")
dbutils.widgets.text("schema", "laya", "Schema")
dbutils.widgets.text("epochs", "3", "Epochs")
dbutils.widgets.text("micro_batch", "8", "Micro batch")
dbutils.widgets.text("grad_accum", "4", "Gradient accumulation")
dbutils.widgets.text("lr", "3.5e-5", "Learning rate")
dbutils.widgets.text("top_layers", "8", "Trainable encoder layers")
dbutils.widgets.text("pos_weight", "2.5", "Positive class weight (noul heads)")
dbutils.widgets.text("num_gpus", "1", "GPUs")
dbutils.widgets.text("model_name", "laya_typed_decisions", "Registered model name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EPOCHS = int(dbutils.widgets.get("epochs"))
MICRO_BATCH = int(dbutils.widgets.get("micro_batch"))
GRAD_ACCUM = int(dbutils.widgets.get("grad_accum"))
LR = float(dbutils.widgets.get("lr"))
TOP_LAYERS = int(dbutils.widgets.get("top_layers"))
POS_WEIGHT = float(dbutils.widgets.get("pos_weight"))
NUM_GPUS = int(dbutils.widgets.get("num_gpus"))
MODEL_NAME = dbutils.widgets.get("model_name")

SEQUENCES_TABLE = f"{CATALOG}.{SCHEMA}.training_sequences"
REGISTERED_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# COMMAND ----------

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("No GPU on this cluster. Attach a g5 or g4dn node before running.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Stage the sequences to local disk
# MAGIC
# MAGIC The training loop reads tensors directly rather than through Spark, so the Delta
# MAGIC table is materialized once to the driver's local SSD. At 35k sequences this is a
# MAGIC few hundred MB.

# COMMAND ----------

import os
from pyspark.sql import functions as F

STAGING_DIR = "/local_disk0/laya_training"
os.makedirs(STAGING_DIR, exist_ok=True)

sequences = spark.table(SEQUENCES_TABLE)
train_pd = sequences.filter(F.col("split") == "train").toPandas()
val_pd = sequences.filter(F.col("split") == "val").toPandas()

print(f"Train: {len(train_pd)} sequences")
print(f"Val:   {len(val_pd)} sequences")
print(train_pd.groupby(["decision_kind", "label"]).size())


def to_items(df):
    return [
        {
            "ids": list(row["ids"]),
            "markers": list(row["markers"]),
            "qtype": int(row["qtype"]),
            "target": list(row["target"]),
            "label": int(row["label"]),
            "qid": row["qid"],
            "decision_kind": row["decision_kind"],
        }
        for _, row in df.iterrows()
    ]


torch.save(to_items(train_pd), f"{STAGING_DIR}/train_items.pt")
torch.save(to_items(val_pd), f"{STAGING_DIR}/val_items.pt")
print(f"Staged tensors to {STAGING_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Training function
# MAGIC
# MAGIC This is `backend/train_laya.py` adapted for Databricks. Two substantive changes:
# MAGIC
# MAGIC - Loss on `noul` heads is weighted so a missed positive costs `pos_weight` times
# MAGIC   a missed negative. Combined with the rebalanced tags from notebook 01 this is
# MAGIC   what pushes true tags past the decision threshold instead of stalling at 0.4.
# MAGIC - Validation runs after every epoch and logs to MLflow, so an overfitting run is
# MAGIC   visible partway through rather than only at the end.

# COMMAND ----------

train_fn_source = '''
import json
import logging
import os
import random
import time

import mlflow
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laya_databricks")


def evaluate(model, items, pad_token_id, device, batch_size=16, max_items=600):
    """Returns overall and per-kind accuracy plus raw logits for temperature fitting."""
    from laya.common import collate_items

    model.eval()
    subset = items[:max_items]
    correct_by_kind, total_by_kind = {}, {}
    total_loss, n_batches = 0.0, 0
    logits_by_type = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
            b = collate_items([chunk], pad_token_id)
            if b is None:
                continue

            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            marker_pos = b["marker_pos"].to(device)
            marker_mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)
            labels = b["label"].to(device)

            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            masked = logits.masked_fill(~marker_mask, -1e4)
            total_loss += -(target * torch.log_softmax(masked, -1)).sum(-1).mean().item()
            n_batches += 1

            preds = masked.argmax(-1)
            matches = (preds == labels).cpu().numpy()

            for it, match, qt, logit_row, targ_row, mmask in zip(
                chunk, matches, qtype.cpu().numpy(), logits.cpu().numpy(),
                target.cpu().numpy(), marker_mask.cpu().numpy(),
            ):
                kind = it["decision_kind"]
                total_by_kind[kind] = total_by_kind.get(kind, 0) + 1
                correct_by_kind[kind] = correct_by_kind.get(kind, 0) + int(match)
                k = int(mmask.sum())
                logits_by_type[int(qt)].append((logit_row[:k].tolist(), targ_row[:k].tolist()))

    by_kind = {
        k: round(correct_by_kind.get(k, 0) / v * 100, 2)
        for k, v in total_by_kind.items()
    }
    total_correct = sum(correct_by_kind.values())
    total_seen = sum(total_by_kind.values())
    model.train()
    return {
        "overall_accuracy_pct": round(total_correct / max(1, total_seen) * 100, 2),
        "by_kind": by_kind,
        "loss": total_loss / max(1, n_batches),
        "raw_logits_by_type": logits_by_type,
    }


def fit_temperature(logits_by_type):
    """Fits one temperature scalar per question type on held-out logits."""
    temps = []
    for qt in (0, 1, 2):
        rows = logits_by_type.get(qt, [])
        if not rows:
            temps.append(1.0)
            continue
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = torch.tensor(0.0)
            for logit_row, targ_row in rows:
                lg = torch.tensor(logit_row) / log_t.exp()
                tg = torch.tensor(targ_row)
                loss = loss - (tg * torch.log_softmax(lg, -1)).sum()
            loss = loss / len(rows)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            temps.append(round(float(log_t.exp().item()), 4))
        except Exception:
            temps.append(1.0)
    return temps


def train():
    import laya
    from laya.common import collate_items
    from safetensors.torch import save_file

    staging_dir = os.environ["LAYA_STAGING_DIR"]
    output_dir = os.environ["LAYA_OUTPUT_DIR"]
    epochs = int(os.environ["LAYA_EPOCHS"])
    micro_batch = int(os.environ["LAYA_MICRO_BATCH"])
    grad_accum = int(os.environ["LAYA_GRAD_ACCUM"])
    lr = float(os.environ["LAYA_LR"])
    top_layers = int(os.environ["LAYA_TOP_LAYERS"])
    pos_weight = float(os.environ["LAYA_POS_WEIGHT"])
    run_id = os.environ["LAYA_MLFLOW_RUN_ID"]

    device = torch.device("cuda")
    train_items = torch.load(f"{staging_dir}/train_items.pt", weights_only=False)
    val_items = torch.load(f"{staging_dir}/val_items.pt", weights_only=False)
    logger.info(f"Loaded {len(train_items)} train, {len(val_items)} val sequences")

    base_agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device="cuda")
    model = base_agent.model
    model.to(device)
    tok = base_agent.tok
    cfg = dict(base_agent.cfg)

    with mlflow.start_run(run_id=run_id):
        pre = evaluate(model, val_items, tok.pad_token_id, device)
        logger.info(f"Pre-training accuracy: {pre['overall_accuracy_pct']}% | {pre['by_kind']}")
        mlflow.log_metric("pre_train_accuracy_pct", pre["overall_accuracy_pct"])
        for kind, acc in pre["by_kind"].items():
            mlflow.log_metric(f"pre_train_accuracy_{kind}", acc)

        for param in model.encoder.parameters():
            param.requires_grad = False
        encoder_layers = getattr(model.encoder, "layers", None)
        if isinstance(encoder_layers, (torch.nn.ModuleList, list)):
            for layer in list(encoder_layers)[-top_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for module in (model.head, model.type_emb, model.scorer, model.act_head):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        total_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable_count:,} / {total_count:,}")
        mlflow.log_param("trainable_params", trainable_count)
        mlflow.log_param("total_params", total_count)

        model.train()
        optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        total_steps = max(1, (len(train_items) // (micro_batch * grad_accum)) * epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        logger.info(f"Training {epochs} epochs, ~{total_steps} optimizer steps")

        step_count = 0
        t_start = time.perf_counter()

        for epoch in range(epochs):
            random.seed(42 + epoch)
            random.shuffle(train_items)
            running_loss, n_batches, accum_step = 0.0, 0, 0

            for b_idx in range(0, len(train_items), micro_batch):
                chunk = train_items[b_idx:b_idx + micro_batch]
                b = collate_items([chunk], tok.pad_token_id)
                if b is None:
                    continue

                input_ids = b["input_ids"].to(device)
                attention_mask = b["attention_mask"].to(device)
                marker_pos = b["marker_pos"].to(device)
                marker_mask = b["marker_mask"].to(device)
                qtype = b["qtype"].to(device)
                target = b["target"].to(device)

                with torch.autocast(device_type="cuda", enabled=True):
                    logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                    masked = logits.masked_fill(~marker_mask, -1e4)
                    per_sample = -(target * torch.log_softmax(masked, -1)).sum(-1)

                    # Weight the positive class on noul questions. qtype 0 is noul, and
                    # index 1 of the target is the "true" option.
                    is_noul = (qtype == 0)
                    is_positive = (target.argmax(-1) == 1)
                    weights = torch.where(is_noul & is_positive, pos_weight, 1.0).to(per_sample.dtype)
                    loss_ce = (per_sample * weights).sum() / weights.sum()
                    loss = loss_ce / grad_accum + 0.0 * act.sum()

                scaler.scale(loss).backward()
                accum_step += 1
                running_loss += loss_ce.detach().item()
                n_batches += 1

                if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(train_items):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_count += 1
                    if step_count % 50 == 0:
                        logger.info(f"Step {step_count}/{total_steps} | loss {loss_ce.item():.4f}")
                        mlflow.log_metric("train_loss", loss_ce.item(), step=step_count)
                        mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=step_count)

            epoch_eval = evaluate(model, val_items, tok.pad_token_id, device)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | train loss {running_loss / max(1, n_batches):.4f} | "
                f"val {epoch_eval['overall_accuracy_pct']}% | {epoch_eval['by_kind']}"
            )
            mlflow.log_metric("epoch_train_loss", running_loss / max(1, n_batches), step=epoch + 1)
            mlflow.log_metric("epoch_val_accuracy_pct", epoch_eval["overall_accuracy_pct"], step=epoch + 1)
            mlflow.log_metric("epoch_val_loss", epoch_eval["loss"], step=epoch + 1)
            for kind, acc in epoch_eval["by_kind"].items():
                mlflow.log_metric(f"epoch_val_accuracy_{kind}", acc, step=epoch + 1)

        duration = time.perf_counter() - t_start
        logger.info(f"Training finished in {duration / 60:.1f} min")

        post = evaluate(model, val_items, tok.pad_token_id, device)
        logger.info(f"Post-training accuracy: {post['overall_accuracy_pct']}% | {post['by_kind']}")
        mlflow.log_metric("post_train_accuracy_pct", post["overall_accuracy_pct"])
        for kind, acc in post["by_kind"].items():
            mlflow.log_metric(f"post_train_accuracy_{kind}", acc)
        mlflow.log_metric("training_minutes", round(duration / 60, 2))

        temps = fit_temperature(post["raw_logits_by_type"])
        model.temperature.data = torch.tensor(temps, dtype=torch.float32)
        logger.info(f"Calibrated temperatures: {temps}")
        mlflow.log_param("temperature", temps)

        os.makedirs(output_dir, exist_ok=True)
        save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
        cfg["temperature"] = temps
        cfg["finetuned_on"] = "curation_engine_audit_blobs"
        cfg["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        tok_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok.save_pretrained(tok_dir)

        enc_dir = os.path.join(output_dir, "encoder")
        os.makedirs(enc_dir, exist_ok=True)
        model.encoder.config.save_pretrained(enc_dir)

        return {
            "pre_train_accuracy_pct": pre["overall_accuracy_pct"],
            "post_train_accuracy_pct": post["overall_accuracy_pct"],
            "by_kind": post["by_kind"],
            "temperature": temps,
            "training_minutes": round(duration / 60, 2),
            "output_dir": output_dir,
        }
'''

with open("/local_disk0/laya_train_fn.py", "w") as f:
    f.write(train_fn_source)

print("Wrote training function to /local_disk0/laya_train_fn.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run it
# MAGIC
# MAGIC `TorchDistributor` with `local_mode=True` runs on the driver GPU. On a multi-GPU
# MAGIC node, raise `num_gpus` and it switches to DDP without further changes.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
experiment_path = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/laya_training"
mlflow.set_experiment(experiment_path)

OUTPUT_DIR = "/local_disk0/laya_checkpoint"

with mlflow.start_run(run_name=f"laya-{EPOCHS}ep-top{TOP_LAYERS}-pw{POS_WEIGHT}") as run:
    mlflow.log_params({
        "epochs": EPOCHS,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": MICRO_BATCH * GRAD_ACCUM,
        "learning_rate": LR,
        "top_layers": TOP_LAYERS,
        "pos_weight": POS_WEIGHT,
        "num_gpus": NUM_GPUS,
        "sequences_table": SEQUENCES_TABLE,
        "train_sequences": len(train_pd),
        "val_sequences": len(val_pd),
        "gpu": torch.cuda.get_device_name(0),
    })

    os.environ.update({
        "LAYA_STAGING_DIR": STAGING_DIR,
        "LAYA_OUTPUT_DIR": OUTPUT_DIR,
        "LAYA_EPOCHS": str(EPOCHS),
        "LAYA_MICRO_BATCH": str(MICRO_BATCH),
        "LAYA_GRAD_ACCUM": str(GRAD_ACCUM),
        "LAYA_LR": str(LR),
        "LAYA_TOP_LAYERS": str(TOP_LAYERS),
        "LAYA_POS_WEIGHT": str(POS_WEIGHT),
        "LAYA_MLFLOW_RUN_ID": run.info.run_id,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    import sys
    sys.path.insert(0, "/local_disk0")
    import laya_train_fn
    import importlib
    importlib.reload(laya_train_fn)

    if NUM_GPUS > 1:
        from pyspark.ml.torch.distributor import TorchDistributor
        summary = TorchDistributor(
            num_processes=NUM_GPUS, local_mode=True, use_gpu=True
        ).run(laya_train_fn.train)
    else:
        summary = laya_train_fn.train()

    print(json.dumps(summary, indent=2))
    mlflow.log_dict(summary, "training_summary.json")
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="laya_model")

    FINAL_RUN_ID = run.info.run_id

print(f"MLflow run: {FINAL_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register the weights
# MAGIC
# MAGIC Registering is gated on the run beating the current production model. Without the
# MAGIC gate a worse run silently becomes the new candidate.

# COMMAND ----------

MIN_ACCURACY = 83.0

if summary["post_train_accuracy_pct"] >= MIN_ACCURACY:
    result = mlflow.register_model(
        model_uri=f"runs:/{FINAL_RUN_ID}/laya_model",
        name=REGISTERED_MODEL,
    )
    print(f"Registered {REGISTERED_MODEL} version {result.version}")
    print(f"Accuracy {summary['post_train_accuracy_pct']}% | {summary['by_kind']}")
else:
    print(
        f"Not registered. Accuracy {summary['post_train_accuracy_pct']}% "
        f"is below the {MIN_ACCURACY}% gate."
    )

# COMMAND ----------

dbutils.notebook.exit(json.dumps(summary))
