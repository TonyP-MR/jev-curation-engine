# Databricks notebook source
# MAGIC %md
# MAGIC # Laya sequence prep
# MAGIC
# MAGIC Distributed replacement for `backend/dataset_builder.py`.
# MAGIC
# MAGIC Reads Curation Engine audit blobs and published config snapshots, builds tokenized
# MAGIC typed-decision sequences with Spark, and writes them to a Delta table.
# MAGIC
# MAGIC Output table: `muckrack_data.laya.training_sequences`
# MAGIC
# MAGIC ## Before the first run
# MAGIC
# MAGIC The audit blobs live in Azure Blob Storage and this workspace is on AWS, so the
# MAGIC storage key has to be registered once as a Databricks secret:
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope laya
# MAGIC databricks secrets put-secret laya azure_storage_key
# MAGIC ```
# MAGIC
# MAGIC Set `source_mode` to `volume` instead if you stage the blobs into a UC volume.

# COMMAND ----------

# MAGIC %md
# MAGIC `%restart_python` resets the interpreter, so it has to run before anything
# MAGIC else. Any variable defined above it is discarded.

# COMMAND ----------

# MAGIC %pip install laya>=0.3.4 transformers>=5.17.0 --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("source_mode", "azure", ["azure", "volume"], "Blob source")
dbutils.widgets.text("azure_account", "stcurationauditprod", "Azure storage account")
dbutils.widgets.text("azure_container", "pipeline-audit", "Azure container")
dbutils.widgets.text("volume_path", "/Volumes/muckrack_data/laya/audit_blobs", "UC volume path")
dbutils.widgets.text("config_path", "/Volumes/muckrack_data/laya/configs", "Config snapshot path")
dbutils.widgets.text("catalog", "muckrack_data", "Catalog")
dbutils.widgets.text("schema", "laya", "Schema")
dbutils.widgets.text("blobs_limit", "0", "Blob limit (0 = all)")
dbutils.widgets.text("parse_limit", "0", "Blobs to parse from bronze (0 = all)")
dbutils.widgets.dropdown("skip_ingest", "false", ["true", "false"], "Skip ingest, reuse bronze")
dbutils.widgets.text("tag_pos_ratio", "0.5", "Target positive tag ratio")

SOURCE_MODE = dbutils.widgets.get("source_mode")
AZURE_ACCOUNT = dbutils.widgets.get("azure_account")
AZURE_CONTAINER = dbutils.widgets.get("azure_container")
VOLUME_PATH = dbutils.widgets.get("volume_path")
CONFIG_PATH = dbutils.widgets.get("config_path")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
BLOBS_LIMIT = int(dbutils.widgets.get("blobs_limit"))
PARSE_LIMIT = int(dbutils.widgets.get("parse_limit"))
SKIP_INGEST = dbutils.widgets.get("skip_ingest") == "true"
TAG_POS_RATIO = float(dbutils.widgets.get("tag_pos_ratio"))

BRONZE_BLOBS = f"{CATALOG}.{SCHEMA}.bronze_audit_blobs"
BRONZE_CONFIGS = f"{CATALOG}.{SCHEMA}.bronze_configs"
GOLD_SEQUENCES = f"{CATALOG}.{SCHEMA}.training_sequences"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

if SOURCE_MODE == "azure":
    spark.conf.set(
        f"fs.azure.account.key.{AZURE_ACCOUNT}.dfs.core.windows.net",
        dbutils.secrets.get(scope="laya", key="azure_storage_key"),
    )
    blob_source = f"abfss://{AZURE_CONTAINER}@{AZURE_ACCOUNT}.dfs.core.windows.net/"
else:
    blob_source = VOLUME_PATH

print(f"Reading audit blobs from {blob_source}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest audit blobs and config snapshots
# MAGIC
# MAGIC Auto Loader rather than a plain glob. The `pipeline-audit` container holds far
# MAGIC more than the 5,000 blobs a single list page returns, and `spark.read` with a
# MAGIC `*.json` glob enumerates the whole container before `limit()` applies. A 300-blob
# MAGIC smoke test then spends its time listing hundreds of thousands of files.
# MAGIC
# MAGIC Auto Loader lists incrementally and checkpoints what it has seen, so a bounded
# MAGIC run stays bounded and the next run picks up only new blobs.
# MAGIC
# MAGIC Blobs are read as whole text so the Python parsing below stays identical to the
# MAGIC local `dataset_builder.py`. `wholetext` avoids Spark's JSON schema inference,
# MAGIC which flattens the nested `subject_results` and `scored_attributes` structures.

# COMMAND ----------

from pyspark.sql import functions as F

CHECKPOINT = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/bronze_audit_blobs"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")

if SKIP_INGEST:
    blob_count = spark.table(BRONZE_BLOBS).count()
    print(f"Skipping ingest. {BRONZE_BLOBS} already holds {blob_count} blobs.")
else:
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("cloudFiles.schemaLocation", f"{CHECKPOINT}/schema")
        .option("wholetext", "true")
        .option("pathGlobFilter", "*.json")
    )

    if BLOBS_LIMIT > 0:
        stream = stream.option("cloudFiles.maxFilesPerTrigger", str(BLOBS_LIMIT))

    raw_blobs = (
        stream.load(blob_source)
        .select(
            F.col("value").alias("payload"),
            F.col("_metadata.file_path").alias("source_path"),
            F.col("_metadata.file_modification_time").alias("ingested_at"),
        )
    )

    # `availableNow` drains what is currently there and stops, honouring
    # maxFilesPerTrigger as a per-batch cap. A bounded run uses one batch.
    query = (
        raw_blobs.writeStream
        .option("checkpointLocation", f"{CHECKPOINT}/state")
        .trigger(availableNow=True)
        .toTable(BRONZE_BLOBS)
    )

    if BLOBS_LIMIT > 0:
        # Stop after the first batch so a bounded run does not drain the
        # container. It holds millions of blobs; draining it is never wanted.
        import time
        while query.isActive:
            seen = sum(p["numInputRows"] for p in query.recentProgress or [])
            if seen >= BLOBS_LIMIT:
                query.stop()
                break
            time.sleep(2)
    query.awaitTermination()

    blob_count = spark.table(BRONZE_BLOBS).count()
    print(f"Ingested {blob_count} audit blobs into {BRONZE_BLOBS}")

# COMMAND ----------

raw_configs = (
    spark.read.format("text")
    .option("wholetext", "true")
    .load(f"{CONFIG_PATH.rstrip('/')}/*.json")
    .select(
        F.col("value").alias("payload"),
        F.regexp_extract(F.col("_metadata.file_path"), r"([^/]+)\.json$", 1).alias("config_id"),
    )
)

raw_configs.write.mode("overwrite").saveAsTable(BRONZE_CONFIGS)
config_count = spark.table(BRONZE_CONFIGS).count()
print(f"Ingested {config_count} config snapshots into {BRONZE_CONFIGS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Broadcast the config snapshots
# MAGIC
# MAGIC There are a few hundred configs against tens of thousands of articles, so the
# MAGIC configs go to every executor once rather than being shuffled per row.

# COMMAND ----------

import json

config_rows = spark.table(BRONZE_CONFIGS).collect()
config_map = {}
for row in config_rows:
    try:
        parsed = json.loads(row["payload"])
    except json.JSONDecodeError:
        continue
    snapshot = parsed.get("snapshot") or parsed.get("snapshot_data")
    if snapshot:
        config_map[row["config_id"]] = snapshot

config_bc = spark.sparkContext.broadcast(config_map)
print(f"Broadcast {len(config_map)} config snapshots")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Question conversion helpers
# MAGIC
# MAGIC These are the same functions as `backend/question_converter.py`. They are inlined
# MAGIC so the notebook runs without the repo on the cluster. Keep them in sync when the
# MAGIC backend prompts change, or install the repo as a wheel and import instead.

# COMMAND ----------

import re
from typing import Any

PROMINENCE_MAP = {"primary": 0, "significant": 1, "passing": 2}
SENTIMENT_MAP = {"positive": 0, "negative": 1, "neutral": 2, "balanced": 3}

GENERIC_CATEGORY_WORDS = {
    "cola", "systems", "technologies", "software", "foods", "health",
    "group", "holdings", "brands", "international", "global", "solutions",
    "network", "networks", "enterprises", "energy", "capital", "partners",
    "mobile", "wireless", "telecom", "communications", "media", "online",
    "digital", "care", "services", "financial", "management",
}

PARTNERSHIP_GUARD = (
    " Explicit negative exclusion: Standard third-party software compatibility, app store listings, "
    "operating system integrations, APIs, generic client reviews, and multi-vendor list roundups DO NOT qualify "
    "as a partnership. A partnership requires an explicit, announced mutual corporate or technology agreement."
)
DEVELOPER_GUARD = (
    " Explicit negative exclusion: General consumer how-to guides, personal password tips, and general security "
    "tutorials aimed at everyday end-users DO NOT qualify as Developer content. The content must specifically "
    "target software engineers, developer tooling, SDKs, or programming workflows."
)
AI_GUARD = (
    " Explicit negative exclusion: Passing biographical mentions in author blurbs, routine automation, "
    "or generic software algorithms DO NOT qualify. The content must substantively discuss artificial intelligence "
    "or machine learning directly related to the subject."
)


def sanitize_heading(text: str) -> str:
    return "\n".join(re.sub(r"^#+\s*", "", line) for line in (text or "").splitlines()).strip()


def tag_guardrail(tag_name: str) -> str:
    lowered = tag_name.lower()
    if any(w in lowered for w in ("partnership", "partner", "alliance", "joint venture")):
        return PARTNERSHIP_GUARD
    if any(w in lowered for w in ("developer", "devops", "engineering")):
        return DEVELOPER_GUARD
    if lowered == "ai" or "artificial intelligence" in lowered:
        return AI_GUARD
    return ""


def convert_config_to_questions(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for s in snapshot.get("subjects", []):
        s_id = str(s["id"])
        name = s.get("name") or f"Subject {s_id}"

        questions[f"subj_{s_id}_valid"] = {
            "type": "noul",
            "instructions": (
                f"Is the provided content meaningfully relevant to '{name}' based on the validation criteria?"
            ),
            "criteria": {
                "true": f"Content satisfies validation criteria for {name}: "
                        f"{sanitize_heading(s.get('validation_prompt') or '')[:600]}",
                "false": f"Content is unrelated, passing mention without substance, or fails validation "
                         f"criteria for {name}.",
            },
        }

        questions[f"subj_{s_id}_prominence"] = {
            "type": "choice",
            "instructions": (
                f"What is the prominence of '{name}' in this content? "
                f"Return exactly one API choice: primary, significant, or passing."
            ),
            "criteria": {
                "primary": f"Primary dominant focus of the article. {name} is the central subject "
                           f"featured in the headline, lead, and narrative.",
                "significant": f"Key topic discussed substantively, or featured in the headline or lead. "
                               f"Not for incidental mentions in multi-vendor lists.",
                "passing": f"Incidental reference: {name} appears in a list of companies, is cited as a "
                           f"secondary data source, or is mentioned briefly without headline focus.",
            },
        }

        questions[f"subj_{s_id}_sentiment"] = {
            "type": "choice",
            "instructions": (
                f"What is the specific tone and sentiment towards '{name}' in this content? "
                f"Isolate sentiment strictly to '{name}'."
            ),
            "criteria": {
                "positive": f"Explicit institutional praise, awards, or celebratory acclaim directed at {name}.",
                "negative": f"Direct adverse coverage, criticism, scandal, litigation, or severe failure "
                            f"directed at {name}.",
                "neutral": f"Editorial baseline: factual reporting, earnings, operational transactions, "
                           f"executive hires, and routine industry news regarding {name}.",
                "balanced": f"Substantial elements of both explicit praise and severe criticism regarding {name}.",
            },
        }

        for t in s.get("tag_evaluations", []):
            if t.get("evaluation_type") != "llm" or not t.get("prompt_text"):
                continue
            t_id = t["tag_id"]
            t_name = t.get("tag_name") or t_id
            criteria_text = sanitize_heading(t.get("prompt_text") or "") + tag_guardrail(t_name)
            questions[f"tag_{s_id}_{t_id}"] = {
                "type": "noul",
                "instructions": f"Does the content match the tag criteria for '{t_name}' regarding '{name}'?",
                "criteria": {
                    "true": f"Criteria satisfied for {t_name}: {criteria_text.strip()}",
                    "false": f"Criteria NOT satisfied for {t_name}.",
                },
            }

    return questions


def build_state(blob: dict[str, Any], max_chars: int = 4000) -> str:
    inbound = blob.get("inbound_data") or {}
    headline = inbound.get("headline") or blob.get("headline") or ""
    body = inbound.get("body") or blob.get("body") or ""
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    lead = paragraphs[0] if paragraphs else ""
    rest = "\n\n".join(paragraphs[1:])
    state = f"Headline: {headline}\n\n\nLead Paragraph:\n{lead}\n\nArticle Body:\n{rest}"
    return state[:max_chars]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tokenize with a Spark UDF
# MAGIC
# MAGIC The tokenizer loads once per executor inside a module-level cache, so the
# MAGIC HuggingFace download happens once per worker rather than once per row.

# COMMAND ----------

from pyspark.sql.types import (
    ArrayType, FloatType, IntegerType, LongType, StringType, StructField, StructType,
)

SEQUENCE_SCHEMA = StructType([
    StructField("correlation_id", StringType()),
    StructField("config_id", StringType()),
    StructField("qid", StringType()),
    StructField("decision_kind", StringType()),
    StructField("ids", ArrayType(LongType())),
    StructField("markers", ArrayType(LongType())),
    StructField("qtype", IntegerType()),
    StructField("target", ArrayType(FloatType())),
    StructField("label", IntegerType()),
])

_TOKENIZER_CACHE: dict[str, Any] = {}


def get_tokenizer():
    """Loads the Laya tokenizer once per executor process."""
    if "tok" in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE["tok"], _TOKENIZER_CACHE["max_len"], _TOKENIZER_CACHE["head_max_len"]

    import os
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    raw_dir = snapshot_download("convaiinnovations/laya", allow_patterns=["typed-decisions/*"])
    model_dir = os.path.join(raw_dir, "typed-decisions")
    _fix_tokenizer_config(model_dir)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    _TOKENIZER_CACHE["tok"] = tok
    _TOKENIZER_CACHE["max_len"] = cfg.get("max_len", 1024)
    _TOKENIZER_CACHE["head_max_len"] = cfg.get("head_max_len", 256)
    return tok, _TOKENIZER_CACHE["max_len"], _TOKENIZER_CACHE["head_max_len"]


def build_sequences_partition(rows):
    """Turns a partition of raw audit blobs into tokenized training sequences."""
    from laya.common import QTYPES, build_sequence, render_options

    tok, max_len, head_max_len = get_tokenizer()
    configs = config_bc.value

    for row in rows:
        try:
            blob = json.loads(row["payload"])
        except json.JSONDecodeError:
            continue

        cfg_id = blob.get("config_id")
        snapshot = configs.get(cfg_id)
        if not snapshot:
            continue

        questions = convert_config_to_questions(snapshot)
        state_text = build_state(blob)
        corr_id = blob.get("correlation_id")

        def emit(qid, kind, q_def, target, label):
            internal_q = {
                "t": q_def["type"],
                "ins": q_def["instructions"],
                "crit": q_def.get("criteria", {}),
            }
            try:
                seq, markers = build_sequence(tok, state_text, internal_q, max_len, head_max_len)
            except (ValueError, KeyError, IndexError):
                return None
            if len(markers) != len(render_options(internal_q)):
                return None
            return {
                "correlation_id": corr_id,
                "config_id": cfg_id,
                "qid": qid,
                "decision_kind": kind,
                "ids": [int(i) for i in seq],
                "markers": [int(m) for m in markers],
                "qtype": int(QTYPES[q_def["type"]]),
                "target": [float(t) for t in target],
                "label": int(label),
            }

        for s in blob.get("subject_results") or []:
            s_id = str(s.get("subject_id"))

            val_qid = f"subj_{s_id}_valid"
            if val_qid in questions:
                is_valid = bool(s.get("is_valid", False))
                item = emit(
                    val_qid, "validation", questions[val_qid],
                    [0.0, 1.0] if is_valid else [1.0, 0.0],
                    1 if is_valid else 0,
                )
                if item:
                    yield item

            prom_qid = f"subj_{s_id}_prominence"
            prom_val = str(s.get("prominence", "")).lower().strip()
            if prom_qid in questions and prom_val in PROMINENCE_MAP:
                label = PROMINENCE_MAP[prom_val]
                target = [0.0, 0.0, 0.0]
                target[label] = 1.0
                item = emit(prom_qid, "prominence", questions[prom_qid], target, label)
                if item:
                    yield item

            sent_qid = f"subj_{s_id}_sentiment"
            sent_val = str(s.get("sentiment", "")).lower().strip()
            if sent_qid in questions and sent_val in SENTIMENT_MAP:
                label = SENTIMENT_MAP[sent_val]
                target = [0.0, 0.0, 0.0, 0.0]
                target[label] = 1.0
                item = emit(sent_qid, "sentiment", questions[sent_qid], target, label)
                if item:
                    yield item

        blob_llm_tags: dict[str, bool] = {}
        for group in (blob.get("scored_attributes") or {}).get("tag_groups", []):
            for t in group.get("tags", []):
                if t.get("evaluation_type") == "llm":
                    blob_llm_tags[str(t.get("tag_id"))] = bool(t.get("result", False))

        for s_cfg in snapshot.get("subjects", []):
            s_id = str(s_cfg["id"])
            for t_cfg in s_cfg.get("tag_evaluations", []):
                if t_cfg.get("evaluation_type") != "llm" or not t_cfg.get("prompt_text"):
                    continue
                t_id = str(t_cfg["tag_id"])
                tag_qid = f"tag_{s_id}_{t_id}"
                if tag_qid not in questions or t_id not in blob_llm_tags:
                    continue
                tag_res = blob_llm_tags[t_id]
                item = emit(
                    tag_qid, "tag", questions[tag_qid],
                    [0.0, 1.0] if tag_res else [1.0, 0.0],
                    1 if tag_res else 0,
                )
                if item:
                    yield item

# COMMAND ----------

# The bronze table accumulates across runs, and the source container holds
# millions of blobs. Training set size is therefore a deliberate choice here,
# not a side effect of how long ingestion was left running.
bronze = spark.table(BRONZE_BLOBS)
if PARSE_LIMIT > 0:
    bronze = bronze.orderBy(F.rand(seed=17)).limit(PARSE_LIMIT)
    print(f"Parsing a {PARSE_LIMIT}-blob sample of {BRONZE_BLOBS}")

sequences_rdd = bronze.rdd.mapPartitions(build_sequences_partition)
sequences = spark.createDataFrame(sequences_rdd, schema=SEQUENCE_SCHEMA).cache()

print(f"Generated {sequences.count()} raw sequences")
display(sequences.groupBy("decision_kind", "label").count().orderBy("decision_kind", "label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Rebalance the tag class
# MAGIC
# MAGIC Roughly 85% of production tags are false. Training on that ratio taught the model
# MAGIC to hedge: true tags came back at p=0.30 to 0.50 and never cleared the decision
# MAGIC threshold. Downsampling the negatives to the target ratio is the fix.
# MAGIC
# MAGIC Validation, prominence, and sentiment are capped per label the same way
# MAGIC `rebalance_kind` did locally.

# COMMAND ----------

from pyspark.sql import Window

tags = sequences.filter(F.col("decision_kind") == "tag")
non_tags = sequences.filter(F.col("decision_kind") != "tag")

pos_tags = tags.filter(F.col("label") == 1)
neg_tags = tags.filter(F.col("label") == 0)

pos_count = pos_tags.count()
neg_count = neg_tags.count()
neg_target = int(pos_count * (1 - TAG_POS_RATIO) / TAG_POS_RATIO)

print(f"Tags before rebalance: {pos_count} positive, {neg_count} negative")

if neg_count > neg_target > 0:
    neg_tags = neg_tags.sample(fraction=neg_target / neg_count, seed=42)
    print(f"Downsampled negatives to ~{neg_target}")

balanced_tags = pos_tags.unionByName(neg_tags)

# Cap the other kinds evenly across their labels.
label_window = Window.partitionBy("decision_kind", "label").orderBy(F.rand(42))
per_label_cap = 12000
balanced_non_tags = (
    non_tags
    .withColumn("_rank", F.row_number().over(label_window))
    .filter(F.col("_rank") <= per_label_cap)
    .drop("_rank")
)

balanced = balanced_tags.unionByName(balanced_non_tags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Split and write
# MAGIC
# MAGIC The split is hashed on `correlation_id` so every sequence from one article lands
# MAGIC on the same side. A random row split would leak the same article text into both
# MAGIC train and validation and inflate the reported accuracy.

# COMMAND ----------

final = (
    balanced
    .withColumn("split", F.when(F.abs(F.hash("correlation_id")) % 100 < 15, "val").otherwise("train"))
)

(
    final.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("split")
    .saveAsTable(GOLD_SEQUENCES)
)

result = spark.table(GOLD_SEQUENCES)
print(f"Wrote {result.count()} sequences to {GOLD_SEQUENCES}")
display(result.groupBy("split", "decision_kind", "label").count().orderBy("split", "decision_kind", "label"))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "table": GOLD_SEQUENCES,
    "total": result.count(),
    "train": result.filter(F.col("split") == "train").count(),
    "val": result.filter(F.col("split") == "val").count(),
}))
