import glob
import json
import logging
import os
import random
from typing import Any

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options
from transformers import AutoTokenizer

from config import settings
from config_manager import config_manager
from laya_questions import (
    HEAD_MAX_LEN,
    MAX_LEN,
    PROMINENCE_MAP,
    SENTIMENT_MAP,
    build_laya_questions,
    build_laya_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_curation_engine_dataset(
    model_id: str = "convaiinnovations/laya",
    subfolder: str | None = "typed-decisions",
    blobs_limit: int = 10000,
    output_dir: str = "../runs/laya_training_data_10k",
    val_split: float = 0.15,
    cap_per_kind: int = 12000,
) -> dict[str, Any]:
    """Extracts ground truth from cached Curation Engine blobs and formats them into

    tokenized sequences for Laya RLCD fine-tuning.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading tokenizer and agent config from {model_id}...")
    kw = {}
    if subfolder:
        kw["allow_patterns"] = [f"{subfolder}/*"]
    raw_dir = snapshot_download(model_id, **kw)
    model_dir = os.path.join(raw_dir, subfolder) if subfolder else raw_dir
    _fix_tokenizer_config(model_dir)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    # Budgets are fixed by the canonical module, not read from the checkpoint, so a
    # config change cannot silently retrain at a width serving does not use.
    max_len = MAX_LEN
    head_max_len = HEAD_MAX_LEN
    if (cfg.get("max_len"), cfg.get("head_max_len")) != (max_len, head_max_len):
        logger.warning(
            "Checkpoint cfg budgets %s/%s differ from canonical %s/%s; building at canonical.",
            cfg.get("max_len"), cfg.get("head_max_len"), max_len, head_max_len,
        )

    blobs_pattern = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".cache", "production", "blobs", "*.json")
    )
    blob_paths = sorted(glob.glob(blobs_pattern))
    logger.info(f"Found {len(blob_paths)} cached blobs. Processing up to {blobs_limit}...")

    random.seed(42)
    selected_paths = blob_paths[:blobs_limit]

    items: list[dict[str, Any]] = []
    cfg_cache: dict[str, Any] = {}
    skipped_count = 0

    for idx, path in enumerate(selected_paths):
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            continue

        cfg_id = blob.get("config_id")
        if not cfg_id:
            continue
        if cfg_id not in cfg_cache:
            cfg_disk_path = os.path.join(settings.CACHE_DIR, "configs", f"{cfg_id}.json")
            if os.path.exists(cfg_disk_path):
                try:
                    with open(cfg_disk_path, "r", encoding="utf-8") as cf:
                        cfg_cache[cfg_id] = json.load(cf)
                except Exception:
                    cfg_cache[cfg_id] = None
            else:
                cfg_cache[cfg_id] = None

        if not cfg_cache.get(cfg_id):
            continue

        snapshot = cfg_cache[cfg_id]["snapshot"]
        questions = build_laya_questions(snapshot)
        state_text = build_laya_state(blob)

        # 1. Subject-level ground truth (validation, prominence, sentiment)
        subj_results = blob.get("subject_results") or []
        for s in subj_results:
            s_id = str(s.get("subject_id"))

            # Validation (noul)
            val_qid = f"subj_{s_id}_valid"
            if val_qid in questions:
                is_valid = bool(s.get("is_valid", False))
                val_target = [0.0, 1.0] if is_valid else [1.0, 0.0]
                val_label = 1 if is_valid else 0
                q_def = questions[val_qid]
                internal_q = {"t": "noul", "ins": q_def["instructions"], "crit": q_def.get("criteria", {})}
                try:
                    seq, markers = build_sequence(tok, state_text, internal_q, max_len, head_max_len)
                    k = len(render_options(internal_q))
                    if len(markers) == k:
                        items.append({
                            "ids": seq,
                            "markers": markers,
                            "qtype": QTYPES["noul"],
                            "target": val_target,
                            "label": val_label,
                            "qid": val_qid,
                            "decision_kind": "validation",
                        })
                    else:
                        skipped_count += 1
                except (ValueError, KeyError, IndexError):
                    skipped_count += 1

            # Prominence (choice: primary, significant, passing)
            prom_qid = f"subj_{s_id}_prominence"
            prom_val = str(s.get("prominence", "")).lower().strip()
            if prom_qid in questions and prom_val in PROMINENCE_MAP:
                prom_label = PROMINENCE_MAP[prom_val]
                prom_target = [0.0, 0.0, 0.0]
                prom_target[prom_label] = 1.0
                q_def = questions[prom_qid]
                internal_q = {"t": "choice", "ins": q_def["instructions"], "crit": q_def.get("criteria", {})}
                try:
                    seq, markers = build_sequence(tok, state_text, internal_q, max_len, head_max_len)
                    k = len(render_options(internal_q))
                    if len(markers) == k:
                        items.append({
                            "ids": seq,
                            "markers": markers,
                            "qtype": QTYPES["choice"],
                            "target": prom_target,
                            "label": prom_label,
                            "qid": prom_qid,
                            "decision_kind": "prominence",
                        })
                    else:
                        skipped_count += 1
                except (ValueError, KeyError, IndexError):
                    skipped_count += 1

            # Sentiment (choice: positive, negative, neutral, balanced)
            sent_qid = f"subj_{s_id}_sentiment"
            sent_val = str(s.get("sentiment", "")).lower().strip()
            if sent_qid in questions and sent_val in SENTIMENT_MAP:
                sent_label = SENTIMENT_MAP[sent_val]
                sent_target = [0.0, 0.0, 0.0, 0.0]
                sent_target[sent_label] = 1.0
                q_def = questions[sent_qid]
                internal_q = {"t": "choice", "ins": q_def["instructions"], "crit": q_def.get("criteria", {})}
                try:
                    seq, markers = build_sequence(tok, state_text, internal_q, max_len, head_max_len)
                    k = len(render_options(internal_q))
                    if len(markers) == k:
                        items.append({
                            "ids": seq,
                            "markers": markers,
                            "qtype": QTYPES["choice"],
                            "target": sent_target,
                            "label": sent_label,
                            "qid": sent_qid,
                            "decision_kind": "sentiment",
                        })
                    else:
                        skipped_count += 1
                except (ValueError, KeyError, IndexError):
                    skipped_count += 1

        # 2. Tag-level ground truth (noul)
        blob_llm_tags: dict[str, bool] = {}
        scored_attrs = blob.get("scored_attributes") or {}
        for group in scored_attrs.get("tag_groups", []):
            for t in group.get("tags", []):
                if t.get("evaluation_type") == "llm":
                    blob_llm_tags[str(t.get("tag_id"))] = bool(t.get("result", False))

        for s_cfg in snapshot.get("subjects", []):
            s_id = str(s_cfg["id"])
            for t_cfg in s_cfg.get("tag_evaluations", []):
                if t_cfg.get("evaluation_type") == "llm" and t_cfg.get("prompt_text"):
                    t_id = str(t_cfg["tag_id"])
                    tag_qid = f"tag_{s_id}_{t_id}"
                    if tag_qid in questions and t_id in blob_llm_tags:
                        tag_res = blob_llm_tags[t_id]
                        tag_target = [0.0, 1.0] if tag_res else [1.0, 0.0]
                        tag_label = 1 if tag_res else 0
                        q_def = questions[tag_qid]
                        internal_q = {"t": "noul", "ins": q_def["instructions"], "crit": q_def.get("criteria", {})}
                        try:
                            seq, markers = build_sequence(tok, state_text, internal_q, max_len, head_max_len)
                            k = len(render_options(internal_q))
                            if len(markers) == k:
                                items.append({
                                    "ids": seq,
                                    "markers": markers,
                                    "qtype": QTYPES["noul"],
                                    "target": tag_target,
                                    "label": tag_label,
                                    "qid": tag_qid,
                                    "decision_kind": "tag",
                                })
                            else:
                                skipped_count += 1
                        except (ValueError, KeyError, IndexError):
                            skipped_count += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == len(selected_paths):
            logger.info(f"Processed {idx + 1}/{len(selected_paths)} blobs -> {len(items)} training sequences")

    logger.info(f"Dataset extraction finished: {len(items)} total sequences (skipped {skipped_count})")

    # Balanced task sampling across validation, prominence, sentiment, and tags
    by_kind: dict[str, list[dict[str, Any]]] = {
        "validation": [it for it in items if it["decision_kind"] == "validation"],
        "prominence": [it for it in items if it["decision_kind"] == "prominence"],
        "sentiment": [it for it in items if it["decision_kind"] == "sentiment"],
        "tag": [it for it in items if it["decision_kind"] == "tag"],
    }

    # For prominence and sentiment, prioritize under-represented non-default classes
    def rebalance_kind(kind_items: list[dict[str, Any]], target_cap: int) -> list[dict[str, Any]]:
        by_label: dict[int, list[dict[str, Any]]] = {}
        for it in kind_items:
            by_label.setdefault(it["label"], []).append(it)
        per_label = max(10, target_cap // max(1, len(by_label)))
        result = []
        for lbl_items in by_label.values():
            random.shuffle(lbl_items)
            result.extend(lbl_items[:per_label])
        return result

    cap = cap_per_kind
    balanced_items: list[dict[str, Any]] = []
    balanced_items.extend(rebalance_kind(by_kind["validation"], cap))
    balanced_items.extend(rebalance_kind(by_kind["prominence"], cap))
    balanced_items.extend(rebalance_kind(by_kind["sentiment"], cap))
    # Sample tags up to cap
    tag_sample = by_kind["tag"][:]
    random.shuffle(tag_sample)
    balanced_items.extend(tag_sample[:cap])
    random.shuffle(balanced_items)
    n_val = max(10, int(len(balanced_items) * val_split))
    val_items = balanced_items[:n_val]
    train_items = balanced_items[n_val:]

    train_path = os.path.join(output_dir, "train_items.pt")
    val_path = os.path.join(output_dir, "val_items.pt")
    torch.save(train_items, train_path)
    torch.save(val_items, val_path)
    stats = {
        "total_blobs_processed": len(selected_paths),
        "total_sequences": len(items),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_path": train_path,
        "val_path": val_path,
        "by_decision_kind": {
            "validation": sum(1 for it in items if it["decision_kind"] == "validation"),
            "prominence": sum(1 for it in items if it["decision_kind"] == "prominence"),
            "sentiment": sum(1 for it in items if it["decision_kind"] == "sentiment"),
            "tag": sum(1 for it in items if it["decision_kind"] == "tag"),
        },
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved dataset stats to {stats_path}: {stats}")
    return stats


if __name__ == "__main__":
    build_curation_engine_dataset(blobs_limit=10000)
