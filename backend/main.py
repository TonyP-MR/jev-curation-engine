import asyncio
import datetime
import os
import json
import re
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
print(f"[startup] ENVIRONMENT={settings.ENVIRONMENT} MYSQL_HOST={settings.MYSQL_HOST} BLOB={settings.AUDIT_BLOB_ACCOUNT_URL}")
from config_manager import config_manager
from audit_index import audit_index
from blob_manager import blob_manager
from prompt_optimizer import PromptOptimizationError, prompt_optimizer
from question_converter import (
    build_article_state,
    build_distilled_article_state,
    convert_config_to_jev_questions,
    extract_subject_aliases,
    strip_boilerplate_and_recirculation,
)
from typesafe_runner import typesafe_runner
from comparator import compare_article_results
from run_logger import run_logger
from error_evaluator import error_evaluator, ErrorEvaluatorError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_rig_api")

app = FastAPI(title="jev-curation-engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory execution job tracking
ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}

BATCH_ARTICLES_PER_CONFIG = 50

class RunBenchmarkRequest(BaseModel):
    blob_names: List[str]
    config_id: Optional[str] = None
    noul_threshold: float = 0.50
    model: Optional[str] = None
    provider: Optional[str] = None
    optimize_prompts: bool = False
    # Baseline LLM cost override (USD/article). None keeps the configured default.
    llm_cost_override_usd: Optional[float] = None


class BatchBenchmarkRequest(BaseModel):
    config_ids: List[str]
    noul_threshold: float = 0.50
    model: Optional[str] = None
    provider: Optional[str] = None
    optimize_prompts: bool = False
    # Baseline LLM cost override (USD/article). None keeps the configured default.
    llm_cost_override_usd: Optional[float] = None


class PreviewRequest(BaseModel):
    blob_name: str
    config_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "typesafe_model": settings.TYPESAFE_MODEL,
        "jev_provider": settings.JEV_PROVIDER,
        "jev_model": settings.OPENROUTER_JEV_MODEL if settings.JEV_PROVIDER.lower() == "openrouter" else settings.TYPESAFE_MODEL,
        "laya_model": getattr(settings, "LAYA_MODEL", "laya:azure:t4"),
        "laya_available": True,
        "latest_audit_config": audit_index.latest_config_id(),
        "prompt_optimization_enabled": settings.PROMPT_OPTIMIZATION_ENABLED,
        "prompt_optimization_key_configured": bool(settings.GEMINI_API_KEY),
    }

@app.get("/api/providers")
def list_providers():
    return {
        "current_provider": settings.JEV_PROVIDER,
        "providers": [
            {
                "id": "laya_azure",
                "name": "Laya Azure GPU (NVIDIA T4 - 39ms)",
                "models": [
                    {
                        "id": "laya:azure:t4",
                        "name": "Laya: Azure Tesla T4 (Fine-Tuned 10K, Cloud GPU ~39ms)",
                        "description": "Running fine-tuned 10K model on dev-002 in uksouth with native FP16 Tensor Cores",
                        "default": True,
                    }
                ],
            },
            {
                "id": "laya_databricks",
                "name": "Laya Databricks (Model Serving GPU)",
                "configured": bool(settings.DATABRICKS_HOST and settings.DATABRICKS_TOKEN),
                "models": [
                    {
                        "id": "laya:databricks",
                        "name": "Laya: Databricks Model Serving (GPU endpoint)",
                        "description": (
                            f"Serving endpoint '{settings.LAYA_DATABRICKS_ENDPOINT}' backed by the "
                            "Unity Catalog model trained on the Databricks GPU cluster"
                        ),
                    },
                    {
                        "id": "laya:databricks:local",
                        "name": "Laya: Databricks Weights (local MPS)",
                        "description": (
                            "Same weights pulled down by scripts/fetch_databricks_model.py "
                            "and run on local Apple Silicon"
                        ),
                    },
                ],
            },
            {
                "id": "openrouter",
                "name": "OpenRouter Jev (Proxy Cloud API)",
                "models": [{"id": "typesafe/jev-1.13", "name": "typesafe/jev-1.13"}],
            },
        ],
    }
@app.get("/api/configs")
def list_configs():
    try:
        return config_manager.list_available_configs()
    except Exception as e:
        logger.error(f"Error listing configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs/{config_id}")
def get_config_details(config_id: str):
    try:
        data = config_manager.fetch_published_config(config_id=config_id)
        jev_questions = convert_config_to_jev_questions(data["snapshot"])
        return {
            "metadata": data["metadata"],
            "subjects_count": len(data["snapshot"].get("subjects", [])),
            "jev_questions_count": len(jev_questions),
            "jev_questions": jev_questions
        }
    except Exception as e:
        logger.error(f"Error getting config {config_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/blobs")
def list_blobs(
    limit: int = 50,
    prefix: str = "",
    config_id: Optional[str] = None,
    include_azure: bool = False,
    refresh: bool = False,
):
    """List processed articles from pipeline_audit_log.

    Blob storage is accessed only after selecting an exact correlation ID.
    `refresh` remains accepted for frontend compatibility and no longer scans Azure.
    """
    try:
        rows = audit_index.list_articles(config_id=config_id, limit=limit)
        if prefix:
            rows = [row for row in rows if row["name"].startswith(prefix)]
        return rows
    except Exception as e:
        logger.error(f"Error listing processed articles from SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/blob-configs")
def blob_configs():
    """Config IDs actually present in the cached blob index, with counts.

    Re-reads the index from disk so pre-warmers running in other processes are seen.
    """
    try:
        blob_manager._index = blob_manager._load_index()
        blobs = blob_manager._index.get("blobs") or {}
        counts: Dict[str, int] = {}
        for b in blobs.values():
            cfg = b.get("config_id") or "unknown"
            counts[cfg] = counts.get(cfg, 0) + 1
        return [{"config_id": k, "count": v} for k, v in sorted(counts.items())]
    except Exception as e:
        logger.error(f"Error computing blob config counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blobs/{blob_name}")
def get_blob_details(blob_name: str):
    try:
        blob = blob_manager.get_blob(blob_name)
        state_text, metadata = build_article_state(blob)
        return {
            "blob_name": blob_name,
            "correlation_id": blob.get("correlation_id"),
            "article_id": blob.get("article_id"),
            "config_id": blob.get("config_id"),
            "tracker_id": blob.get("tracker_id"),
            "metadata": metadata,
            "preview_state": state_text[:1000] + ("..." if len(state_text) > 1000 else ""),
            "llm_provider": blob.get("llm_provider"),
            "llm_model": blob.get("llm_model"),
            "llm_tokens": blob.get("llm_tokens"),
            "subject_results_count": len(blob.get("subject_results", [])),
            "subject_results": blob.get("subject_results", [])
        }
    except Exception as e:
        logger.error(f"Error getting blob {blob_name}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/preview")
def preview_payload(req: PreviewRequest):
    """Assemble (without calling Jev) the exact TypeSafe request for one blob+config,
    alongside the baseline LLM decision archived in the blob."""
    try:
        blob = blob_manager.get_blob(req.blob_name)
        cfg_id = req.config_id or blob.get("config_id")
        if not cfg_id:
            raise HTTPException(status_code=400, detail="No config_id supplied or archived on blob")

        try:
            config_data = config_manager.fetch_published_config(config_id=cfg_id)
        except LookupError:
            config_data = config_manager.fetch_historical_config(cfg_id)
        if not config_data:
            raise HTTPException(status_code=404, detail=f"No published config for {cfg_id}")
        snapshot = config_data["snapshot"]
        questions = convert_config_to_jev_questions(snapshot)
        state_text, state_meta = build_article_state(blob)

        return {
            "blob_name": req.blob_name,
            "config": config_data["metadata"],
            "question_count": len(questions),
            "state_meta": state_meta,
            "typesafe_request": {
                "model": settings.TYPESAFE_MODEL,
                "state": state_text,
                "questions": questions,
            },
            "llm_baseline": {
                "provider": blob.get("llm_provider"),
                "model": blob.get("llm_model"),
                "validation_result": blob.get("validation_result"),
                "subject_results": blob.get("subject_results") or [],
                "scored_attributes": blob.get("scored_attributes") or {},
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building preview for {req.blob_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_benchmark_task(
    job_id: str,
    req: RunBenchmarkRequest,
    work_items: Optional[List[tuple[Optional[str], str]]] = None,
):
    ACTIVE_JOBS[job_id]["status"] = "running"
    comparisons = []
    items = work_items or [(req.config_id, blob_name) for blob_name in req.blob_names]
    total = len(items)
    config_ids = list(dict.fromkeys(cfg_id for cfg_id, _ in items if cfg_id))
    config_totals: Dict[str, int] = {}
    config_completed: Dict[str, int] = {}
    for cfg_id, _ in items:
        if cfg_id:
            config_totals[cfg_id] = config_totals.get(cfg_id, 0) + 1
            config_completed[cfg_id] = 0

    optimized_rubrics: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}
    config_snapshots: Dict[str, Dict[str, Any]] = {}
    lock = asyncio.Lock()
    is_laya_run = (
        (req.provider and req.provider.lower().startswith("laya"))
        or (req.model and ("laya" in str(req.model).lower() or str(req.model).lower().startswith("local:")))
        or settings.JEV_PROVIDER.lower().startswith("laya")
    )
    concurrency_limit = getattr(settings, "LAYA_BENCHMARK_CONCURRENCY", 15) if is_laya_run else settings.BENCHMARK_CONCURRENCY
    sem = asyncio.Semaphore(concurrency_limit)

    try:
        run_logger.create_run_session(
            job_id,
            config_ids[0] if len(config_ids) == 1 else "batch",
            total,
            config_ids=config_ids or None,
        )

        # Pre-fetch and cache configs & rubrics before starting parallel workers
        for cfg_id in config_ids:
            try:
                config_data = config_manager.fetch_published_config(config_id=cfg_id)
            except LookupError:
                config_data = config_manager.fetch_historical_config(cfg_id)
            config_snapshots[cfg_id] = config_data
            if req.optimize_prompts and cfg_id not in optimized_rubrics:
                ACTIVE_JOBS[job_id]["phase"] = "optimizing_prompts"
                ACTIVE_JOBS[job_id]["optimizer_config"] = cfg_id
                ACTIVE_JOBS[job_id]["optimizer_status"] = "compiling"
                optimized_rubrics[cfg_id] = await prompt_optimizer.optimize_snapshot(
                    config_data["snapshot"], config_data["metadata"]
                )

        ACTIVE_JOBS[job_id]["phase"] = "evaluating"

        async def process_one(idx: int, item_config_id: Optional[str], blob_name: str):
            async with sem:
                try:
                    blob_audit = blob_manager.get_blob(blob_name)
                    cfg_id = item_config_id or req.config_id or blob_audit.get("config_id")

                    if cfg_id in config_snapshots:
                        config_data = config_snapshots[cfg_id]
                    else:
                        try:
                            config_data = config_manager.fetch_published_config(config_id=cfg_id)
                        except LookupError:
                            config_data = config_manager.fetch_historical_config(cfg_id)
                        config_snapshots[cfg_id] = config_data

                    snapshot = config_data["snapshot"]
                    optimized_rubric, optimizer_metadata = (
                        optimized_rubrics.get(cfg_id, (None, {"enabled": False}))
                    )
                    is_laya = (
                        (req.provider and req.provider.lower().startswith("laya"))
                        or (req.model and ("laya" in str(req.model).lower() or str(req.model).lower().startswith("local:")))
                        or settings.JEV_PROVIDER.lower().startswith("laya")
                    )
                    jev_questions = convert_config_to_jev_questions(
                        snapshot, optimized_rubric, is_system_one=is_laya
                    )
                    if is_laya:
                        blob_for_distill = dict(blob_audit)
                        if "subjects" in snapshot:
                            blob_for_distill["subjects"] = snapshot["subjects"]
                        state_text, state_meta = build_distilled_article_state(blob_for_distill, max_chars=4000)
                        hl_entities = set(state_meta.get("headline_entities", []))
                        lead_entities = set(state_meta.get("lead_entities", []))

                        # Pipeline Gating: detect completely absent entities (0 mentions of name OR aliases)
                        # and strip recirculation/boilerplate footers to prevent false-positive validation.
                        inbound_data = blob_audit.get("inbound_data", {})
                        raw_body = inbound_data.get("body") or blob_audit.get("body") or ""
                        clean_body = strip_boilerplate_and_recirculation(raw_body)
                        full_text = " ".join([
                            inbound_data.get("headline") or blob_audit.get("headline") or "",
                            clean_body
                        ])
                        gated_answers: Dict[str, Any] = {}
                        active_questions = dict(jev_questions)

                        # Stage 0: Deterministic presence gating & Headline Primacy
                        VENUE_SUFFIXES = ("coliseum", "arena", "stadium", "center", "field", "park", "theater", "theatre", "amphitheatre", "pavilion")
                        for s in snapshot.get("subjects", []):
                            s_id = str(s["id"])
                            s_name = s.get("name")
                            aliases = extract_subject_aliases(s)
                            alias_res = [re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE) for a in aliases]
                            
                            # Filter out naming-rights physical venues (e.g. Coca-Cola Coliseum, Kia Center)
                            # unless article explicitly discusses naming rights, sponsorship, or corporate strategy
                            has_venue_only = False
                            m_all = []
                            for r in alias_res:
                                for match in r.finditer(full_text):
                                    m_all.append(match)
                            
                            if m_all:
                                venue_context_count = 0
                                for m in m_all:
                                    window = full_text[m.start():m.end() + 25].lower()
                                    if any(vs in window for vs in VENUE_SUFFIXES):
                                        venue_context_count += 1
                                if venue_context_count == len(m_all) and not any(w in full_text.lower() for w in ("sponsorship", "naming rights", "partnership agreement", "brand strategy")):
                                    has_venue_only = True

                            has_mention = bool(m_all) and not has_venue_only
                            in_hl = (s_name in hl_entities or any(r.search(inbound_data.get("headline") or blob_audit.get("headline") or "") for r in alias_res)) and not has_venue_only
                            
                            if not has_mention and not in_hl:
                                gated_answers[f"subj_{s_id}_valid"] = {"type": "noul", "noul": 0.0, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                gated_answers[f"subj_{s_id}_prominence"] = {"type": "choice", "choice": "passing", "probabilities": {"primary": 0.0, "significant": 0.0, "passing": 1.0}, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                gated_answers[f"subj_{s_id}_sentiment"] = {"type": "choice", "choice": "neutral", "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "balanced": 0.0}, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                for t in s.get("tag_evaluations", []):
                                    t_id = t["tag_id"]
                                    gated_answers[f"tag_{s_id}_{t_id}"] = {"type": "noul", "noul": 0.0, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                for k in list(active_questions.keys()):
                                    if k.startswith(f"subj_{s_id}_") or k.startswith(f"tag_{s_id}_"):
                                        active_questions.pop(k, None)
                            elif in_hl:
                                # Headline Primacy: Corporate releases naming the entity in the headline are 100% valid
                                gated_answers[f"subj_{s_id}_valid"] = {"type": "noul", "noul": 1.0, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                active_questions.pop(f"subj_{s_id}_valid", None)

                        # Stage 1: Validation Gating (fast single-pass check)
                        val_questions = {k: v for k, v in active_questions.items() if k.endswith("_valid")}
                        val_res = {}
                        if val_questions:
                            val_res = await typesafe_runner.evaluate_article(
                                state=state_text,
                                questions=val_questions,
                                model=req.model,
                                provider=req.provider,
                            )
                            val_answers = val_res.get("answers", {})
                            gated_answers.update(val_answers)

                            val_threshold = getattr(settings, "VALIDATION_THRESHOLD", 0.45)
                            for s in snapshot.get("subjects", []):
                                s_id = str(s["id"])
                                s_name = s.get("name")
                                val_key = f"subj_{s_id}_valid"
                                in_lead = s_name in lead_entities
                                effective_threshold = 0.35 if in_lead else val_threshold
                                if val_key in val_answers:
                                    noul_score = val_answers[val_key].get("noul", 0.0)
                                    if noul_score < effective_threshold:
                                        # Entity failed validation: gate prominence, sentiment, and tags!
                                        gated_answers[f"subj_{s_id}_prominence"] = {"type": "choice", "choice": "passing", "probabilities": {"primary": 0.0, "significant": 0.0, "passing": 1.0}, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                        gated_answers[f"subj_{s_id}_sentiment"] = {"type": "choice", "choice": "neutral", "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "balanced": 0.0}, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                        for t in s.get("tag_evaluations", []):
                                            t_id = t["tag_id"]
                                            gated_answers[f"tag_{s_id}_{t_id}"] = {"type": "noul", "noul": 0.0, "confidence": 1.0, "action": {"act_probability": 1.0}}
                                        for k in list(active_questions.keys()):
                                            if k.startswith(f"subj_{s_id}_") or k.startswith(f"tag_{s_id}_"):
                                                active_questions.pop(k, None)

                        # Stage 2: Downstream Evaluation (only for genuinely valid entities)
                        downstream_questions = {k: v for k, v in active_questions.items() if not k.endswith("_valid")}
                        if downstream_questions:
                            down_res = await typesafe_runner.evaluate_article(
                                state=state_text,
                                questions=downstream_questions,
                                model=req.model,
                                provider=req.provider,
                            )
                            total_dur = val_res.get("duration_ms", 0.0) + down_res.get("duration_ms", 0.0)
                            jev_result = down_res
                            jev_result["duration_ms"] = round(total_dur, 2)
                            jev_result.setdefault("answers", {}).update(gated_answers)
                        elif val_questions:
                            jev_result = val_res
                            jev_result.setdefault("answers", {}).update(gated_answers)
                        else:
                            jev_result = {
                                "provider": (req.provider or "laya_azure").lower(),
                                "model": req.model or "laya:azure:t4:finetuned-10k",
                                "answers": gated_answers,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                                "duration_ms": 1.0,
                                "cost_usd": 0.0,
                            }

                        # Post-Processing: Headline/Lead Primacy, Floor/Ceiling Heuristics & Neutral Sentiment Calibration
                        answers_map = jev_result.get("answers", {})
                        
                        # Pre-calculate mention counts across all subjects for relative share calculation
                        subject_aliases_map = {
                            str(s["id"]): [re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE) for a in extract_subject_aliases(s)]
                            for s in snapshot.get("subjects", [])
                        }
                        subject_mentions_map = {
                            s_id: sum(len(r.findall(full_text)) for r in res_list)
                            for s_id, res_list in subject_aliases_map.items()
                        }
                        max_mentions_in_article = max(subject_mentions_map.values(), default=0)
                        # Identify multi-entity guides/comparisons where 2 or more entities have substantial mentions (>= 3)
                        major_entity_count = sum(1 for cnt in subject_mentions_map.values() if cnt >= 3)
                        is_multi_entity_overview = (major_entity_count >= 2)
                        for s in snapshot.get("subjects", []):
                            s_id = str(s["id"])
                            s_name = s.get("name")
                            alias_res = subject_aliases_map.get(s_id, [])
                            mention_count = subject_mentions_map.get(s_id, 0)
                            in_hl = s_name in hl_entities
                            in_lead = s_name in lead_entities
                            prom_key = f"subj_{s_id}_prominence"
                            sent_key = f"subj_{s_id}_sentiment"
                            valid_key = f"subj_{s_id}_valid"
                            is_valid = answers_map.get(valid_key, {}).get("noul", 0.0) >= 0.45

                            # Prominence floor, ceiling, and relative mention share constraints
                            if is_valid and prom_key in answers_map:
                                p_ans = answers_map[prom_key]
                                current_choice = p_ans.get("choice")
                                
                                # Strict single/zero mention ceiling: An entity with <= 1 mention outside the headline is strictly passing
                                if not in_hl and mention_count <= 1:
                                    p_ans["choice"] = "passing"
                                    p_ans["confidence"] = 0.95
                                # Multi-entity overview ceiling: In a multi-carrier guide/roundup without headline focus,
                                # parallel major competitors cannot be primary (must be significant or passing)
                                elif is_multi_entity_overview and not in_hl:
                                    p_ans["choice"] = "significant" if mention_count >= 3 else "passing"
                                    p_ans["confidence"] = 0.90
                                # Relative share constraint: An entity with far fewer mentions than the dominant entity cannot be primary
                                elif max_mentions_in_article >= 5 and mention_count <= 2 and not in_hl:
                                    p_ans["choice"] = "passing"
                                    p_ans["confidence"] = 0.90
                                # Ceiling constraint: Multi-company list or brief citation cannot be primary
                                else:
                                    in_list_context = any(
                                        bool(
                                            re.search(r'(?:,\s*|\band\s+)' + re.escape(a) + r'(?:,\s*|\band\s+|\s*\))', full_text, re.IGNORECASE)
                                            or re.search(r'\(\s*(?:[^)]*,\s*)?' + re.escape(a) + r'(?:,\s*[^)]*)?\)', full_text, re.IGNORECASE)
                                        )
                                        for a in extract_subject_aliases(s)
                                    )
                                    if (in_list_context or mention_count < 3) and not in_hl:
                                        if current_choice == "primary":
                                            p_ans["choice"] = "significant" if mention_count >= 2 else "passing"
                                            p_ans["confidence"] = 0.85
                                    # Floor constraint: An entity with >= 4 mentions cannot be passing
                                    elif mention_count >= 4 and current_choice == "passing":
                                        p_ans["choice"] = "significant" if mention_count < 8 else "primary"
                                        p_ans["confidence"] = 0.90
                                    elif in_hl and current_choice == "passing":
                                        p_ans["choice"] = "primary" if len(hl_entities) <= 1 else "significant"
                                        p_ans["confidence"] = 0.95
                                    elif in_lead and current_choice == "passing" and mention_count >= 2:
                                        p_ans["choice"] = "significant"
                                        p_ans["confidence"] = 0.90
                            # Recalibrate Sentiment: Neutral baseline for routine business / sports
                            if is_valid and sent_key in answers_map:
                                s_ans = answers_map[sent_key]
                                if s_ans.get("type") == "choice":
                                    probs = dict(s_ans.get("probabilities", {}))
                                    if probs and "neutral" in probs:
                                        # Check for explicit corporate adversity / conflict signals
                                        adversity_markers = ("severed", "cut line", "disrupt", "damage", "complaint", "halted", "stop-work", "lawsuit", "investigat", "outage", "fine", "penalty", "critic")
                                        has_adversity = any(m in full_text.lower() for m in adversity_markers)
                                        raw_neg = probs.get("negative", 0.0)
                                        
                                        if has_adversity and raw_neg >= 0.25:
                                            # Do not dilute negative sentiment when concrete adversity exists
                                            probs["negative"] = raw_neg * 1.5
                                            probs["neutral"] = probs.get("neutral", 0.0) * 0.9
                                        else:
                                            probs["neutral"] = probs.get("neutral", 0.0) * 1.8
                                            probs["positive"] = probs.get("positive", 0.0) * 0.55
                                            probs["negative"] = probs.get("negative", 0.0) * 0.85
                                        
                                        probs["balanced"] = probs.get("balanced", 0.0) * 0.5
                                        s_tot = sum(probs.values())
                                        if s_tot > 0:
                                            probs = {k: round(v / s_tot, 4) for k, v in probs.items()}
                                        s_ans["probabilities"] = probs
                                        s_ans["choice"] = max(probs, key=probs.get)
                    else:
                        state_text, _ = build_article_state(blob_audit)
                        jev_result = await typesafe_runner.evaluate_article(
                            state=state_text,
                            questions=jev_questions,
                            model=req.model,
                            provider=req.provider,
                        )

                    comparison = compare_article_results(
                        blob_audit=blob_audit,
                        jev_result=jev_result,
                        snapshot=snapshot,
                        noul_threshold=req.noul_threshold,
                        llm_cost_override_usd=req.llm_cost_override_usd,
                        tag_threshold=req.noul_threshold if req.noul_threshold != 0.50 else getattr(settings, "TAG_THRESHOLD", 0.40),
                    )
                    comparison["config_version"] = config_data["metadata"]
                    comparison["prompt_optimization"] = optimizer_metadata
                    comparison["jev_request"] = jev_result.get("request_payload")
                    comparison["jev_response"] = jev_result.get("raw_response")

                    corr_id = blob_audit.get("correlation_id") or blob_name.replace(".json", "")
                    record_key = f"{cfg_id}__{corr_id}" if work_items is not None else None
                    await asyncio.to_thread(run_logger.log_article_result, job_id, corr_id, comparison, record_key=record_key)

                    async with lock:
                        comparisons.append(comparison)
                        processed_count = len(comparisons)
                        ACTIVE_JOBS[job_id]["processed"] = processed_count
                        ACTIVE_JOBS[job_id]["current_index"] = processed_count
                        ACTIVE_JOBS[job_id]["current_blob"] = blob_name
                        if item_config_id:
                            config_completed[item_config_id] = config_completed.get(item_config_id, 0) + 1
                            ACTIVE_JOBS[job_id].update({
                                "current_config_id": item_config_id,
                                "config_article_index": config_completed[item_config_id],
                                "config_article_total": config_totals.get(item_config_id, 0),
                                "config_index": (config_ids.index(item_config_id) + 1) if item_config_id in config_ids else 1,
                                "config_total": len(config_ids),
                            })

                except LookupError as skip_err:
                    reason = str(skip_err)
                    logger.warning(f"Skipping {blob_name}: {reason}")
                    skipped = {
                        "blob_name": blob_name,
                        "correlation_id": blob_name.replace(f"{settings.AUDIT_BLOB_SUFFIX}.json", "").replace(".json", ""),
                        "config_id": item_config_id or req.config_id,
                        "skipped": True,
                        "reason": reason,
                    }
                    record_key = (
                        f"{item_config_id or req.config_id}__{skipped['correlation_id']}"
                        if work_items is not None else None
                    )
                    await asyncio.to_thread(
                        run_logger.log_article_result,
                        job_id,
                        skipped["correlation_id"] or blob_name,
                        skipped,
                        record_key=record_key,
                    )
                    async with lock:
                        ACTIVE_JOBS[job_id].setdefault("skipped", []).append(blob_name)
                except Exception as blob_err:
                    logger.exception(f"Article {blob_name} failed: {blob_err}")
                    failed_record = {
                        "blob_name": blob_name,
                        "correlation_id": blob_name.replace(f"{settings.AUDIT_BLOB_SUFFIX}.json", "").replace(".json", ""),
                        "config_id": item_config_id or req.config_id,
                        "failed": True,
                        "error": str(blob_err),
                    }
                    record_key = (
                        f"{item_config_id or req.config_id}__{failed_record['correlation_id']}"
                        if work_items is not None else None
                    )
                    await asyncio.to_thread(
                        run_logger.log_article_result,
                        job_id,
                        failed_record["correlation_id"] or blob_name,
                        failed_record,
                        record_key=record_key,
                    )
                    async with lock:
                        ACTIVE_JOBS[job_id].setdefault("failed_blobs", []).append(
                            {"blob_name": blob_name, "error": str(blob_err)}
                        )

        tasks = [
            process_one(idx, item_config_id, blob_name)
            for idx, (item_config_id, blob_name) in enumerate(items)
        ]
        await asyncio.gather(*tasks)

        summary = run_logger.finalize_run(
            job_id,
            comparisons,
            failed_articles=len(ACTIVE_JOBS[job_id].get("failed_blobs", [])),
        )
        ACTIVE_JOBS[job_id]["status"] = "completed"
        ACTIVE_JOBS[job_id]["summary"] = summary
        ACTIVE_JOBS[job_id]["comparisons"] = comparisons

    except Exception as e:
        logger.exception(f"Error in benchmark job {job_id}: {e}")
        ACTIVE_JOBS[job_id]["status"] = "failed"
        ACTIVE_JOBS[job_id]["error"] = str(e)


@app.post("/api/benchmark/run")
async def start_benchmark(req: RunBenchmarkRequest, background_tasks: BackgroundTasks):
    if not req.blob_names:
        raise HTTPException(status_code=400, detail="Must provide at least one blob name")
    if req.optimize_prompts and not settings.PROMPT_OPTIMIZATION_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Prompt optimization is disabled; set PROMPT_OPTIMIZATION_ENABLED=true",
        )
    if req.optimize_prompts and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Prompt optimization requires GEMINI_API_KEY in the environment",
        )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"run_{timestamp}_{len(req.blob_names)}_items"
    
    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "phase": "starting",
        "total": len(req.blob_names),
        "processed": 0,
        "optimize_prompts": req.optimize_prompts,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    background_tasks.add_task(execute_benchmark_task, job_id, req)
    return {"job_id": job_id, "status": "started", "total": len(req.blob_names)}

@app.post("/api/benchmark/batch-run")
async def start_batch_benchmark(req: BatchBenchmarkRequest, background_tasks: BackgroundTasks):
    config_ids = list(dict.fromkeys(config_id.strip() for config_id in req.config_ids if config_id and config_id.strip()))
    if not config_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one config ID")
    if req.optimize_prompts and not settings.PROMPT_OPTIMIZATION_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Prompt optimization is disabled; set PROMPT_OPTIMIZATION_ENABLED=true",
        )
    if req.optimize_prompts and not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Prompt optimization requires GEMINI_API_KEY in the environment",
        )

    # Select the newest indexed articles once, then preserve config and article
    # order while the background task evaluates them sequentially.
    config_articles: Dict[str, List[str]] = {}
    config_counts = []
    for config_id in config_ids:
        articles = audit_index.list_articles(config_id=config_id, limit=BATCH_ARTICLES_PER_CONFIG)
        names = [article["name"] for article in articles]
        config_articles[config_id] = names
        config_counts.append({
            "config_id": config_id,
            "requested": BATCH_ARTICLES_PER_CONFIG,
            "selected": len(names),
        })

    work_items: List[tuple[Optional[str], str]] = []
    max_count = max((len(names) for names in config_articles.values()), default=0)
    for idx in range(max_count):
        for cid in config_ids:
            names = config_articles.get(cid, [])
            if idx < len(names):
                work_items.append((cid, names[idx]))
    if not work_items:
        raise HTTPException(status_code=400, detail="None of the selected configs has processed articles")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_id = f"batch_run_{timestamp}_{len(config_ids)}_configs"
    total = len(work_items)
    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "phase": "starting",
        "mode": "batch",
        "config_ids": config_ids,
        "config_counts": config_counts,
        "articles_per_config": BATCH_ARTICLES_PER_CONFIG,
        "total": total,
        "processed": 0,
        "optimize_prompts": req.optimize_prompts,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    execution_req = RunBenchmarkRequest(
        blob_names=[],
        noul_threshold=req.noul_threshold,
        model=req.model,
        provider=req.provider,
        optimize_prompts=req.optimize_prompts,
        llm_cost_override_usd=req.llm_cost_override_usd,
    )
    background_tasks.add_task(execute_benchmark_task, job_id, execution_req, work_items)
    return {
        "job_id": job_id,
        "status": "started",
        "mode": "batch",
        "config_counts": config_counts,
        "total": total,
    }

@app.get("/api/benchmark/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in ACTIVE_JOBS:
        # Check if saved on disk
        saved = run_logger.get_run(job_id)
        if saved:
            return {"job_id": job_id, "status": "completed", "summary": saved.get("summary")}
        raise HTTPException(status_code=404, detail="Job not found")
    return ACTIVE_JOBS[job_id]

@app.get("/api/runs")
def list_runs():
    return run_logger.list_runs()

@app.get("/api/runs/{run_id}")
def get_run_details(run_id: str):
    run = run_logger.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

class AnalyzeErrorsRequest(BaseModel):
    sample_size: Optional[int] = 15


@app.post("/api/runs/{run_id}/analyze-errors")
async def analyze_run_errors(run_id: str, req: Optional[AnalyzeErrorsRequest] = None):
    run = run_logger.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    sample_size = req.sample_size if req is not None else 15

    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured in the environment")

    def get_article_text(record: Dict[str, Any]) -> str:
        blob_name = record.get("blob_name")
        if blob_name:
            try:
                blob = blob_manager.get_blob(blob_name)
                state_text, _ = build_article_state(blob)
                return state_text
            except Exception:
                pass
        jev_req = record.get("jev_request") or {}
        return jev_req.get("state") or record.get("headline", "")
    try:
        analysis = await error_evaluator.run_error_analysis(
            run_id=run_id,
            records=run.get("records", []),
            get_article_text_fn=get_article_text,
            sample_size=sample_size,
        )
        run_logger.save_error_analysis(run_id, analysis)
        return analysis
    except ErrorEvaluatorError as e:
        logger.error(f"Error analysis failed for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error analyzing {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/error-analysis")
def get_error_analysis(run_id: str):
    run = run_logger.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    analysis = run.get("error_analysis")
    if not analysis:
        raise HTTPException(status_code=404, detail="No error analysis found for this run")
    return analysis


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    # Refuse to delete a run whose job is still executing.
    job = ACTIVE_JOBS.get(run_id)
    if job and job.get("status") in ("running", "pending", "starting"):
        raise HTTPException(status_code=409, detail="Run is still executing")
    if not run_logger.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    ACTIVE_JOBS.pop(run_id, None)
    return {"deleted": run_id}


@app.delete("/api/runs")
def clear_runs():
    running = [
        jid for jid, job in ACTIVE_JOBS.items()
        if job.get("status") in ("running", "pending", "starting")
    ]
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot clear while runs are executing: {', '.join(running)}"
        )
    removed = run_logger.clear_all_runs()
    ACTIVE_JOBS.clear()
    return {"removed": removed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
