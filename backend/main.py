import asyncio
import datetime
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
from question_converter import build_article_state, convert_config_to_jev_questions
from typesafe_runner import typesafe_runner
from comparator import compare_article_results
from run_logger import run_logger

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

class RunBenchmarkRequest(BaseModel):
    blob_names: List[str]
    config_id: Optional[str] = None
    noul_threshold: float = 0.50
    model: Optional[str] = None
    # Baseline LLM cost override (USD/article). None keeps the configured default.
    llm_cost_override_usd: Optional[float] = None


class PreviewRequest(BaseModel):
    blob_name: str
    config_id: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "typesafe_model": settings.TYPESAFE_MODEL,
        "latest_audit_config": audit_index.latest_config_id(),
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

        config_data = config_manager.fetch_published_config(config_id=cfg_id)
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


async def execute_benchmark_task(job_id: str, req: RunBenchmarkRequest):
    ACTIVE_JOBS[job_id]["status"] = "running"
    comparisons = []
    total = len(req.blob_names)

    try:
        run_folder = run_logger.create_run_session(job_id, req.config_id or "auto", total)
        for idx, blob_name in enumerate(req.blob_names):
            ACTIVE_JOBS[job_id]["current_index"] = idx + 1
            ACTIVE_JOBS[job_id]["current_blob"] = blob_name

            try:
                # 1. Fetch blob audit
                blob_audit = blob_manager.get_blob(blob_name)
                cfg_id = req.config_id or blob_audit.get("config_id")
                if not cfg_id:
                    raise LookupError("blob has no config_id")

                # 2. Fetch config snapshot & convert questions
                config_data = config_manager.fetch_published_config(config_id=cfg_id)
                snapshot = config_data["snapshot"]
                jev_questions = convert_config_to_jev_questions(snapshot)

                # 3. Build state
                state_text, _ = build_article_state(blob_audit)

                # 4. Evaluate with Jev
                jev_result = await typesafe_runner.evaluate_article(
                    state=state_text,
                    questions=jev_questions,
                    model=req.model
                )

                # 5. Compare against LLM baseline
                comparison = compare_article_results(
                    blob_audit=blob_audit,
                    jev_result=jev_result,
                    snapshot=snapshot,
                    noul_threshold=req.noul_threshold,
                    llm_cost_override_usd=req.llm_cost_override_usd,
                )
                # Observability: retain the exact wire payload and response for this article.
                comparison["config_version"] = config_data["metadata"]
                comparison["jev_request"] = jev_result.get("request_payload")
                comparison["jev_response"] = jev_result.get("raw_response")

                # 6. Log record
                corr_id = blob_audit.get("correlation_id") or blob_name.replace(".json", "")
                run_logger.log_article_result(job_id, corr_id, comparison)
                comparisons.append(comparison)
                ACTIVE_JOBS[job_id]["processed"] = len(comparisons)

            except LookupError as skip_err:
                # Prod blobs can reference configs that are archived or lack an
                # active published snapshot; record and continue the run.
                reason = str(skip_err)
                logger.warning(f"Skipping {blob_name}: {reason}")
                skipped = {
                    "blob_name": blob_name,
                    "correlation_id": blob_name.replace(f"{settings.AUDIT_BLOB_SUFFIX}.json", "").replace(".json", ""),
                    "skipped": True,
                    "reason": reason,
                }
                run_logger.log_article_result(job_id, skipped["correlation_id"] or blob_name, skipped)
                ACTIVE_JOBS[job_id].setdefault("skipped", []).append(blob_name)
            except Exception as blob_err:
                logger.exception(f"Article {blob_name} failed; continuing run: {blob_err}")
                ACTIVE_JOBS[job_id].setdefault("failed_blobs", []).append(
                    {"blob_name": blob_name, "error": str(blob_err)}
                )

        # Finalize run
        summary = run_logger.finalize_run(job_id, comparisons)
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"run_{timestamp}_{len(req.blob_names)}_items"
    
    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "total": len(req.blob_names),
        "processed": 0,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    background_tasks.add_task(execute_benchmark_task, job_id, req)
    return {"job_id": job_id, "status": "started", "total": len(req.blob_names)}

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
