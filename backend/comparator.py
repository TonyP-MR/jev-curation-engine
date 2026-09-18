import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


def _find_llm_stage_duration_ms(blob_audit: Dict[str, Any]) -> float:
    """Locate the pipeline LLM stage duration.

    The archived stage name is `LLMProcessingStage`; older/newer pipeline revisions
    may differ, so match case-insensitively on any stage containing "llm".
    """
    for stage in blob_audit.get("stage_timings") or []:
        name = str(stage.get("stage", "")).lower()
        if "llm" in name:
            return float(stage.get("duration_ms", 0.0))
    return 0.0


def estimate_llm_cost_usd(
    blob_audit: Dict[str, Any],
    flat_override_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve the baseline LLM cost for one article.

    Precedence: blob-recorded cost > caller override > flat settings override >
    token-based estimate. Returns the value plus its provenance.
    """
    recorded = blob_audit.get("llm_cost_usd")
    if recorded is not None:
        return {"cost_usd": float(recorded), "source": "recorded_in_blob"}

    if flat_override_usd is not None:
        return {"cost_usd": float(flat_override_usd), "source": "request_override"}

    if settings.LLM_FLAT_COST_PER_ARTICLE_USD > 0:
        return {
            "cost_usd": float(settings.LLM_FLAT_COST_PER_ARTICLE_USD),
            "source": "flat_override",
        }

    tokens = blob_audit.get("llm_tokens") or {}
    in_tok = int(tokens.get("input", 0) or 0)
    out_tok = int(tokens.get("output", 0) or 0)
    cost = (
        in_tok / 1_000_000.0 * settings.LLM_INPUT_COST_PER_MTOK
        + out_tok / 1_000_000.0 * settings.LLM_OUTPUT_COST_PER_MTOK
    )
    return {"cost_usd": cost, "source": "token_estimate"}


def estimate_classification_cost_usd(
    blob_audit: Dict[str, Any],
    snapshot: Dict[str, Any],
    subject_comparisons: List[Dict[str, Any]],
    tag_comparisons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate LLM cost for only the four structured classification decisions.

    Audit blobs provide aggregate token counts, not field-level attribution. We
    therefore remove known summary/rejection input text and estimate comparable
    output tokens from the compact classification-only JSON. The result is labeled
    estimated and must not replace the full-call billing metric.
    """
    import json

    tokens = blob_audit.get("llm_tokens") or {}
    input_tokens = int(tokens.get("input", 0) or 0)
    cached_tokens = int(blob_audit.get("llm_cached_tokens", 0) or 0)
    summary = snapshot.get("summary_prompt") or {}
    excluded_text = (
        summary.get("prompt_text", "")
        + summary.get("rejected_prompt_text", "")
        + "Summary instructions Rejection reason instructions"
    )
    excluded_input_tokens = max(0, round(len(excluded_text) / 4))
    comparable_input_tokens = max(0, input_tokens - cached_tokens - excluded_input_tokens)

    classification_output = {
        "validation_result": blob_audit.get("validation_result"),
        "subjects": [
            {
                "subject_id": c["subject_id"],
                "is_valid": c["validation"]["llm"],
                "prominence": c["prominence"]["llm"],
                "sentiment": c["sentiment"]["llm"],
            }
            for c in subject_comparisons
        ],
        "tags": [
            {"subject_id": t["subject_id"], "tag_id": t["tag_id"], "result": t["llm_result"]}
            for t in tag_comparisons
        ],
    }
    comparable_output_tokens = max(
        0, round(len(json.dumps(classification_output, separators=(",", ":"))) / 4)
    )
    cost = (
        comparable_input_tokens / 1_000_000 * settings.LLM_INPUT_COST_PER_MTOK
        + comparable_output_tokens / 1_000_000 * settings.LLM_OUTPUT_COST_PER_MTOK
    )
    return {
        "cost_usd": cost,
        "source": "estimated_classification_only",
        "input_tokens": comparable_input_tokens,
        "output_tokens": comparable_output_tokens,
        "excluded_input_tokens": excluded_input_tokens,
        "excluded_output_fields": ["summary", "rejection_reason", "reasoning"],
    }


def compare_article_results(
    blob_audit: Dict[str, Any],
    jev_result: Dict[str, Any],
    snapshot: Dict[str, Any],
    noul_threshold: float = 0.50,
    llm_cost_override_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare LLM audit ground truth against TypeSafe Jev answers for elements 1-4:
    subject validation, prominence, sentiment, and LLM tags.
    """
    answers = jev_result.get("answers", {})
    subject_results = blob_audit.get("subject_results", [])
    
    # Map blob subject results by subject_id
    llm_subjects: Dict[str, Dict[str, Any]] = {}
    for s in subject_results:
        s_id = str(s.get("subject_id"))
        llm_subjects[s_id] = s

    # Map snapshot subjects for reference
    snapshot_subjects = {str(s["id"]): s for s in snapshot.get("subjects", [])}

    subject_comparisons = []
    val_correct = 0
    val_total = 0
    prom_correct = 0
    prom_total = 0
    sent_correct = 0
    sent_total = 0

    for s_id, s_info in snapshot_subjects.items():
        name = s_info.get("name") or f"Subject {s_id}"
        llm_sub = llm_subjects.get(s_id)
        if not llm_sub:
            continue

        # 1. Validation comparison
        llm_is_valid = bool(llm_sub.get("is_valid", False))
        jev_val_ans = answers.get(f"subj_{s_id}_valid", {})
        jev_val_prob = jev_val_ans.get("noul", 0.0) if jev_val_ans.get("type") == "noul" else 0.0
        jev_is_valid = jev_val_prob >= noul_threshold
        val_match = (llm_is_valid == jev_is_valid)
        val_total += 1
        if val_match:
            val_correct += 1

        # 2. Prominence comparison
        llm_prom = str(llm_sub.get("prominence", "")).lower()
        jev_prom_ans = answers.get(f"subj_{s_id}_prominence", {})
        jev_prom = str(jev_prom_ans.get("choice", "")).lower()
        jev_prom_conf = jev_prom_ans.get("confidence", 0.0)
        prom_match = (llm_prom == jev_prom)
        prom_total += 1
        if prom_match:
            prom_correct += 1

        # 3. Sentiment comparison
        llm_sent = str(llm_sub.get("sentiment", "")).lower()
        jev_sent_ans = answers.get(f"subj_{s_id}_sentiment", {})
        jev_sent = str(jev_sent_ans.get("choice", "")).lower()
        jev_sent_conf = jev_sent_ans.get("confidence", 0.0)
        sent_match = (llm_sent == jev_sent)
        sent_total += 1
        if sent_match:
            sent_correct += 1

        subject_comparisons.append({
            "subject_id": s_id,
            "name": name,
            "validation": {
                "llm": llm_is_valid,
                "jev": jev_is_valid,
                "jev_prob": jev_val_prob,
                "match": val_match
            },
            "prominence": {
                "llm": llm_prom,
                "jev": jev_prom,
                "jev_conf": jev_prom_conf,
                "match": prom_match
            },
            "sentiment": {
                "llm": llm_sent,
                "jev": jev_sent,
                "jev_conf": jev_sent_conf,
                "match": sent_match
            }
        })

    # 4. LLM Tag evaluations
    # Gather LLM tags from scored_attributes or raw response
    blob_llm_tags: Dict[str, bool] = {}
    scored_attrs = blob_audit.get("scored_attributes", {})
    for group in scored_attrs.get("tag_groups", []):
        for t in group.get("tags", []):
            if t.get("evaluation_type") == "llm":
                blob_llm_tags[str(t.get("tag_id"))] = bool(t.get("result", False))

    tag_comparisons = []
    tag_correct = 0
    tag_total = 0

    for s in snapshot.get("subjects", []):
        s_id = str(s["id"])
        for t in s.get("tag_evaluations", []):
            if t.get("evaluation_type") == "llm" and t.get("prompt_text"):
                t_id = str(t["tag_id"])
                t_name = t.get("tag_name") or t_id
                q_key = f"tag_{s_id}_{t_id}"
                if q_key in answers:
                    jev_tag_ans = answers[q_key]
                    jev_tag_prob = jev_tag_ans.get("noul", 0.0)
                    jev_tag_res = jev_tag_prob >= noul_threshold
                    
                    llm_tag_res = blob_llm_tags.get(t_id, False)
                    tag_match = (llm_tag_res == jev_tag_res)
                    tag_total += 1
                    if tag_match:
                        tag_correct += 1

                    tag_comparisons.append({
                        "subject_id": s_id,
                        "tag_id": t_id,
                        "tag_name": t_name,
                        "llm_result": llm_tag_res,
                        "jev_result": jev_tag_res,
                        "jev_prob": jev_tag_prob,
                        "match": tag_match
                    })

    # Baseline LLM timing and cost.
    llm_duration_ms = _find_llm_stage_duration_ms(blob_audit)

    llm_tokens = blob_audit.get("llm_tokens") or {}
    llm_input_tokens = int(llm_tokens.get("input", 0) or 0)
    llm_output_tokens = int(llm_tokens.get("output", 0) or 0)
    llm_cost_info = estimate_llm_cost_usd(blob_audit, llm_cost_override_usd)
    llm_cost_usd = llm_cost_info["cost_usd"]
    classification_cost_info = estimate_classification_cost_usd(
        blob_audit, snapshot, subject_comparisons, tag_comparisons
    )

    jev_duration_ms = jev_result.get("duration_ms", 0.0)
    jev_cost_usd = jev_result.get("cost_usd", 0.0)
    speedup_ratio = round(llm_duration_ms / max(jev_duration_ms, 1.0), 2) if llm_duration_ms > 0 else 1.0
    classification_cost_usd = classification_cost_info["cost_usd"]
    classification_cost_multiple = round(classification_cost_usd / max(jev_cost_usd, 0.00000001), 1) if jev_cost_usd > 0 else None
    classification_savings_pct = round(
        (classification_cost_usd - jev_cost_usd) / max(classification_cost_usd, 0.00000001) * 100,
        1,
    ) if classification_cost_usd > 0 else 0.0
    cost_savings_usd = (llm_cost_usd or 0.0) - jev_cost_usd
    cost_savings_pct = round((cost_savings_usd / max(llm_cost_usd or 0.00001, 0.00001)) * 100.0, 1) if (llm_cost_usd or 0) > 0 else 0.0

    inbound = blob_audit.get("inbound_data", {})
    headline = inbound.get("headline") or blob_audit.get("headline") or "Untitled"

    return {
        "correlation_id": blob_audit.get("correlation_id"),
        "article_id": blob_audit.get("article_id"),
        "headline": headline,
        "config_id": blob_audit.get("config_id"),
        "subject_comparisons": subject_comparisons,
        "tag_comparisons": tag_comparisons,
        "metrics": {
            "validation_accuracy": round((val_correct / val_total) * 100.0, 1) if val_total > 0 else 100.0,
            "prominence_accuracy": round((prom_correct / prom_total) * 100.0, 1) if prom_total > 0 else 100.0,
            "sentiment_accuracy": round((sent_correct / sent_total) * 100.0, 1) if sent_total > 0 else 100.0,
            "tag_accuracy": round((tag_correct / tag_total) * 100.0, 1) if tag_total > 0 else 100.0,
            "val_counts": {"correct": val_correct, "total": val_total},
            "prom_counts": {"correct": prom_correct, "total": prom_total},
            "sent_counts": {"correct": sent_correct, "total": sent_total},
            "tag_counts": {"correct": tag_correct, "total": tag_total},
        },
        "llm_output": _build_llm_output(blob_audit),
        "jev_output": _build_jev_output(jev_result, snapshot, subject_comparisons, tag_comparisons),
        "performance": {
            "llm_provider": blob_audit.get("llm_provider"),
            "llm_model": blob_audit.get("llm_model"),
            "llm_duration_ms": llm_duration_ms,
            "llm_duration_source": "stage_timings",
            "llm_tokens": llm_tokens,
            "llm_cost_usd": llm_cost_usd,
            "llm_cost_source": llm_cost_info["source"],
            "llm_classification_cost_usd": classification_cost_usd,
            "llm_classification_cost_source": classification_cost_info["source"],
            "llm_classification_tokens": {
                "input": classification_cost_info["input_tokens"],
                "output": classification_cost_info["output_tokens"],
                "excluded_input": classification_cost_info["excluded_input_tokens"],
            },
            "jev_provider": jev_result.get("provider"),
            "jev_model": jev_result.get("model"),
            "jev_duration_ms": jev_duration_ms,
            "jev_usage": jev_result.get("usage"),
            "jev_cost_usd": jev_cost_usd,
            "classification_cost_multiple": classification_cost_multiple,
            "classification_cost_savings_pct": classification_savings_pct,
            "speedup_ratio": speedup_ratio,
            "cost_savings_usd": cost_savings_usd,
            "cost_savings_pct": cost_savings_pct
        }
    }


def _build_llm_output(blob_audit: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized view of the baseline LLM decision as archived in the blob."""
    parsed_raw = None
    raw = blob_audit.get("llm_raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            import json as _json

            parsed_raw = _json.loads(raw)
        except Exception:
            parsed_raw = raw

    return {
        "validation_result": blob_audit.get("validation_result"),
        "rejection_reason": blob_audit.get("rejection_reason"),
        "summary": blob_audit.get("summary"),
        "subjects": [
            {
                "subject_id": str(s.get("subject_id")),
                "name": s.get("name"),
                "is_valid": s.get("is_valid"),
                "prominence": s.get("prominence"),
                "sentiment": s.get("sentiment"),
                "tags": s.get("tags", []),
            }
            for s in blob_audit.get("subject_results") or []
        ],
        "tag_groups": (blob_audit.get("scored_attributes") or {}).get("tag_groups") or [],
        "parsed_raw_response": parsed_raw,
    }


def _build_jev_output(
    jev_result: Dict[str, Any],
    snapshot: Dict[str, Any],
    subject_comparisons: List[Dict[str, Any]],
    tag_comparisons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalized view of the TypeSafe Jev decision for the same four elements."""
    answers = jev_result.get("answers", {})
    tags_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for t in tag_comparisons:
        tags_by_subject.setdefault(t["subject_id"], []).append(
            {
                "tag_id": t["tag_id"],
                "tag_name": t["tag_name"],
                "result": t["jev_result"],
                "probability": t["jev_prob"],
            }
        )

    subjects = []
    for c in subject_comparisons:
        s_id = c["subject_id"]
        subjects.append(
            {
                "subject_id": s_id,
                "name": c["name"],
                "is_valid": c["validation"]["jev"],
                "validation_probability": c["validation"]["jev_prob"],
                "prominence": c["prominence"]["jev"],
                "prominence_confidence": c["prominence"]["jev_conf"],
                "sentiment": c["sentiment"]["jev"],
                "sentiment_confidence": c["sentiment"]["jev_conf"],
                "tags": tags_by_subject.get(s_id, []),
            }
        )

    return {
        "subjects": subjects,
        "answers": answers,
    }
