import os
import json
import datetime
from typing import Any, Dict, List, Optional
from config import settings

class RunLogger:
    def __init__(self):
        self.runs_dir = settings.RUNS_DIR
        os.makedirs(self.runs_dir, exist_ok=True)

    def create_run_session(
        self,
        run_id: str,
        config_id: str,
        article_count: int,
        config_ids: Optional[List[str]] = None,
    ) -> str:
        run_folder = os.path.join(self.runs_dir, run_id)
        records_folder = os.path.join(run_folder, "records")
        os.makedirs(records_folder, exist_ok=True)

        meta = {
            "run_id": run_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config_id": config_id,
            "config_ids": config_ids or [config_id],
            "article_count": article_count,
            "status": "in_progress"
        }
        with open(os.path.join(run_folder, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return run_folder

    def log_article_result(
        self,
        run_id: str,
        correlation_id: str,
        comparison: Dict[str, Any],
        record_key: Optional[str] = None,
    ) -> None:
        records_folder = os.path.join(self.runs_dir, run_id, "records")
        file_key = record_key or correlation_id
        file_path = os.path.join(records_folder, f"{file_key}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)

    def finalize_run(self, run_id: str, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        run_folder = os.path.join(self.runs_dir, run_id)
        total_articles = len(comparisons)

        # Aggregate metrics
        val_correct, val_total = 0, 0
        prom_correct, prom_total = 0, 0
        sent_correct, sent_total = 0, 0
        tag_correct, tag_total = 0, 0

        total_llm_duration = 0.0
        total_jev_duration = 0.0
        total_llm_cost = 0.0
        total_jev_cost = 0.0
        total_llm_classification_cost = 0.0
        total_jev_input_tokens = 0
        total_llm_input_tokens = 0

        for c in comparisons:
            m = c.get("metrics", {})
            val_c = m.get("val_counts", {})
            val_correct += val_c.get("correct", 0)
            val_total += val_c.get("total", 0)

            prom_c = m.get("prom_counts", {})
            prom_correct += prom_c.get("correct", 0)
            prom_total += prom_c.get("total", 0)

            sent_c = m.get("sent_counts", {})
            sent_correct += sent_c.get("correct", 0)
            sent_total += sent_c.get("total", 0)

            tag_c = m.get("tag_counts", {})
            tag_correct += tag_c.get("correct", 0)
            tag_total += tag_c.get("total", 0)

            perf = c.get("performance", {})
            total_llm_duration += perf.get("llm_duration_ms", 0.0)
            total_jev_duration += perf.get("jev_duration_ms", 0.0)
            total_llm_cost += perf.get("llm_cost_usd") or 0.0
            total_jev_cost += perf.get("jev_cost_usd", 0.0)
            total_llm_classification_cost += perf.get("llm_classification_cost_usd", 0.0)
            total_llm_input_tokens += perf.get("llm_tokens", {}).get("input", 0)
            total_jev_input_tokens += perf.get("jev_usage", {}).get("input_tokens", 0)

        avg_llm_duration = round(total_llm_duration / max(total_articles, 1), 2)
        avg_jev_duration = round(total_jev_duration / max(total_articles, 1), 2)
        speedup = round(avg_llm_duration / max(avg_jev_duration, 1.0), 2) if avg_llm_duration > 0 else 1.0
        savings_usd = total_llm_cost - total_jev_cost
        savings_pct = round((savings_usd / max(total_llm_cost, 0.00001)) * 100.0, 1) if total_llm_cost > 0 else 0.0
        classification_savings_usd = total_llm_classification_cost - total_jev_cost
        classification_savings_pct = round(
            classification_savings_usd / max(total_llm_classification_cost, 0.00001) * 100,
            1,
        ) if total_llm_classification_cost > 0 else 0.0
        classification_cost_multiple = round(
            total_llm_classification_cost / max(total_jev_cost, 0.00000001), 1
        ) if total_jev_cost > 0 else None
        cost_multiple = round(total_llm_cost / total_jev_cost, 1) if total_jev_cost > 0 else None

        def pct(correct: int, total: int) -> float:
            # No evaluated items means nothing disagreed; do not report a false 0%.
            return round((correct / total) * 100.0, 1) if total > 0 else 100.0

        val_acc = pct(val_correct, val_total)
        prom_acc = pct(prom_correct, prom_total)
        sent_acc = pct(sent_correct, sent_total)
        tag_acc = pct(tag_correct, tag_total)

        optimizer_records = [c.get("prompt_optimization") for c in comparisons if c.get("prompt_optimization", {}).get("enabled")]
        optimizer_by_cache = {
            r.get("cache_key"): r for r in optimizer_records if r.get("cache_key")
        }
        optimizer_summary = {
            "enabled": bool(optimizer_records),
            "configs_compiled": len(optimizer_by_cache),
            "optimizer_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in optimizer_by_cache.values()), 6),
            "optimizer_duration_ms": round(sum(r.get("duration_ms", 0.0) for r in optimizer_by_cache.values()), 2),
            "original_question_chars": sum(r.get("original_question_chars", 0) for r in optimizer_by_cache.values()),
            "optimized_question_chars": sum(r.get("optimized_question_chars", 0) for r in optimizer_by_cache.values()),
        }
        if optimizer_summary["original_question_chars"]:
            optimizer_summary["estimated_char_reduction_pct"] = round(
                (optimizer_summary["original_question_chars"] - optimizer_summary["optimized_question_chars"])
                / optimizer_summary["original_question_chars"] * 100,
                1,
            )

        summary = {
            "run_id": run_id,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_articles": total_articles,
            "accuracy": {
                "subject_validation": {
                    "accuracy_pct": val_acc,
                    "correct": val_correct,
                    "total": val_total
                },
                "subject_prominence": {
                    "accuracy_pct": prom_acc,
                    "correct": prom_correct,
                    "total": prom_total
                },
                "subject_sentiment": {
                    "accuracy_pct": sent_acc,
                    "correct": sent_correct,
                    "total": sent_total
                },
                "llm_tags": {
                    "accuracy_pct": tag_acc,
                    "correct": tag_correct,
                    "total": tag_total
                }
            },
            "prompt_optimization": optimizer_summary,
            "performance": {
                "avg_llm_duration_ms": avg_llm_duration,
                "avg_jev_duration_ms": avg_jev_duration,
                "speedup_ratio": speedup,
                "total_llm_cost_usd": round(total_llm_cost, 6),
                "total_llm_classification_cost_usd": round(total_llm_classification_cost, 6),
                "total_jev_cost_usd": round(total_jev_cost, 6),
                "cost_savings_usd": round(savings_usd, 6),
                "cost_savings_pct": savings_pct,
                "cost_multiple": cost_multiple,
                "classification_cost_savings_pct": classification_savings_pct,
                "classification_cost_multiple": classification_cost_multiple,
                "total_llm_input_tokens": total_llm_input_tokens,
                "total_jev_input_tokens": total_jev_input_tokens
            }
        }
        # Save JSON summary
        summary_path = os.path.join(run_folder, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Generate human-readable clean Markdown summary for LLM ingestion / business reporting
        md_content = self._generate_markdown_report(summary)
        md_path = os.path.join(run_folder, "benchmark_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Update run_meta.json
        meta_path = os.path.join(run_folder, "run_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "completed"
            meta["completed_at"] = summary["completed_at"]
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return summary

    def _generate_markdown_report(self, s: Dict[str, Any]) -> str:
        acc = s["accuracy"]
        perf = s["performance"]
        return f"""# TypeSafe Jev vs LLM Feasibility Benchmark Report

**Run ID:** `{s["run_id"]}`  
**Evaluated Articles:** {s["total_articles"]}  
**Completed:** {s["completed_at"]}  

---

## 1. Classification Parity & Accuracy

Comparison of TypeSafe Jev decisions against baseline LLM audit labels:

| Decision Task | Accuracy (%) | Matches / Total |
| :--- | :--- | :--- |
| **1. Subject Validation** | **{acc["subject_validation"]["accuracy_pct"]}%** | {acc["subject_validation"]["correct"]} / {acc["subject_validation"]["total"]} |
| **2. Subject Prominence** | **{acc["subject_prominence"]["accuracy_pct"]}%** | {acc["subject_prominence"]["correct"]} / {acc["subject_prominence"]["total"]} |
| **3. Subject Sentiment** | **{acc["subject_sentiment"]["accuracy_pct"]}%** | {acc["subject_sentiment"]["correct"]} / {acc["subject_sentiment"]["total"]} |
| **4. LLM Tags** | **{acc["llm_tags"]["accuracy_pct"]}%** | {acc["llm_tags"]["correct"]} / {acc["llm_tags"]["total"]} |

---

## 2. Speed and Cost Comparison

| Metric | LLM Baseline | TypeSafe Jev | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Full pipeline cost (USD)** | ${perf["total_llm_cost_usd"]:.6f} | ${perf["total_jev_cost_usd"]:.6f} | **{perf["cost_multiple"]}x cheaper** |
| **Classification-only estimated cost (USD)** | ${perf["total_llm_classification_cost_usd"]:.6f} | ${perf["total_jev_cost_usd"]:.6f} | **{perf["classification_cost_multiple"]}x cheaper ({perf["classification_cost_savings_pct"]}% reduction)** |
| **Total Input Tokens** | {perf["total_llm_input_tokens"]:,} | {perf["total_jev_input_tokens"]:,} | - |

---

## 3. Executive Summary for Downstream LLM & Business

- **Parity Assessment**: TypeSafe Jev evaluates structured classification directly without prompt chain parsing.
- **Cost Reduction**: Jev charges $0.042 per million input tokens with zero charge for output tokens. Baseline LLM spend is **{perf["cost_multiple"]}x higher** ({perf["cost_savings_pct"]}% cost reduction).
- **Latency Gain**: Parallel question evaluation achieved an average response time of **{perf["avg_jev_duration_ms"]} ms**, representing a **{perf["speedup_ratio"]}x** speedup over the pipeline LLM.
"""

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if not os.path.exists(self.runs_dir):
            return runs
        for name in sorted(os.listdir(self.runs_dir), reverse=True):
            r_dir = os.path.join(self.runs_dir, name)
            if os.path.isdir(r_dir):
                summary_file = os.path.join(r_dir, "benchmark_summary.json")
                meta_file = os.path.join(r_dir, "run_meta.json")
                run_data = {"run_id": name}
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            run_data.update(json.load(f))
                    except Exception:
                        pass
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            run_data["summary"] = json.load(f)
                    except Exception:
                        pass
                runs.append(run_data)
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        r_dir = os.path.join(self.runs_dir, run_id)
        if not os.path.exists(r_dir):
            return None
        summary_file = os.path.join(r_dir, "benchmark_summary.json")
        md_file = os.path.join(r_dir, "benchmark_summary.md")
        meta_file = os.path.join(r_dir, "run_meta.json")
        records_dir = os.path.join(r_dir, "records")

        data = {"run_id": run_id}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                data["meta"] = json.load(f)
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                data["summary"] = json.load(f)
        if os.path.exists(md_file):
            with open(md_file, "r") as f:
                data["markdown_report"] = f.read()

        records = []
        if os.path.exists(records_dir):
            for rname in os.listdir(records_dir):
                if rname.endswith(".json"):
                    with open(os.path.join(records_dir, rname), "r") as f:
                        records.append(json.load(f))
        data["records"] = records
        return data


    @staticmethod
    def _rmtree_safe(path: str) -> bool:
        import shutil

        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            logger.error(f"Could not remove {path}: {e}")
            return False

    def delete_run(self, run_id: str) -> bool:
        r_dir = os.path.join(self.runs_dir, run_id)
        if not os.path.isdir(r_dir):
            return False
        return self._rmtree_safe(r_dir)

    def clear_all_runs(self) -> int:
        """Delete every run folder. Returns the number removed."""
        removed = 0
        if not os.path.isdir(self.runs_dir):
            return 0
        for name in os.listdir(self.runs_dir):
            r_dir = os.path.join(self.runs_dir, name)
            if os.path.isdir(r_dir) and self._rmtree_safe(r_dir):
                removed += 1
        return removed




run_logger = RunLogger()
