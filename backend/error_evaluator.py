import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import httpx
from config import settings

logger = logging.getLogger("error_evaluator")

class ErrorEvaluatorError(Exception):
    pass

class ErrorEvaluator:
    """Evaluates benchmark classification discrepancies between baseline LLM and TypeSafe Jev
    using Gemini to determine which model made the correct judgment and provide a comprehensive
    analysis summary.
    """

    def __init__(self):
        self.model = settings.GEMINI_OPTIMIZER_MODEL  # e.g., gemini-3.8-flash

    async def _call_gemini(self, prompt: str, response_mime_type: str = "application/json") -> str:
        if not settings.GEMINI_API_KEY:
            raise ErrorEvaluatorError("GEMINI_API_KEY is not configured in the environment")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": response_mime_type,
            },
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                params={"key": settings.GEMINI_API_KEY},
                json=payload,
            )
            if response.is_error:
                raise ErrorEvaluatorError(f"Gemini API returned {response.status_code}: {response.text[:500]}")
            data = response.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ErrorEvaluatorError("Gemini returned an empty or invalid response structure") from exc

    def find_discrepancies(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify articles with disagreements, failed evaluations, or classification differences."""
        issues = []
        for r in records:
            if r.get("failed"):
                issues.append({
                    "type": "execution_failure",
                    "correlation_id": r.get("correlation_id"),
                    "headline": r.get("headline", "Untitled"),
                    "config_id": r.get("config_id"),
                    "error": r.get("error"),
                    "record": r,
                })
                continue

            m = r.get("metrics", {})
            has_mismatch = any(
                m.get(k, 100) < 100.0
                for k in ["validation_accuracy", "prominence_accuracy", "sentiment_accuracy", "tag_accuracy"]
            )
            if has_mismatch:
                subject_mismatches = []
                for s in r.get("subject_comparisons", []):
                    s_name = s.get("name")
                    for cat in ["validation", "prominence", "sentiment"]:
                        cat_data = s.get(cat, {})
                        if not cat_data.get("match"):
                            subject_mismatches.append({
                                "subject": s_name,
                                "dimension": cat,
                                "llm": cat_data.get("llm"),
                                "jev": cat_data.get("jev"),
                            })

                tag_mismatches = []
                for t in r.get("tag_comparisons", []):
                    if not t.get("match"):
                        tag_mismatches.append({
                            "tag": t.get("name"),
                            "llm": t.get("llm"),
                            "jev": t.get("jev"),
                        })

                issues.append({
                    "type": "disagreement",
                    "correlation_id": r.get("correlation_id"),
                    "headline": r.get("headline", "Untitled"),
                    "config_id": r.get("config_id"),
                    "subject_mismatches": subject_mismatches,
                    "tag_mismatches": tag_mismatches,
                    "record": r,
                })
        return issues

    async def evaluate_article_discrepancy(
        self,
        issue: Dict[str, Any],
        article_state_preview: str,
    ) -> Dict[str, Any]:
        """Ask Gemini to arbitrate which model is correct for a specific article disagreement."""
        headline = issue.get("headline")
        sub_mismatches = issue.get("subject_mismatches", [])
        tag_mismatches = issue.get("tag_mismatches", [])

        prompt = f"""You are an expert editorial auditor comparing two automated classification systems:
1. Baseline LLM
2. TypeSafe Jev (a calibrated classification model)

ARTICLE HEADLINE: {headline}
ARTICLE TEXT (or excerpt):
\"\"\"
{article_state_preview[:15000]}
\"\"\"

The two systems disagreed on the following classifications:
Subject Disagreements:
{json.dumps(sub_mismatches, indent=2)}

Tag Disagreements:
{json.dumps(tag_mismatches, indent=2)}

TASK:
For each disagreement:
1. Determine who is correct: "llm", "jev", or "neither" / "both_defensible".
2. Explain the editorial and contextual reason why that model is right based strictly on the article content.
3. Provide an overall ruling for this article: which model exhibited better judgment.

Return ONLY a valid JSON object matching this schema:
{{
  "verdicts": [
    {{
      "item": "<subject name or tag name>",
      "dimension": "<validation, prominence, sentiment, or tag>",
      "llm_value": "<what LLM chose>",
      "jev_value": "<what Jev chose>",
      "correct_model": "llm" | "jev" | "both_defensible" | "neither",
      "reasoning": "<concise evidence-based explanation referencing article text>"
    }}
  ],
  "overall_preferred_model": "llm" | "jev" | "tie",
  "summary": "<1-2 sentence overall verdict for this article>"
}}
"""
        raw_json = await self._call_gemini(prompt)
        try:
            return json.loads(raw_json)
        except Exception as e:
            logger.warning(f"Failed to parse Gemini arbitration response: {e}")
            return {
                "verdicts": [],
                "overall_preferred_model": "tie",
                "summary": f"Arbitration completed but JSON parse failed: {raw_json[:200]}",
            }

    async def run_error_analysis(
        self,
        run_id: str,
        records: List[Dict[str, Any]],
        get_article_text_fn=None,
    ) -> Dict[str, Any]:
        """Perform a complete error evaluation run across all issues in a benchmark run."""
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        issues = self.find_discrepancies(records)

        if not issues:
            return {
                "run_id": run_id,
                "model": self.model,
                "analyzed_at": started_at,
                "total_issues": 0,
                "disagreements_count": 0,
                "execution_failures_count": 0,
                "llm_preferred_count": 0,
                "jev_preferred_count": 0,
                "ties_count": 0,
                "article_evaluations": [],
                "executive_summary": "No disagreements or errors found in this benchmark run. All evaluated items achieved 100% agreement.",
            }

        disagreements = [i for i in issues if i["type"] == "disagreement"]
        failures = [i for i in issues if i["type"] == "execution_failure"]

        # Evaluate up to 15 representative disagreements in parallel with a concurrency semaphore
        eval_sample = disagreements[:15]
        sem = asyncio.Semaphore(5)

        async def arbitrate_single(issue):
            corr_id = issue.get("correlation_id")
            article_text = ""
            if get_article_text_fn:
                try:
                    article_text = get_article_text_fn(issue.get("record", {}))
                except Exception as ex:
                    logger.warning(f"Could not load article text for {corr_id}: {ex}")

            if not article_text:
                article_text = (
                    issue.get("record", {}).get("jev_request", {}).get("state")
                    or issue.get("headline", "")
                )

            async with sem:
                try:
                    eval_res = await self.evaluate_article_discrepancy(issue, article_text)
                    return {
                        "correlation_id": corr_id,
                        "headline": issue.get("headline"),
                        "config_id": issue.get("config_id"),
                        "subject_mismatches": issue.get("subject_mismatches"),
                        "tag_mismatches": issue.get("tag_mismatches"),
                        "arbitration": eval_res,
                    }
                except Exception as e:
                    logger.exception(f"Error evaluating discrepancy for {corr_id}: {e}")
                    return {
                        "correlation_id": corr_id,
                        "headline": issue.get("headline"),
                        "error": str(e),
                    }

        eval_results = await asyncio.gather(*[arbitrate_single(issue) for issue in eval_sample])
        article_evaluations = list(eval_results)

        llm_preferred = sum(1 for a in article_evaluations if a.get("arbitration", {}).get("overall_preferred_model") == "llm")
        jev_preferred = sum(1 for a in article_evaluations if a.get("arbitration", {}).get("overall_preferred_model") == "jev")
        ties = sum(1 for a in article_evaluations if a.get("arbitration") and a.get("arbitration", {}).get("overall_preferred_model") not in ("llm", "jev"))

        # Synthesize executive summary report across evaluations
        exec_prompt = f"""You are an executive auditor reviewing a model comparison report.
Based on the following disagreement arbitration findings between the baseline LLM and TypeSafe Jev (evaluated using {self.model}):

Total discrepancies found: {len(disagreements)}
Sample analyzed in detail: {len(article_evaluations)}
TypeSafe Jev preferred in: {jev_preferred} articles
Baseline LLM preferred in: {llm_preferred} articles
Ties / both defensible in: {ties} articles
Execution/gateway failures: {len(failures)}

SAMPLE ARTICLE ARBITRATIONS:
{json.dumps([{ 'headline': a.get('headline'), 'arbitration': a.get('arbitration') } for a in article_evaluations[:8]], indent=2)}

Generate a clean, high-impact markdown summary:
1. Executive Verdict: Which model was more accurate overall and where each model excels.
2. Root Causes of Discrepancies: Key patterns (e.g. sentiment calibration, threshold sensitivity on passing mentions, hallucination vs precision).
3. Recommendation: Actionable guidance on whether Jev is production-ready or which thresholds/prompts should be tuned.
Do not use marketing fluff. Stick strictly to concrete observations.
"""
        try:
            exec_summary_raw = await self._call_gemini(exec_prompt, response_mime_type="text/plain")
            exec_summary = self._clean_markdown_summary(exec_summary_raw)
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            exec_summary = f"### Executive Summary\n- **Jev Preferred:** {jev_preferred}\n- **LLM Preferred:** {llm_preferred}\n- **Ties / Defensible:** {ties}\n\n*Detailed article evaluations are available in the table below.*"

        return {
            "run_id": run_id,
            "model": self.model,
            "analyzed_at": started_at,
            "total_issues": len(issues),
            "disagreements_count": len(disagreements),
            "execution_failures_count": len(failures),
            "sample_analyzed_count": len(article_evaluations),
            "llm_preferred_count": llm_preferred,
            "jev_preferred_count": jev_preferred,
            "ties_count": ties,
            "article_evaluations": article_evaluations,
            "failures": [{ "correlation_id": f.get("correlation_id"), "error": f.get("error") } for f in failures],
            "executive_summary": exec_summary,
        }

    @staticmethod
    def _clean_markdown_summary(text: Any) -> str:
        """Transforms raw response (including accidental JSON) into clean, human-readable markdown."""
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except Exception:
                # Strips wrapping markdown codeblocks if Gemini enclosed plain text in ```markdown ... ```
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    return "\n".join(lines).strip()
                return cleaned
        else:
            data = text

        if not isinstance(data, dict):
            return str(data)

        lines = []
        if "executive_verdict" in data:
            ev = data["executive_verdict"]
            lines.append("### Executive Verdict")
            if isinstance(ev, dict):
                if "preferred_model" in ev:
                    lines.append(f"- **Preferred Model:** {ev['preferred_model']}")
                if "win_rate_decisive" in ev:
                    lines.append(f"- **Decisive Win Rate:** {ev['win_rate_decisive']}")
                if "summary" in ev:
                    lines.append(f"\n{ev['summary']}\n")
            else:
                lines.append(f"{ev}\n")

        if "root_causes_of_discrepancies" in data:
            lines.append("### Root Causes of Discrepancies")
            for item in data["root_causes_of_discrepancies"]:
                if isinstance(item, dict):
                    pat = item.get("pattern", "Pattern")
                    desc = item.get("detail") or item.get("description", "")
                    lines.append(f"- **{pat}:** {desc}")
                else:
                    lines.append(f"- {item}")
            lines.append("")

        if "recommendations" in data:
            lines.append("### Recommendations")
            for rec in data["recommendations"]:
                if isinstance(rec, dict):
                    act = rec.get("action", "Action")
                    det = rec.get("details", "")
                    lines.append(f"- **{act}:** {det}")
                else:
                    lines.append(f"- {rec}")

        return "\n".join(lines) if lines else json.dumps(data, indent=2)

error_evaluator = ErrorEvaluator()
