from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx

from config import settings

OPTIMIZER_VERSION = "1"


class PromptOptimizationError(RuntimeError):
    pass


class PromptOptimizer:
    def __init__(self) -> None:
        self.cache_dir = os.path.join(settings.CACHE_DIR, "optimized_prompts")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, snapshot: dict[str, Any], metadata: dict[str, Any]) -> str:
        source = {
            "optimizer_version": OPTIMIZER_VERSION,
            "model": settings.GEMINI_OPTIMIZER_MODEL,
            "config_id": metadata.get("config_id"),
            "version_number": metadata.get("version_number"),
            "subjects": snapshot.get("subjects", []),
        }
        encoded = json.dumps(source, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def optimize_snapshot(
        self,
        snapshot: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not settings.GEMINI_API_KEY:
            raise PromptOptimizationError(
                "Prompt optimization is enabled but GEMINI_API_KEY is not configured"
            )

        cache_key = self._cache_key(snapshot, metadata)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            self._validate_rubric(cached["rubric"], snapshot)
            return cached["rubric"], {
                **cached.get("metadata", {}),
                "cached": True,
                "cache_key": cache_key,
                "cost_usd": 0.0,
            }

        source = self._source_config(snapshot)
        prompt = self._build_optimizer_prompt(source)
        started = time.perf_counter()
        raw = await self._call_gemini(prompt)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        rubric = self._parse_json_response(raw["text"])
        self._validate_rubric(rubric, snapshot)

        input_tokens = int(raw.get("input_tokens", 0) or 0)
        output_tokens = int(raw.get("output_tokens", 0) or 0)
        cost_usd = (
            input_tokens / 1_000_000 * settings.LLM_INPUT_COST_PER_MTOK
            + output_tokens / 1_000_000 * settings.LLM_OUTPUT_COST_PER_MTOK
        )
        optimizer_metadata = {
            "enabled": True,
            "model": settings.GEMINI_OPTIMIZER_MODEL,
            "cached": False,
            "cache_key": cache_key,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"rubric": rubric, "metadata": optimizer_metadata}, f, indent=2)
        return rubric, optimizer_metadata

    @staticmethod
    def _source_config(snapshot: dict[str, Any]) -> dict[str, Any]:
        subjects = []
        for subject in snapshot.get("subjects", []):
            subjects.append(
                {
                    "subject_id": str(subject["id"]),
                    "name": subject.get("name", ""),
                    "entity_definition": subject.get("entity_definition", ""),
                    "validation_prompt": subject.get("validation_prompt", ""),
                    "prominence_prompt": subject.get("prominence_prompt", ""),
                    "sentiment_prompt": subject.get("sentiment_prompt", ""),
                    "tags": [
                        {
                            "tag_id": str(tag["tag_id"]),
                            "tag_name": tag.get("tag_name", ""),
                            "evaluation_type": tag.get("evaluation_type"),
                            "prompt_text": tag.get("prompt_text", ""),
                        }
                        for tag in subject.get("tag_evaluations", [])
                        if tag.get("evaluation_type") == "llm" and tag.get("prompt_text")
                    ],
                }
            )
        return {"subjects": subjects}

    @staticmethod
    def _build_optimizer_prompt(source: dict[str, Any]) -> str:
        source_json = json.dumps(source, ensure_ascii=False, indent=2)
        return f"""You are a conservative configuration compiler. Compact the supplied Curation Engine classification rules for a typed decision model.

Preserve every semantic include rule, exclusion rule, exception, tie-break rule, scope restriction, named product, and tag definition. Do not invent rules. Do not remove a rule merely because it is long. Remove only output-format instructions such as 'return JSON', 'output one word', or 'no explanation', because the receiving API supplies the output type.

Return JSON only with this exact shape:
{{
  "subjects": [
    {{
      "subject_id": "string",
      "validation_criteria": "string",
      "prominence_criteria": "string",
      "sentiment_criteria": "string",
      "tags": [{{"tag_id": "string", "criteria": "string"}}]
    }}
  ]
}}

Canonical labels are fixed and must not be changed:
- prominence: primary, significant, passing. Map client wording such as Prime to primary.
- sentiment: positive, negative, neutral, balanced.
- validation and tags: true or false semantics.

The output is a rubric for another program. Do not return summaries, explanations, or decisions about any article.

SOURCE CONFIGURATION:
{source_json}
"""

    async def _call_gemini(self, prompt: str) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.GEMINI_OPTIMIZER_MODEL}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                params={"key": settings.GEMINI_API_KEY},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptOptimizationError("Gemini returned no optimizer content") from exc
        usage = data.get("usageMetadata") or {}
        return {
            "text": text,
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
        }

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PromptOptimizationError("Gemini optimizer returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise PromptOptimizationError("Gemini optimizer response must be an object")
        return value

    @staticmethod
    def _validate_rubric(rubric: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not isinstance(rubric.get("subjects"), list):
            raise PromptOptimizationError("Optimized rubric has no subjects list")
        expected = {str(s["id"]): s for s in snapshot.get("subjects", [])}
        actual = {str(s.get("subject_id")): s for s in rubric["subjects"]}
        if set(actual) != set(expected):
            raise PromptOptimizationError("Optimized rubric changed the configured subject set")
        for subject_id, source in expected.items():
            optimized = actual[subject_id]
            for field in ("validation_criteria", "prominence_criteria", "sentiment_criteria"):
                if not isinstance(optimized.get(field), str) or not optimized[field].strip():
                    raise PromptOptimizationError(f"Optimized rubric missing {field} for {subject_id}")
            source_tags = {
                str(t["tag_id"])
                for t in source.get("tag_evaluations", [])
                if t.get("evaluation_type") == "llm" and t.get("prompt_text")
            }
            actual_tags = {
                str(t.get("tag_id")): t
                for t in optimized.get("tags", [])
            }
            if set(actual_tags) != source_tags:
                raise PromptOptimizationError(f"Optimized rubric changed LLM tags for {subject_id}")
            if any(not isinstance(t.get("criteria"), str) or not t["criteria"].strip() for t in actual_tags.values()):
                raise PromptOptimizationError(f"Optimized rubric contains an empty tag criterion for {subject_id}")
prompt_optimizer = PromptOptimizer()


def compile_system_one_rubric(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Deterministically compiles verbose Curation Engine prompts into high-density,
    bounded criteria rubrics optimized for ModernBERT System 1 decision models.
    Extracts product/leadership aliases and eliminates output-format boilerplate so
    no criteria exceed the encoder head token budget.
    """
    import re

    compiled_subjects = []
    for s in snapshot.get("subjects", []):
        s_id = str(s["id"])
        name = s.get("name", "")
        raw_text = " ".join([
            s.get("entity_definition", "") or "",
            s.get("validation_prompt", "") or "",
            s.get("prominence_prompt", "") or "",
        ])

        # Extract named products and aliases from parentheses
        aliases = []
        for m in re.finditer(r"\(([^)]{3,100})\)", raw_text):
            cand = m.group(1)
            if any(w in cand.lower() for w in ["product", "platform", "including", "service", "leadership", "e.g.", "brands"]) or "," in cand:
                parts = [p.strip() for p in cand.replace("e.g.", "").split(",") if len(p.strip()) > 2]
                aliases.extend(parts)

        # Unique aliases
        seen = set()
        unique_aliases = []
        for a in aliases:
            a_clean = a.strip()
            if a_clean.lower() not in seen and len(a_clean) < 40:
                seen.add(a_clean.lower())
                unique_aliases.append(a_clean)

        alias_summary = ", ".join(unique_aliases[:8])
        val_rule = (
            f"Relevant to {name} operations"
            + (f" or products ({alias_summary})" if alias_summary else "")
            + ". Excludes incidental partner lists, sponsored ads, or unrelated homonyms."
        )
        prom_rule = (
            "Primary: Dominant story focus in headline/lead. "
            "Significant: Key topic discussed substantively across multiple paragraphs. "
            "Passing: Brief citation, quote, or list entry."
        )
        sent_rule = (
            f"Tone toward {name}. Default is factual neutral. "
            f"Positive: business growth, award, innovation, revenue beat. "
            f"Negative: regulatory scrutiny, litigation, scandal, financial loss, breach."
        )

        tags = []
        for t in s.get("tag_evaluations", []):
            if t.get("evaluation_type") == "llm" and t.get("prompt_text"):
                clean_t = re.sub(r"^(Respond|Return|Classify|Determine).*?\.\s*", "", t["prompt_text"]).strip()
                tags.append({"tag_id": str(t["tag_id"]), "criteria": clean_t[:400]})

        compiled_subjects.append({
            "subject_id": s_id,
            "name": name,
            "aliases": unique_aliases[:10],
            "validation_criteria": val_rule,
            "prominence_criteria": prom_rule,
            "sentiment_criteria": sent_rule,
            "tags": tags,
        })
    return {"subjects": compiled_subjects}
