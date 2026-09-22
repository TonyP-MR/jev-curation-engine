from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from config import settings


OPTIMIZER_VERSION = "3"
_OUTPUT_FORMAT_PATTERNS = (
    re.compile(r"\breturn\s+exactly\b", re.IGNORECASE),
    re.compile(r"\brespond\s+with\b", re.IGNORECASE),
    re.compile(r"\boutput\s+(?:only|exactly)\b", re.IGNORECASE),
    re.compile(r"\b(?:true|false|primary|significant|passing|positive|negative|neutral|balanced)\s+only\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:other\s+text|explanation)\b", re.IGNORECASE),
    re.compile(r"\bapi\s+choice\b", re.IGNORECASE),
    re.compile(r"\bjson\b", re.IGNORECASE),
)
_POLICY_MARKERS = {
    "include": (
        re.compile(r"\binclude(?:s|d|ing)?\s+(?:conditions?|rules?|if|when|only|\(|:)", re.IGNORECASE),
        re.compile(r"\b(?:includ(?:e|es|ed|ing)|qualif(?:y|ies)|true\s+when|relevant\s+when)\b", re.IGNORECASE),
    ),
    "exclude": (
        re.compile(r"\bexclud(?:e|es|ed|ing)\b", re.IGNORECASE),
        re.compile(r"\b(?:exclud(?:e|es|ed|ing)|reject(?:ed)?|false\s+when|not\s+qualif(?:y|ies)|irrelevant)\b", re.IGNORECASE),
    ),
    "exception": (
        re.compile(r"\bexcept(?:ion|ions)?\b", re.IGNORECASE),
        re.compile(r"\b(?:except(?:ion|ions)?|override(?:s|d)?|despite|however)\b", re.IGNORECASE),
    ),
    "only": (
        re.compile(r"\bonly\b", re.IGNORECASE),
        re.compile(r"\bonly\b", re.IGNORECASE),
    ),
    "unless": (
        re.compile(r"\bunless\b", re.IGNORECASE),
        re.compile(r"\b(?:unless|except\s+when)\b", re.IGNORECASE),
    ),
    "regardless": (
        re.compile(r"\bregardless\b", re.IGNORECASE),
        re.compile(r"\b(?:regardless|even\s+if|despite|before|precedence|override(?:s|d)?|supersed(?:e|es|ed|ing)|priorit(?:y|ize|izes|ized))\b", re.IGNORECASE),
    ),
    "before": (
        re.compile(r"\bbefore\b", re.IGNORECASE),
        re.compile(r"\b(?:before|prior|first|precedence)\b", re.IGNORECASE),
    ),
    "after": (
        re.compile(r"\bafter\b", re.IGNORECASE),
        re.compile(r"\b(?:after|then|following|before|prior|first|precedence|override(?:s|d)?|supersed(?:e|es|ed|ing)|priority)\b", re.IGNORECASE),
    ),
}
_QUOTED_LITERAL_RE = re.compile(
    r'"([^"\n]+)"|\'([^\'\n]+)\'|“([^”\n]+)”|‘([^’\n]+)’'
)
_NUMBER_LITERAL_RE = re.compile(
    r"(?<![\w])(?:[$€£])?\d+(?:[,.]\d+)*(?:%|[–—-]\d+(?:[,.]\d+)*)?(?![\w])"
)
_CONFIGURATOR_RE = re.compile(
    r"\b(?:configurator|administrator|config owner)\b|"
    r"\b(?:please\s+)?(?:configure|specify|define|enter|supply)\s+(?:the\s+)?(?:criteria|rule|rules|prompt|tag|subject)\b",
    re.IGNORECASE,
)
_POSITIVE_RULE_RE = re.compile(r"\b(?:include|return\s+true|qualif(?:y|ies))\b", re.IGNORECASE)
_NEGATIVE_RULE_RE = re.compile(r"\b(?:exclude|return\s+false|does\s+not\s+qualify|do\s+not\s+qualify)\b", re.IGNORECASE)
_EXPLICIT_TAG_NAME_RE = re.compile(
    r"\b(?:tag|category|label)\s+(?:named\s+|called\s+|for\s+)?[\"'“‘]([^\"'”’]+)[\"'”’]",
    re.IGNORECASE,
)

QuestionMap = dict[str, dict[str, Any]]


class PromptOptimizationError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(_all_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(_all_text(item) for item in value)
    return ""


def _quoted_literals(text: str) -> set[str]:
    literals: set[str] = set()
    for match in _QUOTED_LITERAL_RE.finditer(text):
        value = (
            next((group for group in match.groups() if group is not None), "")
            .strip()
            .rstrip(",;:")
        )
        if value:
            literals.add(value)
    return literals


def _numeric_literals(text: str) -> set[str]:
    literals: set[str] = set()
    for match in _NUMBER_LITERAL_RE.finditer(text):
        value = match.group(0)
        # Numbered list markers carry layout, not policy. Ranges, percentages, money,
        # and numbers embedded in prose remain protected.
        line_prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
        suffix = text[match.end() : match.end() + 1]
        if value.isdigit() and not line_prefix.strip() and suffix in (".", ")"):
            continue
        literals.add(value)
    return literals


def _alias_values(value: Any, *, key: str = "") -> set[str]:
    aliases: set[str] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            lowered = str(child_key).lower()
            if lowered == "name" or "alias" in lowered:
                aliases.update(_alias_values(child, key=lowered))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            aliases.update(_alias_values(child, key=key))
    elif isinstance(value, str) and value.strip() and (key == "name" or "alias" in key):
        aliases.add(value.strip())
    return aliases


class PromptOptimizer:
    def __init__(self) -> None:
        self.cache_dir = os.path.join(settings.CACHE_DIR, "optimized_prompts")
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _cache_identity(
        questions: QuestionMap,
        snapshot: dict[str, Any],
        metadata: dict[str, Any],
        assembler_version: str,
    ) -> dict[str, Any]:
        return {
            "environment": settings.ENVIRONMENT,
            "numeric_config_id": metadata.get("numeric_config_id"),
            "config_id": metadata.get("config_id"),
            "version_number": metadata.get("version_number"),
            "snapshot_sha256": _sha256_json(snapshot),
            "assembler_version": assembler_version,
            "optimizer_version": OPTIMIZER_VERSION,
            "model": settings.GEMINI_OPTIMIZER_MODEL,
            "assembled_questions": questions,
        }

    @classmethod
    def _cache_key(
        cls,
        questions: QuestionMap,
        snapshot: dict[str, Any],
        metadata: dict[str, Any],
        assembler_version: str,
    ) -> tuple[str, dict[str, Any]]:
        identity = cls._cache_identity(questions, snapshot, metadata, assembler_version)
        return _sha256_json(identity), identity

    async def optimize_questions(
        self,
        questions: QuestionMap,
        snapshot: dict[str, Any],
        metadata: dict[str, Any],
        assembler_version: str,
    ) -> tuple[QuestionMap, dict[str, Any]]:
        source_questions = json.loads(json.dumps(questions, ensure_ascii=False))
        diagnostics = self._detect_anomalies(source_questions, snapshot)
        defective_ids = {item["question_id"] for item in diagnostics}
        safe_questions = {
            question_id: question
            for question_id, question in source_questions.items()
            if question_id not in defective_ids
        }
        if not safe_questions:
            raise PromptOptimizationError(
                "No safe assembled questions remain after source anomaly detection"
            )

        cache_key, identity = self._cache_key(
            source_questions, snapshot, metadata, assembler_version
        )
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        cached = self._read_cache(
            cache_path,
            cache_key,
            identity,
            source_questions,
            snapshot,
            defective_ids,
        )
        if cached is not None:
            optimized, cached_metadata = cached
            return optimized, {
                **cached_metadata,
                "cached": True,
                "duration_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        if not settings.GEMINI_API_KEY:
            raise PromptOptimizationError(
                "Prompt optimization is enabled but GEMINI_API_KEY is not configured"
            )

        prompt = self._build_optimizer_prompt(safe_questions, snapshot)
        started = time.perf_counter()
        raw = await self._call_gemini(prompt)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        optimized_safe = self._parse_json_response(raw["text"])
        self._validate_question_map(optimized_safe, safe_questions, snapshot)

        optimized_questions: QuestionMap = {}
        for question_id, source_question in source_questions.items():
            optimized_questions[question_id] = (
                source_question if question_id in defective_ids else optimized_safe[question_id]
            )
        self._validate_question_map(
            optimized_questions,
            source_questions,
            snapshot,
            defective_question_ids=defective_ids,
        )

        source_chars = len(_canonical_json(source_questions))
        optimized_chars = len(_canonical_json(optimized_questions))
        if optimized_chars >= source_chars:
            raise PromptOptimizationError(
                "Optimized question map must be smaller than the assembled source "
                f"({optimized_chars} >= {source_chars} characters)"
            )

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
            "original_question_chars": source_chars,
            "optimized_question_chars": optimized_chars,
            "character_reduction": source_chars - optimized_chars,
            "character_reduction_pct": round(
                (source_chars - optimized_chars) / source_chars * 100.0, 1
            ),
            "diagnostics": diagnostics,
        }
        artifact = {
            "cache_key": cache_key,
            "identity": identity,
            "questions": optimized_questions,
            "metadata": optimizer_metadata,
        }
        self._write_cache_atomic(cache_path, artifact)
        return optimized_questions, optimizer_metadata

    def _read_cache(
        self,
        cache_path: str,
        cache_key: str,
        identity: dict[str, Any],
        source_questions: QuestionMap,
        snapshot: dict[str, Any],
        defective_ids: set[str],
    ) -> tuple[QuestionMap, dict[str, Any]] | None:
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            if cached.get("cache_key") != cache_key or cached.get("identity") != identity:
                raise PromptOptimizationError("Cached optimizer identity does not match")
            questions = cached["questions"]
            metadata = cached["metadata"]
            if not isinstance(questions, dict) or not isinstance(metadata, dict):
                raise PromptOptimizationError("Cached optimizer artifact has an invalid shape")
            self._validate_question_map(
                questions,
                source_questions,
                snapshot,
                defective_question_ids=defective_ids,
            )
            source_chars = len(_canonical_json(source_questions))
            optimized_chars = len(_canonical_json(questions))
            if optimized_chars >= source_chars:
                raise PromptOptimizationError("Cached optimized question map is not smaller")
            if metadata.get("original_question_chars") != source_chars:
                raise PromptOptimizationError("Cached source character telemetry is stale")
            if metadata.get("optimized_question_chars") != optimized_chars:
                raise PromptOptimizationError("Cached optimized character telemetry is stale")
            return questions, metadata
        except (OSError, ValueError, KeyError, TypeError, PromptOptimizationError):
            return None

    @staticmethod
    def _write_cache_atomic(path: str, artifact: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(path),
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                json.dump(artifact, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @classmethod
    def _build_optimizer_prompt(
        cls, questions: QuestionMap, snapshot: dict[str, Any]
    ) -> str:
        known_literals = cls._known_literals_by_question(snapshot)
        protected_literals: dict[str, list[str]] = {}
        for question_id, question in questions.items():
            question_text = _all_text(question)
            literals = _quoted_literals(question_text)
            literals.update(_numeric_literals(question_text))
            literals.update(known_literals.get(question_id, set()))
            protected_literals[question_id] = sorted(literals)
        protected_json = json.dumps(protected_literals, ensure_ascii=False, indent=2)
        source_json = json.dumps(questions, ensure_ascii=False, indent=2)
        return f"""You are a conservative compiler optimizing an already assembled typed Jev question map.

Rewrite only string values under `instructions` and `criteria`. Make the complete map smaller while preserving every decision rule and its precedence. Preserve every protected literal listed below verbatim, along with all include rules, exclusion rules, exceptions, scope restrictions, and per-subject overrides. Remove output-format prose (for example requests for JSON, one word, an API choice, or no explanation) because the typed API controls the response. Do not repair, reinterpret, summarize away, or invent policy.

Return JSON only with exactly this shape:
{{"questions": <the optimized question map>}}

The question IDs and their order, each question type, all object/list shapes, and every criterion label and its order must be identical to the input. Do not add fields.

PROTECTED LITERALS BY QUESTION (every value must remain in that question):
{protected_json}

ASSEMBLED JEV QUESTIONS:
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
    def _parse_json_response(text: str) -> QuestionMap:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise PromptOptimizationError(
                        f"Gemini optimizer returned duplicate key {key!r}"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(text, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as exc:
            raise PromptOptimizationError("Gemini optimizer returned invalid JSON") from exc
        if not isinstance(value, dict) or list(value) != ["questions"]:
            raise PromptOptimizationError(
                "Gemini optimizer response must contain only a questions object"
            )
        questions = value["questions"]
        if not isinstance(questions, dict):
            raise PromptOptimizationError("Gemini optimizer questions must be an object")
        return questions

    @classmethod
    def _validate_question_map(
        cls,
        optimized: QuestionMap,
        source: QuestionMap,
        snapshot: dict[str, Any],
        defective_question_ids: set[str] | None = None,
    ) -> None:
        defective_ids = defective_question_ids or set()
        if list(optimized) != list(source):
            raise PromptOptimizationError(
                "Optimized question IDs or order differ from the assembled source"
            )

        known_literals = cls._known_literals_by_question(snapshot)
        for question_id, source_question in source.items():
            optimized_question = optimized.get(question_id)
            if not isinstance(optimized_question, dict):
                raise PromptOptimizationError(
                    f"Optimized question {question_id} must be an object"
                )
            if question_id in defective_ids:
                if optimized_question != source_question:
                    raise PromptOptimizationError(
                        f"Defective question {question_id} was not preserved verbatim"
                    )
                continue
            if list(optimized_question) != list(source_question):
                raise PromptOptimizationError(
                    f"Optimized question {question_id} changed its field set or order"
                )
            if optimized_question.get("type") != source_question.get("type"):
                raise PromptOptimizationError(
                    f"Optimized question {question_id} changed its type"
                )
            for field, source_value in source_question.items():
                if field == "type":
                    continue
                if field not in ("instructions", "criteria"):
                    if optimized_question[field] != source_value:
                        raise PromptOptimizationError(
                            f"Optimized question {question_id} changed protected field {field}"
                        )
                    continue
                cls._validate_text_tree(
                    optimized_question[field],
                    source_value,
                    f"{question_id}.{field}",
                )

            source_text = _all_text(source_question)
            optimized_text = _all_text(optimized_question)
            if any(pattern.search(optimized_text) for pattern in _OUTPUT_FORMAT_PATTERNS):
                raise PromptOptimizationError(
                    f"Optimized question {question_id} retains output-format prose"
                )

            protected_literals = _quoted_literals(source_text)
            protected_literals.update(_numeric_literals(source_text))
            protected_literals.update(known_literals.get(question_id, set()))
            missing = sorted(
                literal for literal in protected_literals if literal not in optimized_text
            )
            if missing:
                raise PromptOptimizationError(
                    f"Optimized question {question_id} lost protected source literals: "
                    + ", ".join(repr(value) for value in missing[:5])
                )

            source_lower = source_text.lower()
            optimized_lower = optimized_text.lower()
            source_policy_text = source_lower
            for output_pattern in _OUTPUT_FORMAT_PATTERNS:
                source_policy_text = output_pattern.sub("", source_policy_text)
            for marker, (source_pattern, optimized_pattern) in _POLICY_MARKERS.items():
                if source_pattern.search(source_policy_text) and not optimized_pattern.search(
                    optimized_lower
                ):
                    raise PromptOptimizationError(
                        f"Optimized question {question_id} lost policy marker {marker!r}"
                    )
            if re.search(r"\b(?:not|never|false|unrelated|fails?)\b", source_lower) and not re.search(
                r"\b(?:not|never|false|exclude|unrelated|fails?)\b", optimized_lower
            ):
                raise PromptOptimizationError(
                    f"Optimized question {question_id} lost negative policy semantics"
                )
        source_chars = len(_canonical_json(source))
        optimized_chars = len(_canonical_json(optimized))
        if optimized_chars >= source_chars:
            raise PromptOptimizationError(
                "Optimized question map must be smaller than the assembled source "
                f"({optimized_chars} >= {source_chars} characters)"
            )

    @classmethod
    def _validate_text_tree(cls, optimized: Any, source: Any, path: str) -> None:
        if isinstance(source, str):
            if not isinstance(optimized, str) or not optimized.strip():
                raise PromptOptimizationError(f"Optimized text at {path} is empty or invalid")
            return
        if isinstance(source, dict):
            if not isinstance(optimized, dict) or list(optimized) != list(source):
                raise PromptOptimizationError(
                    f"Optimized criteria labels or order changed at {path}"
                )
            for key, source_value in source.items():
                cls._validate_text_tree(optimized[key], source_value, f"{path}.{key}")
            return
        if isinstance(source, list):
            if not isinstance(optimized, list) or len(optimized) != len(source):
                raise PromptOptimizationError(
                    f"Optimized criteria list shape changed at {path}"
                )
            for index, source_value in enumerate(source):
                cls._validate_text_tree(optimized[index], source_value, f"{path}[{index}]")
            return
        if optimized != source:
            raise PromptOptimizationError(f"Optimized non-text value changed at {path}")

    @staticmethod
    def _known_literals_by_question(snapshot: dict[str, Any]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for subject in snapshot.get("subjects", []):
            subject_id = str(subject.get("id"))
            subject_literals = _alias_values(
                {
                    "name": subject.get("name"),
                    "aliases": subject.get("aliases")
                    or subject.get("subject_aliases")
                    or subject.get("alias"),
                }
            )
            for suffix in ("valid", "prominence", "sentiment"):
                result[f"subj_{subject_id}_{suffix}"] = set(subject_literals)
            for tag in subject.get("tag_evaluations", []):
                if tag.get("evaluation_type") != "llm" or not tag.get("prompt_text"):
                    continue
                tag_literals = set(subject_literals)
                tag_literals.update(
                    _alias_values(
                        {
                            "name": tag.get("tag_name"),
                            "aliases": tag.get("aliases") or tag.get("tag_aliases") or tag.get("alias"),
                        }
                    )
                )
                result[f"tag_{subject_id}_{tag.get('tag_id')}"] = tag_literals
        return result

    @classmethod
    def _detect_anomalies(
        cls, questions: QuestionMap, snapshot: dict[str, Any]
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        raw_sources: dict[str, str] = {}
        tags_by_subject: dict[str, list[dict[str, Any]]] = {}
        for subject in snapshot.get("subjects", []):
            subject_id = str(subject.get("id"))
            raw_sources[f"subj_{subject_id}_valid"] = "\n".join(
                filter(
                    None,
                    (
                        str(subject.get("entity_definition") or ""),
                        str(subject.get("validation_prompt") or ""),
                    ),
                )
            )
            raw_sources[f"subj_{subject_id}_prominence"] = str(
                subject.get("prominence_prompt") or ""
            )
            raw_sources[f"subj_{subject_id}_sentiment"] = str(
                subject.get("sentiment_prompt") or ""
            )
            tags = [
                tag
                for tag in subject.get("tag_evaluations", [])
                if tag.get("evaluation_type") == "llm" and tag.get("prompt_text")
            ]
            tags_by_subject[subject_id] = tags
            for tag in tags:
                raw_sources[f"tag_{subject_id}_{tag.get('tag_id')}"] = str(
                    tag.get("prompt_text") or ""
                )

        for question_id in questions:
            source_text = raw_sources.get(question_id, _all_text(questions[question_id]))
            if _CONFIGURATOR_RE.search(source_text):
                diagnostics.append(
                    {
                        "question_id": question_id,
                        "code": "configurator_facing",
                        "message": "Source asks a configurator to supply policy rather than defining a decision.",
                    }
                )
                continue
            if cls._has_rule_contradiction(source_text):
                diagnostics.append(
                    {
                        "question_id": question_id,
                        "code": "contradictory_policy",
                        "message": "Source applies both include and exclude semantics to the same literal.",
                    }
                )
                continue
            tag_parts = question_id.split("_", 2)
            if len(tag_parts) == 3 and tag_parts[0] == "tag":
                subject_id, tag_id = tag_parts[1], tag_parts[2]
                tags = tags_by_subject.get(subject_id, [])
                tag = next((item for item in tags if str(item.get("tag_id")) == tag_id), None)
                if tag is not None and cls._has_tag_name_mismatch(tag, tags):
                    diagnostics.append(
                        {
                            "question_id": question_id,
                            "code": "tag_name_mismatch",
                            "message": "Source tag prompt names a different tag than its configured owner.",
                        }
                    )
        return diagnostics

    @staticmethod
    def _has_rule_contradiction(text: str) -> bool:
        positive_literals: set[str] = set()
        negative_literals: set[str] = set()
        for clause in re.split(r"(?<=[.!?;])\s+|\n+", text):
            literals = _quoted_literals(clause)
            if not literals:
                continue
            if _POSITIVE_RULE_RE.search(clause):
                positive_literals.update(literals)
            if _NEGATIVE_RULE_RE.search(clause):
                negative_literals.update(literals)
        return bool(positive_literals & negative_literals)

    @staticmethod
    def _has_tag_name_mismatch(
        tag: dict[str, Any], sibling_tags: list[dict[str, Any]]
    ) -> bool:
        prompt = str(tag.get("prompt_text") or "")
        configured_name = str(tag.get("tag_name") or "").strip()
        explicit_names = {
            match.group(1).strip() for match in _EXPLICIT_TAG_NAME_RE.finditer(prompt)
        }
        if explicit_names and configured_name not in explicit_names:
            return True
        prompt_lower = prompt.lower()
        if configured_name and configured_name.lower() not in prompt_lower:
            for sibling in sibling_tags:
                sibling_name = str(sibling.get("tag_name") or "").strip()
                if sibling_name and sibling_name != configured_name and sibling_name.lower() in prompt_lower:
                    return True
        return False


prompt_optimizer = PromptOptimizer()
