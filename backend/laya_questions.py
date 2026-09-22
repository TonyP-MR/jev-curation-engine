"""Canonical Laya prompt and state construction.

Single source of truth for everything the Laya encoder sees: the article state
string, the typed-decision questions, and the sequence budgets. Sequence prep
(`databricks/01_laya_sequence_prep.py`), the serving wrapper and the backend all
import from here, because three independent copies of this logic is exactly how
the served model ended up being asked questions it was never trained on.

Deliberately stdlib-only so a Databricks notebook can import the file directly
from its workspace folder without the rest of the repo or a wheel.

The prompts are terse on purpose. Laya scores option strings with an encoder; it
cannot follow instruction prose the way an LLM does, and every token of
"MANDATORY HEADLINE PRIMACY" boilerplate displaces real criteria inside
HEAD_MAX_LEN. Deterministic rules of that kind belong in the caller, not the
prompt.
"""

from __future__ import annotations

import re
from typing import Any

# Budgets from the checkpoint's rl_agent_config.json (encoder
# answerdotai/ModernBERT-large). Training sequences were built at these values,
# so inference MUST use them: a wider window at serving time puts the model in a
# truncation regime it never saw.
MAX_LEN = 1024
HEAD_MAX_LEN = 256

# Article text budget. Measured on the production corpus, the state is a median
# 782 tokens at this cap, which leaves room for the question inside MAX_LEN.
STATE_MAX_CHARS = 4000

# Client validation prose runs to thousands of characters. The head budget cuts
# it anyway; cutting it here keeps the cut deterministic and identical on both
# sides of the train/serve boundary.
VALIDATION_CRITERIA_MAX_CHARS = 600

PROMINENCE_LABELS = ("primary", "significant", "passing")
SENTIMENT_LABELS = ("positive", "negative", "neutral", "balanced")
PROMINENCE_MAP = {label: i for i, label in enumerate(PROMINENCE_LABELS)}
SENTIMENT_MAP = {label: i for i, label in enumerate(SENTIMENT_LABELS)}

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
    """Strips leading markdown heading markers from every line."""
    return "\n".join(re.sub(r"^\s*#+\s*", "", line) for line in (text or "").splitlines()).strip()


def tag_guardrail(tag_name: str) -> str:
    """Negative exclusions for tag families the model reliably over-triggers."""
    lowered = (tag_name or "").lower()
    if any(w in lowered for w in ("partnership", "partner", "alliance", "joint venture")):
        return PARTNERSHIP_GUARD
    if any(w in lowered for w in ("developer", "devops", "engineering")):
        return DEVELOPER_GUARD
    if lowered == "ai" or "artificial intelligence" in lowered:
        return AI_GUARD
    return ""


def build_laya_state(blob: dict[str, Any], max_chars: int = STATE_MAX_CHARS) -> str:
    """Article text in the exact layout the model was trained on."""
    inbound = blob.get("inbound_data") or {}
    headline = inbound.get("headline") or blob.get("headline") or ""
    body = inbound.get("body") or blob.get("body") or ""
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    lead = paragraphs[0] if paragraphs else ""
    rest = "\n\n".join(paragraphs[1:])
    state = f"Headline: {headline}\n\n\nLead Paragraph:\n{lead}\n\nArticle Body:\n{rest}"
    return state[:max_chars]


def build_laya_questions(
    snapshot: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the canonical train/serve question map for a Laya configuration."""
    questions: dict[str, dict[str, Any]] = {}
    for s in snapshot.get("subjects", []):
        s_id = str(s["id"])
        name = s.get("name") or f"Subject {s_id}"
        validation_prompt = sanitize_heading(s.get("validation_prompt") or "")

        questions[f"subj_{s_id}_valid"] = {
            "type": "noul",
            "instructions": (
                f"Is the provided content meaningfully relevant to '{name}' based on the validation criteria?"
            ),
            "criteria": {
                "true": f"Content satisfies validation criteria for {name}: "
                        f"{validation_prompt[:VALIDATION_CRITERIA_MAX_CHARS]}",
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
                "significant": "Key topic discussed substantively, or featured in the headline or lead. "
                               "Not for incidental mentions in multi-vendor lists.",
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
