import re
from typing import Any, Dict, List, Tuple

def sanitize_heading(text: str) -> str:
    """Strips leading markdown heading markers (#) from text lines."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = [re.sub(r"^\s*#+\s*", "", line) for line in lines]
    return "\n".join(cleaned).strip()

def build_article_state(blob_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Constructs the target state string and metadata for Jev from the archived blob.
    State contains article context (headline and body) formatted clearly for System One evaluation.
    """
    inbound = blob_data.get("inbound_data", {})
    headline = inbound.get("headline") or blob_data.get("headline") or ""
    body = inbound.get("body") or blob_data.get("body") or ""
    media_type = inbound.get("media_type") or blob_data.get("media_type") or "article"
    source = inbound.get("source") or blob_data.get("source") or ""
    published_at = inbound.get("published_at") or blob_data.get("published_at") or ""

    is_clip = str(media_type).lower() in ("radio", "television", "tv")
    item_kind = "clip (broadcast transcript)" if is_clip else "written article"

    state_text = f"Item Type: {item_kind}\n"
    if source:
        state_text += f"Source: {source}\n"
    if published_at:
        state_text += f"Published: {published_at}\n"
    state_text += f"\nHeadline: {headline}\n\nBody:\n{body}"

    metadata = {
        "headline": headline,
        "media_type": media_type,
        "source": source,
        "published_at": published_at,
        "is_clip": is_clip,
        "body_length": len(body)
    }
    return state_text, metadata

def convert_config_to_jev_questions(
    snapshot: Dict[str, Any],
    optimized_rubric: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Transforms Curation Engine subjects and LLM tags into typed TypeSafe Jev questions.
    An optional optimized rubric replaces only the client rule text; canonical labels
    and question types remain controlled by this converter.
    """
    questions: Dict[str, Dict[str, Any]] = {}
    subjects = snapshot.get("subjects", [])
    optimized_subjects = {
        str(s.get("subject_id")): s
        for s in (optimized_rubric or {}).get("subjects", [])
    }

    for s in subjects:
        s_id = str(s["id"])
        name = s.get("name") or f"Subject {s_id}"
        entity_def = sanitize_heading(s.get("entity_definition") or "")
        rubric = optimized_subjects.get(s_id) or {}
        val_prompt = sanitize_heading(rubric.get("validation_criteria") or s.get("validation_prompt") or "")
        prom_prompt = sanitize_heading(rubric.get("prominence_criteria") or s.get("prominence_prompt") or "")
        sent_prompt = sanitize_heading(rubric.get("sentiment_criteria") or s.get("sentiment_prompt") or "")
        optimized_tags = {
            str(tag.get("tag_id")): tag.get("criteria", "")
            for tag in rubric.get("tags", [])
        }

        # 1. Subject Validation (Noul)
        val_instructions = f"Is the provided content meaningfully relevant to '{name}' based on the validation criteria?"
        val_criteria = {
            "true": f"Content satisfies validation criteria for {name}:\nEntity: {entity_def}\nCriteria: {val_prompt}",
            "false": f"Content is unrelated, passing mention without substance, or fails validation criteria for {name}."
        }
        questions[f"subj_{s_id}_valid"] = {
            "type": "noul",
            "instructions": val_instructions,
            "criteria": val_criteria
        }

        # 2. Prominence (Choice: primary, significant, passing)
        prom_instructions = (
            f"What is the prominence of '{name}' in this content? "
            "Return exactly one API choice: primary, significant, or passing. "
            "If the client criteria uses 'Prime', treat it as the primary choice."
        )
        if prom_prompt:
            prom_instructions += f"\n\nApply these client-specific prominence criteria:\n{prom_prompt}"
        prom_criteria = {
            "primary": "Primary focus of the article or clip.",
            "significant": "Significant mention, key topic, or major discussion, but not the exclusive main topic.",
            "passing": "Minor, passing reference, incidental mention, or brief quote."
        }
        questions[f"subj_{s_id}_prominence"] = {
            "type": "choice",
            "instructions": prom_instructions,
            "criteria": prom_criteria
        }

        # 3. Sentiment (Choice: positive, negative, neutral, balanced)
        sent_instructions = f"What is the tone and sentiment towards '{name}' in this content?"
        if sent_prompt:
            sent_instructions += f"\n\nApply these client-specific sentiment criteria:\n{sent_prompt}"
        sent_criteria = {
            "positive": "Favorable coverage, accomplishments, praise, or positive development.",
            "negative": "Adverse coverage, criticism, scandal, controversies, or poor performance.",
            "neutral": "Factual reporting without a clear positive or negative tone.",
            "balanced": "Substantial elements of both positive and negative sentiment."
        }
        questions[f"subj_{s_id}_sentiment"] = {
            "type": "choice",
            "instructions": sent_instructions,
            "criteria": sent_criteria
        }

        # 4. LLM-evaluated tags only
        for t in s.get("tag_evaluations", []):
            if t.get("evaluation_type") == "llm" and t.get("prompt_text"):
                t_id = t["tag_id"]
                t_name = t.get("tag_name") or t_id
                t_criteria = sanitize_heading(optimized_tags.get(str(t_id)) or t.get("prompt_text") or "")
                
                questions[f"tag_{s_id}_{t_id}"] = {
                    "type": "noul",
                    "instructions": f"Does the content match the tag criteria for '{t_name}' regarding '{name}'?",
                    "criteria": {
                        "true": f"Criteria satisfied for {t_name}: {t_criteria}",
                        "false": f"Criteria NOT satisfied for {t_name}."
                    }
                }

    return questions
