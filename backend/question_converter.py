import re
from typing import Any


def sanitize_heading(text: str) -> str:
    """Strips leading markdown heading markers (#) from text lines."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = [re.sub(r"^\s*#+\s*", "", line) for line in lines]
    return "\n".join(cleaned).strip()

def build_article_state(blob_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
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
    # Jev limit: 32k tokens max for state (~120k characters).
    # Truncate exceptionally long articles (e.g. 500k+ chars transcript/dump) so request stays in context.
    MAX_STATE_CHARS = 120_000
    full_state = f"Item Type: {item_kind}\n"
    if source:
        full_state += f"Source: {source}\n"
    if published_at:
        full_state += f"Published: {published_at}\n"
    full_state += f"\nHeadline: {headline}\n\nBody:\n{body}"

    if len(full_state) > MAX_STATE_CHARS:
        state_text = full_state[:MAX_STATE_CHARS] + "\n\n[... article body truncated due to context window limits ...]"
    else:
        state_text = full_state
    metadata = {
        "headline": headline,
        "media_type": media_type,
        "source": source,
        "published_at": published_at,
        "is_clip": is_clip,
        "body_length": len(body)
    }
    return state_text, metadata
def strip_boilerplate_and_recirculation(body: str) -> str:
    """Strips recirculation footers, syndicated 'Read More' links, CMS bylines, and photo credits."""
    if not body:
        return ""

    # Strip CMS metadata, photo credits, and journalist bylines
    byline_patterns = [
        r'(?i)(?:foto|photo|image|credit|courtesy|source)\s*:\s*[^\n]+',
        r'(?i)\b[a-zA-Z0-9_\.\-]+\s*\|\|\s*\d{4}-\d{2}-\d{2}[^\n]*',
        r'(?i)\b[a-zA-Z0-9_\.\-]+\s*\|\|\s*\d{1,2}:\d{2}[^\n]*',
        r'(?i)\b(?:by|author|reporter|editor|dipublikasikan oleh)\s*:\s*[a-zA-Z\s]{3,40}(?:\n|$)',
    ]
    for bp in byline_patterns:
        body = re.sub(bp, '', body)

    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    if len(paragraphs) <= 2:
        return body

    footer_regex = re.compile(
        r"^(?:"
        r"read\s+more|related\s+(?:stories|articles?|posts?|links?)|recommended\s+(?:for\s+you|stories)|"
        r"trending\s+(?:stories|now|topics)|popular\s+stories|also\s+on|see\s+also|more\s+from|"
        r"sponsored\s+content|latest\s+news|top\s+stories|editors'?\s+picks?|advertisement|"
        r"sign\s+up\s+for|subscribe\s+to|newsletter|share\s+this\s+article|"
        r"follow\s+us\s+on|copyright\s+©|all\s+rights\s+reserved|"
        r"source\s*:\s*[A-Za-z0-9\s]+$|"
        r"photo\s*:\s*[A-Za-z0-9\s]+$"
        r")[:\s-]",
        re.IGNORECASE,
    )

    cut_index = len(paragraphs)
    for i in range(len(paragraphs) - 1, max(1, len(paragraphs) - 8), -1):
        p = paragraphs[i]
        if footer_regex.match(p) or (p.startswith(("•", "*", "-")) and any(w in p.lower() for w in ["read:", "related:", "source:", "photo:"])):
            cut_index = i

    cleaned = paragraphs[:cut_index]
    return "\n\n".join(cleaned)


def extract_subject_aliases(s: dict[str, Any]) -> list[str]:
    """Extracts official entity name, compound brands (e.g. Air France, KLM for Air France-KLM),
    acronyms, and operating unit tag names from a subject definition."""
    name = s.get("name", "").strip()
    if not name:
        return []

    aliases: set[str] = {name}

    clean_base = re.sub(
        r"(?i)\s+(group|corp|corporation|inc\.?|llc|ltd\.?|limited|co\.?|company|pbc|enterprises|holdings?)$",
        "",
        name,
    ).strip()
    if clean_base and len(clean_base) > 2:
        aliases.add(clean_base)

    GENERIC_CATEGORY_WORDS = {
        "cola", "systems", "technologies", "software", "foods", "health", 
        "group", "holdings", "brands", "international", "global", "solutions",
        "network", "networks", "enterprises", "energy", "capital", "partners"
    }
    for compound in [name, clean_base]:
        if "-" in compound and not compound.startswith("-"):
            parts = [
                p.strip() for p in compound.split("-") 
                if len(p.strip()) >= 3 and p.lower() not in GENERIC_CATEGORY_WORDS
            ]
            for p in parts:
                aliases.add(p)

    # Add canonical informal aliases
    if name.lower() in ("coca-cola", "coca cola"):
        aliases.add("Coke")
    raw_text = " ".join([
        s.get("entity_definition") or "",
        s.get("validation_prompt") or "",
        s.get("prominence_prompt") or "",
    ])
    for m in re.finditer(r"\(([A-Za-z0-9\s&/-]{2,30})\)", raw_text):
        cand = m.group(1).strip()
        if "misspell" in cand.lower():
            continue
        if not any(stop in cand.lower() for stop in ["http", "e.g.", "i.e.", "formerly", "including", "such as", "see", "no "]):
            for sub in cand.split(","):
                sub = sub.strip()
                if 2 <= len(sub) <= 25 and not sub.lower().startswith("nyse:") and not sub.lower().startswith("nasdaq:"):
                    aliases.add(sub)
    for m in re.finditer(r"(?i)(?:referred to (?:in media )?as|known as|often called)\s+[\"\'“]?([A-Za-z0-9\s&/-]{2,35})[\"\'”]?", raw_text):
        cand = m.group(1).strip()
        if 2 <= len(cand) <= 30:
            aliases.add(cand)

    for t in s.get("tag_evaluations", []):
        t_name = t.get("tag_name", "").strip()
        t_group = (t.get("tag_group_name") or "").lower()
        if t_name and any(w in t_group for w in ["brand", "group", "subsidiary", "operating unit", name.lower()]):
            aliases.add(t_name)

    stop_words = {"the", "group", "and", "all", "article", "company", "no explanation", "no other text"}
    filtered = []
    for a in sorted(aliases, key=lambda x: -len(x)):
        if a.lower() not in stop_words and len(a) >= 2:
            filtered.append(a)
    return filtered


def build_distilled_article_state(
    blob_data: dict[str, Any], max_chars: int = 6500
) -> tuple[str, dict[str, Any]]:
    """Constructs a high-density, entity-grounded distilled state for Laya decision models.

    Filters boilerplate, resolves corporate aliases, extracts entity-specific excerpts,
    and enforces headline primacy rules.
    """
    inbound = blob_data.get("inbound_data", {})
    headline = inbound.get("headline") or blob_data.get("headline") or ""
    raw_body = inbound.get("body") or blob_data.get("body") or ""
    body = strip_boilerplate_and_recirculation(raw_body)
    media_type = inbound.get("media_type") or blob_data.get("media_type") or "article"
    source = inbound.get("source") or blob_data.get("source") or ""
    published_at = inbound.get("published_at") or blob_data.get("published_at") or ""

    is_clip = str(media_type).lower() in ("radio", "television", "tv")
    item_kind = "clip (broadcast transcript)" if is_clip else "written article"

    # Build map of subjects
    subject_map: dict[str, dict[str, Any]] = {}
    for s in blob_data.get("subjects") or []:
        if s.get("name"):
            subject_map[s["name"]] = s
    for s in blob_data.get("subject_results") or []:
        name = s.get("name")
        if name and name not in subject_map:
            subject_map[name] = s

    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    total_paras = max(1, len(paragraphs))
    lead_text = "\n\n".join(paragraphs[:2]) if paragraphs else ""
    sentences = re.split(r"(?<=[.!?])\s+", body) if body else []

    # Clean article representation without synthetic entity metrics or extraction artifacts
    sections = [f"Headline: {headline}\n"]
    if lead_text:
        sections.append(f"Lead Paragraph:\n{lead_text[:800]}")
    
    # Body sample (excluding lead if already present)
    remaining_body = body
    if lead_text and remaining_body.startswith(lead_text):
        remaining_body = remaining_body[len(lead_text):].strip()
    if remaining_body:
        sections.append(f"Article Body:\n{remaining_body[:2800]}")

    distilled_state = "\n\n".join(sections)
    if len(distilled_state) > max_chars:
        distilled_state = distilled_state[:max_chars]

    # Identify headline and lead entities for strict primacy enforcement
    hl_entities = set()
    lead_entities = set()
    for s_name, s_dict in subject_map.items():
        aliases = extract_subject_aliases(s_dict) if s_dict else [s_name]
        alias_res = [re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE) for a in aliases]
        if any(r.search(headline) for r in alias_res):
            hl_entities.add(s_name)
        if any(r.search(lead_text) for r in alias_res):
            lead_entities.add(s_name)
    metadata = {
        "headline": headline,
        "media_type": media_type,
        "source": source,
        "published_at": published_at,
        "is_clip": is_clip,
        "body_length": len(body),
        "distilled": True,
        "distilled_chars": len(distilled_state),
        "headline_entities": list(hl_entities),
        "lead_entities": list(lead_entities),
    }
    return distilled_state, metadata
def convert_config_to_jev_questions(
    snapshot: dict[str, Any],
    optimized_rubric: dict[str, Any] | None = None,
    is_system_one: bool = False,
) -> dict[str, dict[str, Any]]:
    """Transforms Curation Engine subjects and LLM tags into typed TypeSafe Jev questions.

    An optional optimized rubric replaces only the client rule text; canonical labels
    and question types remain controlled by this converter. For System 1 (Laya), automatically
    compiles dense, non-truncating criteria when no custom rubric is supplied.
    """
    from prompt_optimizer import compile_system_one_rubric

    rubric_source = optimized_rubric
    if is_system_one and not rubric_source:
        rubric_source = compile_system_one_rubric(snapshot)

    questions: dict[str, dict[str, Any]] = {}
    subjects = snapshot.get("subjects", [])
    optimized_subjects = {
        str(s.get("subject_id")): s
        for s in (rubric_source or {}).get("subjects", [])
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
        val_instructions = (
            f"Is the provided content meaningfully relevant to '{name}' based on the validation criteria?\n\n"
            f"MANDATORY HEADLINE PRIMACY: If '{name}' (or its operating brand/alias) is featured in the HEADLINE or lead paragraph "
            "(e.g., corporate appointment, earnings release, partnership, product launch), it is DEFINITIONALLY VALID AND RELEVANT."
        )
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
            "Return exactly one API choice: primary, significant, or passing.\n\n"
            "PROMINENCE RULES:\n"
            f"1. HEADLINE PRIMACY: If '{name}' is named in the HEADLINE, it MUST be classified as 'primary' (if it is the central story topic) "
            "or 'significant' (if shared with another entity). It is STRICTLY PROHIBITED from being classified as passing.\n"
            f"2. LEAD PARAGRAPH FOCUS: If '{name}' is discussed substantively in the lead paragraph, classify as 'primary' or 'significant'. Never passing.\n"
            f"3. PASSING DEFAULT: Classify as 'passing' ONLY for incidental references: {name} appears in a multi-vendor catalog, "
            "is cited as an external data source, or mentioned briefly 1–2 times without headline/lead focus."
        )
        if prom_prompt:
            prom_instructions += f"\n\nClient Prominence Criteria:\n{prom_prompt}"
        prom_criteria = {
            "primary": f"Primary dominant focus of the article or clip. {name} is the central subject featured in the headline, lead, and narrative.",
            "significant": f"Key topic discussed substantively, or featured in the headline/lead. MUST NOT be chosen for incidental mentions in multi-vendor lists or external data citations.",
            "passing": f"Default for incidental references: {name} is mentioned in a list of multiple companies, cited as a secondary data source, or mentioned briefly without headline focus."
        }
        questions[f"subj_{s_id}_prominence"] = {
            "type": "choice",
            "instructions": prom_instructions,
            "criteria": prom_criteria
        }

        # 3. Sentiment (Choice: positive, negative, neutral, balanced)
        sent_instructions = (
            f"What is the specific tone and sentiment towards '{name}' in this content? "
            f"CRITICAL: Isolate sentiment strictly to '{name}'. Do NOT inherit positive or negative tone from general market mood or other companies.\n\n"
            "SENTIMENT EDITORIAL BASELINE RULES:\n"
            f"- NEUTRAL DEFAULT: Standard operational transactions, financial earnings releases, corporate appointments/hires, "
            "funding announcements, business contracts, game recaps, player transactions, roster moves, and analytical reporting "
            f"MUST DEFAULT TO NEUTRAL for '{name}'.\n"
            f"- POSITIVE: Reserved strictly for overt, explicit institutional praise, prestigious awards, celebratory acclaim, "
            f"or exceptional historic breakthroughs directed specifically at '{name}'. Merely conducting business or reporting quarterly results is NEUTRAL.\n"
            f"- NEGATIVE: Direct adverse coverage, criticism, regulatory investigations, lawsuits, scandals, severe losses, or customer condemnation directed specifically at '{name}'."
        )
        if sent_prompt:
            sent_instructions += f"\n\nClient Sentiment Criteria:\n{sent_prompt}"
        sent_criteria = {
            "positive": f"Explicit institutional praise, prestigious awards, or celebratory acclaim directed specifically at {name}. Standard business transactions and earnings remain neutral.",
            "negative": f"Direct adverse coverage, criticism, scandal, litigation, or severe failure directed specifically at {name}.",
            "neutral": f"Editorial baseline: factual reporting, financial earnings, operational transactions, executive hires, sports recaps, player contracts, and routine industry news regarding {name}.",
            "balanced": f"Substantial elements of both explicit praise and severe criticism directed specifically at {name}."
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
                raw_criteria = sanitize_heading(optimized_tags.get(str(t_id)) or t.get("prompt_text") or "")
                t_lower = t_name.lower()
                
                # Negative guardrail injection for known confusion categories
                extra_guardrails = ""
                if any(w in t_lower for w in ["partnership", "partner", "alliance", "joint venture"]):
                    extra_guardrails = (
                        " Explicit negative exclusion: Standard third-party software compatibility, app store listings, "
                        "operating system integrations, APIs, generic client reviews, and multi-vendor list roundups DO NOT qualify "
                        "as a partnership. A partnership requires an explicit, announced mutual corporate or technology agreement."
                    )
                elif any(w in t_lower for w in ["developer", "devops", "engineering"]):
                    extra_guardrails = (
                        " Explicit negative exclusion: General consumer how-to guides, personal password tips, and general security "
                        "tutorials aimed at everyday end-users DO NOT qualify as Developer content. The content must specifically "
                        "target software engineers, developer tooling, SDKs, or programming workflows."
                    )
                elif t_lower == "ai" or "artificial intelligence" in t_lower:
                    extra_guardrails = (
                        " Explicit negative exclusion: Passing biographical mentions in author blurbs, routine automation, "
                        "or generic software algorithms DO NOT qualify. The content must substantively discuss artificial intelligence "
                        "or machine learning directly related to the subject."
                    )

                t_criteria = (raw_criteria + extra_guardrails).strip()
                
                questions[f"tag_{s_id}_{t_id}"] = {
                    "type": "noul",
                    "instructions": f"Does the content match the tag criteria for '{t_name}' regarding '{name}'?",
                    "criteria": {
                        "true": f"Criteria satisfied for {t_name}: {t_criteria}",
                        "false": f"Criteria NOT satisfied for {t_name}."
                    }
                }

    return questions
