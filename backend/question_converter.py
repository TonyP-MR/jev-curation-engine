import re
from typing import Any

from laya_questions import STATE_MAX_CHARS, build_laya_state, sanitize_heading

ASSEMBLER_VERSION = "2"


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


# Subject labels are curator-authored and only ever carry `name`, `entity_definition`
# and the three prompts — no structured alias list exists in any config snapshot.
# Across 1430 subjects in the cached configs the label is the entity plus, at most, a
# parenthetical qualifier or acronym ("Chobani (creamers)", "Museum of Modern Art
# (MoMA)") or a curation scope phrase ("Redwire All"). Matching the label verbatim
# therefore fails on any decorated name, so matching is done on the label's
# distinctive tokens instead of per-config suffix lists.
_PARENTHETICAL_RE = re.compile(r"\(([^)]{1,40})\)")
_CORP_SUFFIX_RE = re.compile(
    r"(?i)[\s,]+(?:group|corp|corporation|inc\.?|llc|l\.l\.c\.?|ltd\.?|limited|co\.?|company|"
    r"pbc|plc|enterprises|holdings?|ag|nv|bv|sa|gmbh|a/s)$"
)

# Tokens that never identify an entity on their own: corporate/legal boilerplate,
# curation scope words, and category nouns. A label reduced to only these words falls
# back to the whole label.
NON_DISTINCTIVE_TOKENS = {
    "all", "only", "mentions", "mention", "coverage", "exclude", "excluding", "excl",
    "include", "including", "incl", "and", "the", "of", "for", "de", "du", "van", "von",
    "group", "corp", "corporation", "inc", "llc", "ltd", "limited", "co", "company",
    "pbc", "plc", "enterprises", "holding", "holdings", "ag", "nv", "bv", "sa", "gmbh",
    "stocks", "financials", "news", "brand", "brands", "global", "international",
    "systems", "technologies", "solutions", "services", "media", "digital", "online",
    "network", "networks", "communications", "management", "capital", "partners",
    "energy", "health", "care", "foods", "software", "wireless", "mobile", "telecom",
}


def canonical_subject_name(s: dict[str, Any]) -> str:
    """Config label without its parenthetical qualifier ('Chobani (creamers)' -> 'Chobani')."""
    name = (s.get("name") or "").strip()
    return _PARENTHETICAL_RE.sub(" ", name).strip().strip(" \t-–—:,") or name


def subject_match_terms(s: dict[str, Any]) -> list[str]:
    """Strict surface forms of the subject: full label, legal-suffix-free base,
    parenthetical acronym, and hyphen components of compound brands.

    These carry the entity identity on their own, so they are safe to count. Mention
    counts drive the prominence floor/ceiling heuristics, where a loose term would
    inflate an incidental article into 'significant'.
    """
    raw_name = (s.get("name") or "").strip()
    canonical = canonical_subject_name(s)
    if not canonical:
        return []

    terms: set[str] = {canonical}

    base = _CORP_SUFFIX_RE.sub("", canonical).strip()
    if len(base) > 2:
        terms.add(base)

    for qualifier in _PARENTHETICAL_RE.findall(raw_name):
        qualifier = qualifier.strip()
        # Acronyms only: "(MoMA)", "(CHOP)", "(BLLT)". Category notes such as
        # "(creamers)" or "(RTD Coffee)" are not surface forms of the entity.
        is_acronym = 2 <= len(qualifier) <= 10 and qualifier == qualifier.replace(" ", "") and sum(
            1 for c in qualifier if c.isupper()
        ) >= 2
        if is_acronym:
            terms.add(qualifier)

    for compound in (canonical, base):
        if "-" in compound and not compound.startswith("-"):
            for part in compound.split("-"):
                part = part.strip()
                if len(part) >= 3 and part.lower() not in NON_DISTINCTIVE_TOKENS:
                    terms.add(part)

    return sorted({t for t in terms if len(t) >= 2}, key=lambda x: -len(x))


def subject_presence_terms(s: dict[str, Any]) -> list[str]:
    """Strict terms plus the individual distinctive tokens of the label.

    Token-level terms are what let 'Redwire All' match an article about Redwire
    Corporation without a per-config qualifier list. They over-match on generic words
    ("Museum", "Free"), so they are only for the advisory presence check, where a hit
    lowers the validation bar rather than deciding the answer.
    """
    canonical = canonical_subject_name(s)
    if not canonical:
        return []
    terms = set(subject_match_terms(s))
    base = _CORP_SUFFIX_RE.sub("", canonical).strip() or canonical
    terms.update(
        t for t in re.findall(r"[\w&'’+.]+", base)
        if len(t) >= 4 and t.lower() not in NON_DISTINCTIVE_TOKENS
    )
    return sorted(terms, key=lambda x: -len(x))


def _boundary_patterns(terms: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]


def subject_mention_patterns(s: dict[str, Any]) -> list[re.Pattern[str]]:
    """Word-boundary regexes over the strict terms — use for counting mentions."""
    return _boundary_patterns(subject_match_terms(s))


def subject_presence_patterns(s: dict[str, Any]) -> list[re.Pattern[str]]:
    """Word-boundary regexes over the widened terms — use for presence evidence."""
    return _boundary_patterns(subject_presence_terms(s))


def build_distilled_article_state(
    blob_data: dict[str, Any], max_chars: int = STATE_MAX_CHARS
) -> tuple[str, dict[str, Any]]:
    """Laya article state plus the headline/lead entity sets the caller needs.

    The state text itself comes from :func:`build_laya_state` so it is byte-identical
    to what sequence prep produced for training. Boilerplate stripping and per-section
    caps used to live here; they changed the text the model saw relative to its
    training distribution, so they are gone.
    """
    inbound = blob_data.get("inbound_data", {})
    headline = inbound.get("headline") or blob_data.get("headline") or ""
    body = inbound.get("body") or blob_data.get("body") or ""
    media_type = inbound.get("media_type") or blob_data.get("media_type") or "article"
    source = inbound.get("source") or blob_data.get("source") or ""
    published_at = inbound.get("published_at") or blob_data.get("published_at") or ""

    distilled_state = build_laya_state(blob_data, max_chars=max_chars)

    subject_map: dict[str, dict[str, Any]] = {}
    for s in blob_data.get("subjects") or []:
        if s.get("name"):
            subject_map[s["name"]] = s
    for s in blob_data.get("subject_results") or []:
        name = s.get("name")
        if name and name not in subject_map:
            subject_map[name] = s

    # Lead is the first paragraph, matching the state layout exactly.
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    lead_text = paragraphs[0] if paragraphs else ""

    hl_entities = set()
    lead_entities = set()
    for s_name, s_dict in subject_map.items():
        alias_res = (
            subject_mention_patterns(s_dict)
            if s_dict
            else [re.compile(r"\b" + re.escape(s_name) + r"\b", re.IGNORECASE)]
        )
        if any(r.search(headline) for r in alias_res):
            hl_entities.add(s_name)
        if any(r.search(lead_text) for r in alias_res):
            lead_entities.add(s_name)
    metadata = {
        "headline": headline,
        "media_type": media_type,
        "source": source,
        "published_at": published_at,
        "is_clip": str(media_type).lower() in ("radio", "television", "tv"),
        "body_length": len(body),
        "distilled": True,
        "distilled_chars": len(distilled_state),
        "headline_entities": list(hl_entities),
        "lead_entities": list(lead_entities),
    }
    return distilled_state, metadata


def convert_config_to_jev_questions(
    snapshot: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Deterministically assemble Curation Engine policy into typed Jev questions.

    This function is the sole raw Jev assembly boundary. Prompt optimization, when
    requested, operates on its complete returned map rather than feeding rubric
    fragments back into this converter.
    """
    questions: dict[str, dict[str, Any]] = {}
    subjects = snapshot.get("subjects", [])

    for s in subjects:
        s_id = str(s["id"])
        name = s.get("name") or f"Subject {s_id}"
        entity_def = sanitize_heading(s.get("entity_definition") or "")
        val_prompt = sanitize_heading(s.get("validation_prompt") or "")
        prom_prompt = sanitize_heading(s.get("prominence_prompt") or "")
        sent_prompt = sanitize_heading(s.get("sentiment_prompt") or "")

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
                raw_criteria = sanitize_heading(t.get("prompt_text") or "")
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
