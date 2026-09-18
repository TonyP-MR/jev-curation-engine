# Cookbooks & Practical Recipes Summary

TypeSafe documentation includes 18 recipes demonstrating System One in real-world workflows:

1. **Parallel Regulatory / Compliance Screening**
   - Batches dozens of questions (e.g. GDPR compliance checks) over a large document in a single request.
   - 10x-12x cheaper and faster than serial prompting without context rot.

2. **Re-Ranking Search Candidates (BM25 + Jev)**
   - Initial pass retrieves top-N documents with fast lexical search (BM25).
   - Jev evaluates each candidate in a batch Choice question to significantly improve Top-1 and Top-10 precision.

3. **Semantic Line-by-Line Search**
   - Scores individual line IDs of documents against a query via a single Choice question, with companion Noul questions verifying if the document answers the query at all.

4. **Structure & Markdown Recovery**
   - Two-pass pipeline: stitches wrapped text lines, then uses Choice questions to categorize block structures (headers, code, callouts, lists).

5. **Function / Tool Calling**
   - Converts natural-language requests into typed function parameters by mapping function options and arguments to confidence-gated Choice questions.

6. **Skill Selection for Autonomous Agents**
   - Evaluates a catalog of hundreds of skills in a fast parallel batch to select the top skill for an agent turn, rejecting non-applicable ones.

7. **Knowledge Graph Entity Alignment**
   - Evaluates candidate pairs across disjoint databases. A 3-level Score question acts as an action trigger: (0) Leave unlinked, (1) Flag for human curator, (2) Merge.

8. **RAG Passage Quality & Injection Filtering**
   - Evaluates retrieved passages before passing them to generative models: drops passages with prompt injections/harmful instructions, flags contradictions.

9. **Citation Verification & Hallucination Defense**
   - Compares generated citations against source passages. Choice/Noul questions verify if context supports the claim; confidence flags ungrounded statements.

10. **LLM Safety & Guardrails**
    - High-throughput pre- and post-generation screen: assesses jailbreak likelihood via Noul, scores potential harm severity via Score, applying automated blocks or human audits.

11. **SDE Cascade (Structured Data Extraction)**
    - Two-stage cascade: lightweight model extracts tentative schema fields, TypeSafe verifies and evaluates accuracy before invoking expensive heavy reasoning models.

12. **Date Extraction & Verification**
    - TypeSafe identifies relative and absolute date tokens from documents; application code deterministically parses, resolves, and compares timestamps.

13. **Pre-Parsed Value Extraction**
    - High-recall regex patterns extract candidate entities (emails, amounts, identifiers); TypeSafe Choice selects the matching semantic entity.

14. **Hierarchical Category Classification**
    - Deep taxonomy classification (e.g., patent codes, biomedical categories, retail catalogs) via parallel beam search using Choice probabilities.

15. **Feature Engineering for Classical ML (Autoresearch)**
    - Auto-generates qualitative domain questions evaluated by Jev to turn unstructured text into numerical calibrated probability features for downstream tabular models (e.g. CatBoost).

16. **Self-Consistency Audits (Nouls & Choices)**
    - Queries inverse propositions and evaluates probabilistic coherence to detect edge-case ambiguities needing manual review.

17. **Confidence-Guided Hierarchy Fallbacks**
    - Multi-level classification: if confidence is high, assign granular leaf categories; if confidence is low, fall back to the safe broader parent category.

18. **Smart Home Intent Processing**
    - Fast natural-language device intent mapping without generative latency or parsing risks.
