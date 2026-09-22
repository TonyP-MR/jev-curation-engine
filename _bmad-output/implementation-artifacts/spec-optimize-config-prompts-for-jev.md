---
title: 'Optimize assembled config prompts for Jev'
type: 'feature'
created: '2026-09-22'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '5bb9e24e00f6d9b9af41ee006f8ec52bb881eaf5'
context:
  - 'type-safe-docs/primitives.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Jev currently receives verbose, conflicting question text because Gemini compacts only raw config rubrics and `question_converter.py` then wraps them in large global templates. Across the inspected corpus, every subject and LLM tag needs optimization without changing TypeSafe decision structure or silently overriding client policy.

**Approach:** Assemble the complete typed Jev question map first, then use the existing zero-temperature Gemini optimizer to rewrite only `instructions` and criteria text. Validate exact question IDs, types, criterion labels/order, source literals, and scope before caching or use; expose the same optimized artifact in preview and benchmark runs.

## Boundaries & Constraints

**Always:** Cover every subject validation, prominence, sentiment question and every subject-owned `llm` tag; preserve IDs, order, `noul`/`choice` types, label sets, aliases, names, numbers, include/exclude/exception precedence, and per-subject tag overrides. Keep Boolean tags in local evaluation. Cache by environment, numeric/slug identity, version, source snapshot hash, assembler/compiler version, model, and assembled input. Populate actual source/optimized character telemetry.

**Never:** Optimize Laya benchmark prompts at runtime; mutate source configs; send Boolean expressions to Jev; invent missing policy, aliases, or exceptions; retain the old rubric-overlay path beside the assembled-question optimizer; accept malformed, structurally changed, larger, or semantically unsafe LLM output.

**Source-policy decision:** Optimize every safe subject/tag question. Preserve only a detected defective question verbatim and report its diagnostic; never let Gemini infer a repair.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Valid Jev config | One or many subjects with mixed LLM/Boolean tags | Smaller typed map containing all subject questions and exactly the LLM-tag questions | Reject unsafe output; never use partial malformed output |
| Cache reuse | Same immutable config, assembler, model, and assembled map | Byte-equivalent optimized questions with `cached: true` | Corrupt cache is ignored and rebuilt atomically |
| Source anomaly | Contradiction, configurator-facing question, or tag/name mismatch | Optimize safe questions, preserve the defective question verbatim, and surface diagnostics | Never silently infer intent |
| Laya request | `optimize_prompts: true` with a Laya model/provider | No runtime prompt transformation | UI disables the option; API returns a clear 400 |
| Preview/run parity | Same Jev config and optimizer settings | Preview questions equal benchmark wire questions | Any optimizer failure is explicit, not mislabeled as optimized |

</frozen-after-approval>


## Code Map

- `backend/prompt_optimizer.py` -- replace rubric compilation with strict assembled-question optimization, diagnostics, immutable cache identity, atomic writes, and character telemetry.
- `backend/question_converter.py` -- retain deterministic raw Jev assembly; remove the old optimized-rubric overlay so there is one optimization boundary.
- `backend/laya_questions.py` -- remove benchmark-time rubric overlays; keep train/serve canonical prompts unchanged.
- `backend/main.py` -- share assemble-then-optimize logic between preview and benchmark paths; enforce Jev-only optimization.
- `backend/run_logger.py` -- consume the populated optimizer metadata without a parallel reporting contract.
- `frontend/src/App.jsx` -- preview the selected optimized Jev artifact, disable optimization for Laya, and show cache/reduction/diagnostic state.
- `backend/tests/test_prompt_optimizer.py` -- protect typed-shape invariants, anomaly policy, cache invalidation, and malformed-output rejection.

## Tasks & Acceptance

**Execution:**
- [x] `backend/prompt_optimizer.py` -- optimize complete typed maps and validate text-only changes -- prevent wrapper bloat and semantic drift.
- [x] `backend/question_converter.py`, `backend/laya_questions.py` -- remove rubric overlays and keep engine-specific raw assembly deterministic -- eliminate competing conventions.
- [x] `backend/main.py`, `frontend/src/App.jsx` -- route preview and benchmark through one Jev optimization path -- make the improved input visible and usable.
- [x] `backend/tests/test_prompt_optimizer.py` -- exercise invariant, failure, cache, and anomaly cases -- lock the consumer-visible contract.

**Acceptance Criteria:**
- Given any inspected cached config, when Jev questions are optimized, then every subject contributes validation, prominence, and sentiment and every LLM tag contributes one question with unchanged ID/type/labels; Boolean tags contribute none.
- Given unchanged inputs, when preview and benchmark run, then they use byte-equivalent optimized questions and the second compilation is a cache hit.
- Given output that drops a question, changes a type/label, loses protected source literals, adds output-format prose, or is not smaller, when validated, then it is rejected.
- Given the supplied Adventist Health venue-only crime article, when the optimized Jev path is exercised, then subject validation remains `false` and the exact wire questions are persisted.

## Implementation Notes

- Codex had already implemented the full feature (`backend/prompt_optimizer.py`, `backend/question_converter.py`, `backend/laya_questions.py`, `backend/main.py`, `backend/run_logger.py`, `frontend/src/App.jsx`, `backend/tests/test_prompt_optimizer.py`) before running out of usage; this session found no stubs, TODOs, or missing wiring and closed the spec out after verification.
- Confirmed `_assemble_questions` in `backend/main.py` is the single boundary shared by `/api/preview` and the benchmark job, so preview and benchmark questions are byte-equivalent and the second compilation is a cache hit.
- Confirmed the old rubric-overlay path is gone: `question_converter.py` only does raw deterministic assembly, and `prompt_optimizer.py` is the sole optimization boundary operating on the complete assembled map.
- `cd backend && uv run python -m unittest tests/test_prompt_optimizer.py` -- 7 tests pass (cache reuse/corruption/identity invalidation, every unsafe-output-class rejection, defective-question preservation, anomaly detection, preview/run parity, Laya rejection, subject/tag assembly).
- `cd backend && uv run pyright` -- the 71 reported errors are all pre-existing, in files this spec did not touch (`audit_index.py`, `blob_manager.py`, `config_manager.py`, `train_laya.py`); no errors in the changed files.
- `cd frontend && npm run build` -- succeeds. `npm run lint` -- only 2 pre-existing unused-variable warnings (`jobId`, `providersData`), both predating this diff and unrelated to it.
- Not run: the manual browser preview + live Jev benchmark against the supplied Adventist Health article (needs a live `GEMINI_API_KEY`/`TYPESAFE_API_KEY` session and the actual blob) — flagging as the one unverified acceptance criterion.

## Spec Change Log

## Review Triage Log

## Design Notes

Optimization occurs after typed assembly because that is the actual Jev input and the only layer where all global and client prose can be compacted together. Laya remains separate because its canonical prompts are coupled to training.

## Verification

**Commands:**
- `cd backend && uv run python -m unittest tests/test_prompt_optimizer.py` -- structural, cache, anomaly, and failure cases pass.
- `cd backend && uv run pyright` -- changed Python code has no type errors.
- `cd frontend && npm run build && npm run lint` -- UI integration compiles and lints.
- Browser preview followed by one direct Jev benchmark for the supplied Adventist article -- optimized preview equals persisted wire questions, validation is `false`, and a repeated preview reports a cache hit.
