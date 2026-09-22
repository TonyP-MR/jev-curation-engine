---
title: 'Re-enable direct Jev access as an app option'
type: 'feature'
created: '2026-09-22'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The app only exposes Jev through OpenRouter, whose account credit is exhausted. Direct TypeSafe credentials are already configured, but the direct option was removed after its old implementation targeted an invalid URL.

**Approach:** Restore TypeSafe Jev Direct as a selectable engine and route that selection per request to the documented `POST {TYPESAFE_API_BASE}/systemone` endpoint with `TYPESAFE_API_KEY`, while retaining the existing OpenRouter option and behavior.

</frozen-after-approval>

## Implementation Notes

- `type-safe-docs/api_and_sdk.md` defines the direct endpoint as `https://api.typesafe.ai/v1/systemone`; the configured `/v1/models` endpoint currently authenticates successfully.
- `backend/typesafe_runner.py` must resolve direct TypeSafe and OpenRouter as distinct per-request providers. The singleton's configured default must remain only a fallback, not override an explicit app selection.
- `backend/main.py:list_providers` and `frontend/src/App.jsx` must expose the direct model and send an explicit provider for both Jev choices.
- Preserve the user's existing uncommitted work in `backend/main.py` and unrelated files.
- Implemented per-request Jev transport selection in `backend/typesafe_runner.py`; direct calls use `{TYPESAFE_API_BASE}/systemone`, while OpenRouter retains `/api/alpha/decisions` and its attribution headers.
- Restored the direct provider in `backend/main.py` and added separate direct/OpenRouter choices with explicit provider mapping in `frontend/src/App.jsx`.
- Documented `TYPESAFE_API_BASE` in `.env.example` and the provider-specific routing in `README.md`.
- Verified a live direct request with `provider=typesafe` returned model `jev-1.13.0`, answer `is_blue`, and 278 input tokens.
- `uv run python -m py_compile typesafe_runner.py main.py config.py` and `npm run build` completed successfully.
- Browser verification showed both Jev options, selected `jev-latest`, rendered `TypeSafe Jev (Direct Cloud)`, and changed the run action to `Run Jev Benchmark`; `/api/providers` reported direct TypeSafe configured.
- Review corrected the stale provider-routing table in `databricks/README.md`; unrelated findings in the user's pre-existing benchmark and Databricks changes were recorded in `deferred-work.md`.

## Review Triage Log

- **medium / defer** — Absent subjects are gated at 0.65 but reported by post-processing and `compare_article_results` at 0.45, producing inconsistent validation semantics in pre-existing work.
- **medium / defer** — The pre-existing label-only matcher dropped aliases documented in entity definitions and prompts; the removed implementation confirms those aliases were previously consumed.
- **medium / defer** — The same matcher dropped brand, subsidiary, group, and operating-unit aliases from tag configuration, causing parent-subject false absences.
- **medium / defer** — Presence terms discard distinctive three-character tokens such as `Kia`, so decorated labels can be falsely treated as absent.
- **medium / defer** — Unconditional word boundaries cannot match names ending in non-word punctuation such as `C++` or `Yahoo!`.
- **maybe-false / defer** — Spark executors may not resolve the driver Workspace import used by `mapPartitions`; a Databricks executor run would settle whether Workspace files are distributed in this environment.
- **medium / defer** — Dataset guards check validation decision kinds but do not enforce per-label coverage in both training and validation splits.
- **high / defer** — Capped evaluation takes each kind's first rows after label-grouped construction, so the 83% publication gate can measure a one-class prefix.
- **high / defer** — Recovery packages the current prompt module rather than the qualifying run's version, permitting checkpoint/prompt drift.
- **high / defer** — Recovery pins the current cluster environment rather than the source run's recorded serving requirements, permitting runtime drift.
- **high / defer** — Recovery accepts an arbitrary qualifying run and publishes it to a hard-coded production model without validating the run's target provenance.
- **low / patch** — `databricks/README.md` still documented both Jev models as one provider; updated it to show explicit `typesafe` and `openrouter` routes.
