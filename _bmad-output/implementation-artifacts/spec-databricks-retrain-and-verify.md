---
title: 'Retrain and verify the Databricks Laya model'
type: 'bugfix'
created: '2026-09-21'
status: 'in-progress'
route: 'dispatch'
baseline_commit: 'da22d390cbdccc22187ff0d15e461235bd49b36c'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The deployed Databricks model uses final-epoch weights from a run whose reported 89.33% “overall” validation score sampled tags only. Serving also gates subjects at 0.50 while the benchmark backend uses 0.45, and the model environment emits a CloudPickle compatibility warning.

**Approach:** Harden the training and publication notebook, deploy it to the workspace, retrain on the existing four-GPU configuration, publish only a complete qualifying result, then prove the new version with live inference and a representative benchmark.

## Boundaries & Constraints

**Always:** Preserve the current uncommitted work as the baseline. Use `backend/config.py`’s normal `VALIDATION_THRESHOLD` of 0.45 in local and served model gating. Evaluate all four decision kinds (`validation`, `prominence`, `sentiment`, `tag`) with nonzero samples, restore the lowest-validation-loss epoch, retain the existing aggregate 83% registration threshold, and refuse registration when any expected kind is missing. Capture exact serving dependency versions from the training runtime, including CloudPickle. Keep version 2 available for rollback. Record Databricks run, model version, endpoint, and benchmark evidence.

**Never:** Do not change the training dataset, production audit data, prompt semantics, context-sensitive backend thresholds (0.35 lead and 0.65 absent), existing model versions, or unrelated user edits. Do not route endpoint traffic to a run that fails the registration gate. Do not report control-plane readiness as successful inference.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Qualifying training | All four kinds measured; aggregate accuracy ≥83% | Best epoch is packaged, registered as the next version, and deployed | Wait for endpoint readiness before inference |
| Incomplete evaluation | Any expected kind has zero samples or no metric | No model registration or endpoint mutation | Run fails with the missing kinds named |
| Low accuracy | Complete metrics but aggregate accuracy <83% | Artifacts and metrics remain inspectable; endpoint stays on version 2 | Exit without registration |
| Cold endpoint | First inference encounters scale-to-zero startup | Application retries within its configured policy | Failure remains explicit if retries exhaust |

</frozen-after-approval>

## Code Map

- `databricks/02_laya_gpu_training.py` — stratified evaluation, best-epoch restore, MLflow packaging, registration gate, and endpoint update.
- `databricks/03_recover_and_publish.py` — recovery-only publication path; keep threshold and dependency behavior consistent without using it to bypass training quality gates.
- `backend/config.py` — canonical normal validation threshold (`0.45`).
- `backend/laya_runner.py` — local two-stage gate currently hard-coded to `0.50`.
- `backend/laya_questions.py` — canonical prompts and 1024/256 token budgets packaged with the model.
- `backend/typesafe_runner.py` — live Databricks MLflow protocol, 120-second timeout, retry behavior, and response normalization.
- `backend/main.py` — representative benchmark API and contextual post-processing thresholds.
- `databricks/cluster_laya_gpu_4x.json` — four-GPU `g5.12xlarge` training cluster specification.
- `runs/batch_run_20260921_181814_095453_2_configs/` — prior Redwire/1Password benchmark baseline.

## Tasks & Acceptance

**Execution:**
- [ ] `databricks/02_laya_gpu_training.py`, `databricks/03_recover_and_publish.py`, `backend/laya_runner.py` — align normal validation gating at 0.45.
- [ ] `databricks/02_laya_gpu_training.py` — require complete per-kind evaluation and log non-ambiguous per-kind counts/metrics before registration.
- [ ] `databricks/02_laya_gpu_training.py`, `databricks/03_recover_and_publish.py` — pin the pyfunc environment to exact runtime versions, including CloudPickle.
- [ ] Databricks workspace — upload canonical questions and notebooks, submit a three-epoch/four-GPU retraining run, and publish its qualifying model version.
- [ ] Live endpoint and backend benchmark — prove one direct prediction and run the known Redwire/1Password 100-article benchmark.

**Acceptance Criteria:**
- Given a completed run, when MLflow metrics are inspected, then all four decision kinds have nonzero counts and individually named accuracy metrics.
- Given the best validation-loss epoch, when packaging completes, then the registered model uses those restored weights and exact serving dependency pins.
- Given a qualifying registered version, when endpoint deployment completes, then 100% traffic targets that version while version 2 remains registered.
- Given a live smoke request, when inference returns, then its prediction contains the expected model, answers, and usage envelope without a CloudPickle mismatch warning.
- Given the Redwire/1Password batch, when it completes, then 100 articles are evaluated with zero failed articles and all four comparison dimensions have nonzero totals.

## Implementation Notes
- Active run `293817740715256` exposed a dependency conflict: `transformers==5.17.0` was incompatible with the cluster's `mosaicml-streaming==0.12.0` and `sentence-transformers==4.0.1`, which both require Transformers 4.x.
- Pinned `transformers==4.57.6` in all three Databricks notebooks and the backend environment, regenerated the backend lockfile, and deployed byte-identical notebook sources to the workspace.
- Dependency resolution was verified with `laya==0.3.4`, `sentence-transformers==4.0.1`, and `mosaicml-streaming==0.12.0`. Retraining monitoring and restart were intentionally stopped at the user's request.
- The latest four-GPU run reached 80.83% overall and 67.33% sentiment accuracy; it did not clear the 83% registration gate, so model version 2 remains deployed.
- Genie profiling found sentiment labels are 74.92% neutral and 1.05% balanced in training (71.2:1 majority/minority). The training loss also treated `qtype == 0` as `noul`, but Laya maps `choice` to 0 and `noul` to 2, so `pos_weight` was incorrectly applied to prominence/sentiment label 1 rather than validation/tag positives.
- Notebook 02 now uses Laya's canonical `QTYPES["noul"]`, applies mean-one inverse-frequency sentiment weights with default power 0.5, persists a durable training summary and standalone checkpoint artifacts, and packages the prompt module from local disk. Notebook 03 can recover from the standalone checkpoint artifact before falling back to a pyfunc artifact.

## Spec Change Log
- 2026-09-22: User requested a sentiment-focused retrain while retaining the existing 83% publication gate and version 2 rollback.

## Review Triage Log

## Design Notes

The registration gate protects deployment, not experimentation: failed runs keep MLflow evidence but cannot mutate the endpoint. Exact dependency pins are generated from the runtime that serializes the pyfunc rather than guessed from local lower bounds.

## Verification

**Commands:**
- `uv run ruff check backend databricks` — changed Python passes repository lint rules.
- `uv run pyright` from `backend/` — backend changes remain type-correct.
- Databricks run inspection — successful lifecycle, complete per-kind metrics, best epoch, and registered version are observable.
- Direct `serving-endpoints query` — a real prediction returns from the newly routed version.
- `POST /api/benchmark/batch-run` for `redwire` and `1password`, followed by job polling — 100 articles, zero failures, nonzero totals for all four dimensions.
