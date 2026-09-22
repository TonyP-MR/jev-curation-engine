- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Make absent-entity validation use one effective threshold through gating, post-processing, and comparison.
  evidence: The existing uncommitted benchmark changes gate scores below 0.65 for unmatched subjects, but later classify the same scores with the global 0.45 threshold; this is real but unrelated to restoring direct Jev routing.
- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Restore configured aliases, short distinctive names, and punctuation-safe boundaries in subject matching.
  evidence: The existing uncommitted matcher replaced criteria and tag-derived aliases with label-only terms, drops three-character tokens such as Kia, and wraps punctuation-ending names in incompatible word boundaries; these regressions predate and are unrelated to the provider change.
- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Verify and harden distribution of laya_questions to Spark executors.
  evidence: The prep notebook imports a driver Workspace file whose functions run in mapPartitions; whether that path is available to executors requires a Databricks run, and failure would prevent sequence generation. This is unrelated to direct Jev routing.
- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Enforce per-label train and validation coverage and label-stratified capped evaluation.
  evidence: The training notebook checks validation decision kinds only and takes each kind's first rows despite upstream label-grouped unions, so class coverage and the publication metric can depend on row order. This is unrelated to direct Jev routing.
- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Recover prompt code and serving dependencies from the qualifying MLflow source run.
  evidence: The recovery notebook packages the current neighboring prompt module and current cluster package versions rather than the source run's recorded artifacts, allowing train/serve drift. This is unrelated to direct Jev routing.
- source_spec: `_bmad-output/implementation-artifacts/spec-re-enable-direct-jev-access-option.md`
  summary: Bind checkpoint recovery to the source run's recorded model publication target.
  evidence: The recovery notebook accepts an arbitrary qualifying run but publishes to a hard-coded Unity Catalog model without validating source-run catalog, schema, table, or target parameters. This is unrelated to direct Jev routing.
