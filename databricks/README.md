# Laya on Databricks

Moves Laya training off the single Azure Tesla T4 VM (`dev-002`) and onto the
`muckrack-data` workspace.

## Contents

| File | What it does |
|---|---|
| `01_laya_sequence_prep.py` | Ingests audit blobs and config snapshots, builds tokenized typed-decision sequences with Spark, rebalances the tag class, writes `muckrack_data.laya.training_sequences` |
| `02_laya_gpu_training.py` | Fine-tunes ModernBERT-large on those sequences, logs to MLflow, registers the weights in Unity Catalog |
| `cluster_laya_gpu.json` | Single-node `g5.xlarge` (A10G, 24 GB) GPU cluster spec |
| `ruff.toml` | Silences F821 for the Databricks runtime globals |

## Workspace

- Host: `https://dbc-34034be3-652f.cloud.databricks.com`
- Account: `tony.prime@muckrack.com`
- Catalog: `muckrack_data`, schema `laya` (created by notebook 01)
- Notebooks: `/Users/tony.prime@muckrack.com/laya_curation_engine/`
- Cluster: `laya-gpu-training` (`0921-091413-skcrybyo`)

## CLI setup

The CLI is installed at `~/.local/bin/databricks`. OAuth credentials live in the
macOS keychain and refresh on their own, but the CLI only finds them when the
host is set:

```bash
export DATABRICKS_HOST=https://dbc-34034be3-652f.cloud.databricks.com
```

That line is in `~/.zshrc`. Re-authenticate with:

```bash
databricks auth login --host $DATABRICKS_HOST
```

Do not write a bare token into `~/.databrickscfg`. It expires after an hour and
the next command fails with a stale-credential error.

## Getting the source data across

The audit blobs sit in Azure Blob Storage and this workspace runs on AWS, so
notebook 01 needs a credential. Register the storage key once:

```bash
databricks secrets create-scope laya
databricks secrets put-secret laya azure_storage_key
```

The alternative is staging the blobs into a Unity Catalog volume and setting the
`source_mode` widget to `volume`. That avoids cross-cloud egress on every run and
is the better choice once the pipeline is scheduled.

## Running it

1. Attach `01_laya_sequence_prep` to any CPU cluster. Set `blobs_limit` to a few
   hundred for a first pass, then 0 for everything.
2. Attach `02_laya_gpu_training` to `laya-gpu-training` and run. Defaults are 3
   epochs, top 8 encoder layers, positive class weight 2.5.
3. Open the MLflow experiment at `/Users/tony.prime@muckrack.com/laya_training`
   to compare against earlier runs.

## Why the settings differ from the Azure run

The model trained on the Azure VM hedged on tags. True tags came back at p=0.30
to 0.50 and never cleared the decision threshold, which is why the rig showed
almost every tag as false. Three causes, each addressed here:

- Tags were sampled at the production ratio of roughly 85% negative, so the model
  learned the base rate instead of the criteria. Notebook 01 downsamples the
  negatives to the `tag_pos_ratio` widget, default 50/50.
- Only one epoch ran, 1,098 optimizer steps across 274 client configurations.
  Notebook 02 runs three.
- Loss treated a missed positive and a missed negative the same. Notebook 02
  weights the positive class on `noul` heads by `pos_weight`.

The train/validation split is hashed on `correlation_id` rather than sampled per
row. The Azure split put sequences from the same article on both sides, so the
reported 83.33% included memorized article text.

## Using the trained model in the test rig

Notebook 02 logs an MLflow `pyfunc`, not just the raw weights. A bare artifact
directory registers fine but cannot be invoked, so the rig would have nothing to
call. The wrapper reproduces the two-stage gating in `backend/laya_runner.py`:
validation questions run first, and prominence, sentiment and tag questions are
only asked about subjects that passed. Subjects that failed get the same
hard-coded defaults the local runner applies, so a served answer and a local
answer agree for the same input.

The response envelope matches the Azure service, which is what lets the rig
treat both as interchangeable engines.

Two ways to run it, both in the **Decision Engine** dropdown:

### Databricks Model Serving

Select **Laya: Databricks Model Serving (GPU endpoint)**. Notebook 02 creates or
updates the endpoint named by the `serving_endpoint` widget after a run clears
the accuracy gate. The rig needs three values in the repo-root `.env`:

```
DATABRICKS_HOST=https://dbc-34034be3-652f.cloud.databricks.com
DATABRICKS_TOKEN=<PAT or service-principal token>
LAYA_DATABRICKS_ENDPOINT=laya-curation-engine
```

`GET /api/providers` reports `configured: true` once they resolve. The endpoint
is created with `scale_to_zero_enabled`, so an idle endpoint bills nothing and
the first request after idle pays a cold start.

### Local weights

Select **Laya: Databricks Weights (local MPS)** after pulling the weights down:

```bash
uv run python scripts/fetch_databricks_model.py            # latest version
uv run python scripts/fetch_databricks_model.py --version 4
```

That writes `runs/laya_databricks/`, which `backend/laya_runner.py` resolves by
name. Cheaper than a warm GPU endpoint and one less network hop per decision, so
it is the better option for iterating on thresholds against a fresh model.

### How a selection is routed

The model string identifies the engine and model. The app also sends an explicit
`provider` so direct TypeSafe and OpenRouter Jev requests use separate credentials
and endpoints. Order matters: `laya:databricks:local` is checked before
`laya:databricks` because one is a prefix of the other.

| Dropdown value | Provider / engine | Where it runs |
|---|---|---|
| `laya:azure:t4` | `laya_azure` | `dev-002` in uksouth |
| `laya:databricks` | `laya_databricks` | Databricks Model Serving |
| `laya:databricks:local` | `laya_local` | `runs/laya_databricks` on local MPS |
| `jev-latest` | `typesafe` | TypeSafe direct cloud API |
| `typesafe/jev-1.13` | `openrouter` | OpenRouter proxy cloud API |

## Keeping the prompts in sync

Notebook 01 inlines the question-building logic from
`backend/question_converter.py` so the cluster does not need the repo. When the
backend prompts change, either update the notebook or package `backend/` as a
wheel and import it instead. The wheel is the better long-term answer.
