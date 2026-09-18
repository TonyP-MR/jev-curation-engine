# Processed Article Audit Archive: Location, Access, and JSON Schema

This guide explains how another application can read the Curation Engine's
processed-article audit files from Azure Blob Storage. It covers the normal
classification audit written after a successful pipeline run. Delivery-audit
files and validation-failure files can share the storage container, but they
have different paths and schemas.

The deployment values and a production blob sample were verified on
2026-09-18. Treat the JSON as a versioned integration contract: accept missing
and additional fields so that the consumer remains compatible with older and
newer blobs.

## Important distinction: the archive is not the Raw JSON tab

The Configurator's **Raw JSON** tab displays the response from the HTTP test
endpoint, after frontend normalization. The blob archive contains a separate,
flat audit snapshot assembled by the pipeline.

The two representations overlap, but they are not identical. In particular,
the `diagnostics.rules_debug` object shown in the screenshot is returned by the
test endpoint but is not copied into the current audit blob. Do not build an
archive reader that expects the Raw JSON tab's wrapper or `rules_debug` fields.
The screenshot-specific fields are described near the end of this document.

## Storage locations

| Environment | Account URL | Container | Normal blob name |
|---|---|---|---|
| Production | `https://stcurationauditprod.blob.core.windows.net` | `pipeline-audit` | `{correlation_id}_production.json` |
| Staging | `https://stcurationauditstg.blob.core.windows.net` | `pipeline-audit` | `{correlation_id}_staging.json` |

For example, production correlation ID
`123e4567-e89b-12d3-a456-426614174000` is stored as:

```text
https://stcurationauditprod.blob.core.windows.net/pipeline-audit/123e4567-e89b-12d3-a456-426614174000_production.json
```

The URL is not anonymously readable. The storage accounts have blob public
access disabled and require Microsoft Entra authentication. The endpoint is
reachable over HTTPS, subject to the storage account's network rules.

### What is archived

A normal audit blob is queued after an article has completed processing. Both
Kafka processing (`processing_source = "kafka"`) and HTTP test processing
(`processing_source = "http"`) can create one.

The normal blob contains the article identifiers and metadata, full inbound
article body, overall classification, per-subject and per-tag results, rule
trace, LLM response and settings, token information, and stage timings.

The audit write is deliberately fire-and-forget: an audit-storage failure is
logged but does not fail article processing. Therefore, the container is a
diagnostic audit archive, not a guaranteed system of record for every inbound
article. Articles rejected before successful pipeline completion, deduplicated
before processing, or affected by an audit-write failure may not have a normal
blob.

The current storage accounts have no Azure lifecycle management policy. This
means there is no configured automatic age-based deletion rule, but it is not a
contractual retention guarantee. Confirm retention requirements with the
storage/infrastructure owner before relying on the archive for compliance or
long-term preservation.

### Other objects in the same container

The container can also contain:

- `validation-failure/...` objects for inbound payloads that failed schema
  validation;
- delivery-audit objects written by CE-Cov-Rep-Push, whose paths are recorded
  in `cfg_delivery_audit_batches.blob_path`.

These are not processed-article classification blobs. A consumer should fetch
the exact normal filename from a correlation ID rather than treating every
`.json` object in the container as the schema documented here.

## Authentication and authorization

Use `DefaultAzureCredential` from `azure-identity`. This is the same approach
used by the Curation Engine and Configurator.

For an Azure-hosted application, the recommended setup is:

1. Enable a system-assigned or user-assigned managed identity on the app.
2. Grant that identity **Storage Blob Data Reader** on each storage account it
   needs to read. Production and staging are separate role assignments.
3. Configure the account URL, container, and suffix as environment variables.
4. Let `DefaultAzureCredential` obtain the managed-identity token. Do not put a
   storage account key or SAS token in source code.

`Storage Blob Data Reader` is sufficient for listing and downloading blobs.
Azure management-plane roles such as `Reader` or `Contributor` do not, by
themselves, grant blob data access.

An administrator can grant the production role with the following pattern:

```bash
ACCOUNT_SCOPE=$(az storage account show \
  --name stcurationauditprod \
  --resource-group rg-curation-prod \
  --query id \
  --output tsv)

az role assignment create \
  --assignee-object-id "<application-managed-identity-principal-id>" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Reader" \
  --scope "$ACCOUNT_SCOPE"
```

For local development, run `az login`. `DefaultAzureCredential` will use the
Azure CLI identity when no application or managed-identity credential is
available. The signed-in developer still needs **Storage Blob Data Reader** on
the target account. Role assignments can take several minutes to propagate.

Microsoft's current Python guidance also recommends passwordless
`DefaultAzureCredential` authentication:

- [Get started with Azure Blob Storage and Python](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-python-get-started)
- [Assign an Azure role for Blob data access](https://learn.microsoft.com/en-us/azure/storage/blobs/assign-azure-role-data-access)

## Python integration

Install the supported Azure client libraries:

```bash
python -m pip install azure-identity azure-storage-blob
```

Configure one environment at a time. These values are identifiers, not
credentials:

```bash
# Production
export AUDIT_BLOB_ACCOUNT_URL="https://stcurationauditprod.blob.core.windows.net"
export AUDIT_BLOB_CONTAINER="pipeline-audit"
export AUDIT_BLOB_SUFFIX="_production"
```

Use the correlation ID to construct the exact blob name:

```python
from __future__ import annotations

import json
import os
from typing import Any

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_URL = os.environ["AUDIT_BLOB_ACCOUNT_URL"]
CONTAINER_NAME = os.getenv("AUDIT_BLOB_CONTAINER", "pipeline-audit")
ENVIRONMENT_SUFFIX = os.environ["AUDIT_BLOB_SUFFIX"]

credential = DefaultAzureCredential()
service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
container = service.get_container_client(CONTAINER_NAME)


def fetch_processed_article(correlation_id: str) -> dict[str, Any]:
    """Download and decode one processed-article audit blob."""
    if not correlation_id or "/" in correlation_id:
        raise ValueError("correlation_id must be a non-empty ID, not a path")

    blob_name = f"{correlation_id}{ENVIRONMENT_SUFFIX}.json"
    try:
        raw = container.get_blob_client(blob_name).download_blob().readall()
    except ResourceNotFoundError as exc:
        raise LookupError(f"No audit blob exists at {blob_name}") from exc
    except HttpResponseError as exc:
        raise RuntimeError(
            f"Azure Blob request failed for {blob_name}: HTTP {exc.status_code}"
        ) from exc

    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON root in {blob_name}; expected an object")
    return payload
```

Example use:

```python
audit = fetch_processed_article("123e4567-e89b-12d3-a456-426614174000")

print(audit["article_id"])
print(audit["validation_result"])

for subject in audit.get("subject_results", []):
    print(subject["subject_id"], subject["is_valid"], subject["sentiment"])
```

Use `get()` and sensible defaults for optional fields. Do not log the complete
payload: it can contain full licensed article text, prompts, and model output.

### Fetching more than one known correlation ID

When the application already has correlation IDs, construct each filename and
download it directly. Bounded concurrency is preferable to listing and scanning
the whole container.

Blob names are not organized by article ID or date and no blob index tags are
written. The archive therefore does not support an efficient server-side query
such as "find the file for this article ID." The consuming application should
store the correlation ID when it first receives or records the processing
result. If it only has an article ID, it needs another index or an expensive
container scan that downloads JSON documents to inspect their contents.

To enumerate a known correlation-ID prefix:

```python
def matching_blob_names(correlation_id_prefix: str) -> list[str]:
    return [
        item.name
        for item in container.list_blobs(name_starts_with=correlation_id_prefix)
    ]
```

Listing also requires the **Storage Blob Data Reader** role. Azure paginates the
results internally; do not assume one response contains the entire archive.

## Archived JSON schema

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `correlation_id` | string | Unique identifier for this processing run. It is the lookup key used in the blob filename. |
| `tracker_id` | string | Muck Rack tracker that supplied or owns the article. |
| `config_id` | string | Curation Engine configuration selected for this run. |
| `config_name` | string | Human-readable configuration name at processing time. |
| `article_id` | string | Upstream article identifier. It is not part of the normal blob name. |
| `headline` | string | Article headline used by the pipeline. |
| `source` | string | Article publication/source. This is different from `processing_source`. |
| `country` | string or null | Article/source country supplied upstream. |
| `language` | string or null | Article language supplied upstream. |
| `published_at` | ISO-8601 string | Article publication time. Do not use this as the archive-write time. |
| `media_type` | string or null | Upstream media category, such as online, print, broadcast, or another source-defined value. |
| `validation_result` | boolean | Overall LLM article validation decision: `true` means accepted and `false` means rejected. |
| `rejection_reason` | string or null | Overall explanation when `validation_result` is `false`; normally null for accepted articles. |
| `summary` | string or null | LLM-produced article summary. It can be null when no summary was produced or licensing/settings prevented one. |
| `scored_attributes` | object | Final assembled subjects and grouped tags, including rule-driven overrides. See below. |
| `boolean_results` | object | Map of boolean tag/expression ID to its raw keyword-expression result, before rules mutate active tags. |
| `rules_trace` | array | Rule-evaluation trace. Entries may have no actions if a rule was evaluated but did not fire. |
| `subject_results` | array | Canonical per-subject results, including validity, prominence, sentiment, active tag IDs, and optional reasoning. |
| `llm_provider` | string | Provider that produced the classification, for example `google` or `openai`. |
| `llm_model` | string or null | Exact model identifier reported by the pipeline. |
| `llm_tokens` | object | Input/output token counts for the LLM call. |
| `llm_cost_usd` | number or null | Recorded call cost in US dollars when available. Null means cost was not populated, not necessarily that the call was free. |
| `llm_raw_response` | string or null | Raw model response. It is often a string containing JSON, so parsing it can require a second `json.loads`; its internal shape is provider/schema-version dependent. |
| `llm_cached_tokens` | integer | Number of provider-reported cached input tokens. Usually zero when caching was not used or not reported. |
| `llm_generation_config` | object | Provider request settings used for the call. This is diagnostic and can evolve. |
| `stage_timings` | array | Timing records for the pipeline stages that executed. |
| `processing_source` | string | Entry path for this run: currently `kafka` or `http`. |
| `processed_at` | ISO-8601 string | UTC time at which the audit payload was assembled. Use this, not `published_at`, for processing chronology. |
| `inbound_data` | object, optional | Raw inbound article/request fields supplied to the pipeline. Older or exceptional blobs may omit it. |

`subject_results` is the best source for a consuming application's per-subject
classification. `scored_attributes.tag_groups` is the best source for final
grouped tag output. `boolean_results` is diagnostic evidence for the original
boolean expressions, not the final active-tag list.

### `subject_results[]`

| Field | Type | Meaning |
|---|---|---|
| `subject_id` | string | Stable subject identifier from the configuration. |
| `name` | string | Subject name captured at processing time. |
| `is_valid` | boolean | Final validity of this subject after applicable rule overrides. |
| `invalid_reason` | string or null | Explanation for an invalid subject. |
| `is_default` | boolean | Whether this is the configuration's default/fallback subject. |
| `prominence` | string | Controlled value: `primary`, `significant`, or `passing`. |
| `sentiment` | string | Controlled value: `positive`, `negative`, `neutral`, or `balanced`. |
| `tags` | array of strings | Active tag IDs assigned to this subject after mapping and rule effects. These are IDs, not tag names. |
| `validation_reasoning` | string or null | Diagnostic rationale for the subject validity decision when extended diagnostics were requested. |
| `prominence_reasoning` | string or null | Diagnostic rationale for prominence when available. |
| `sentiment_reasoning` | string or null | Diagnostic rationale for sentiment when available. |

The reasoning fields can legitimately be null. Their presence depends on the
request's `include_diagnostics` setting and the model response schema used at
the time.

### `scored_attributes`

| Field | Type | Meaning |
|---|---|---|
| `subjects` | array | Compact final subject results used for assembled output. |
| `tag_groups` | array | Tag results grouped according to the configuration. |
| `subjects_processed` | integer | Number of subject entries assembled. |
| `total_attributes` | integer | Count of assembled subjects plus tags. It is not a confidence score. |

Each `scored_attributes.subjects[]` object contains:

| Field | Type | Meaning |
|---|---|---|
| `subject_id` | string | Subject ID. |
| `name` | string | Subject name. |
| `is_valid` | boolean | Final subject validity. |
| `prominence` | string | Final prominence after rule overrides. |
| `sentiment` | string | Final sentiment after rule overrides. |

Each `scored_attributes.tag_groups[]` object contains:

| Field | Type | Meaning |
|---|---|---|
| `group_id` | string | Configured tag-group ID. |
| `name` | string | Configured tag-group name. |
| `tags` | array | Tags in this group and their final results. |

Each `scored_attributes.tag_groups[].tags[]` object contains:

| Field | Type | Meaning |
|---|---|---|
| `tag_id` | string | Configured tag ID. |
| `name` | string | Configured tag name. |
| `evaluation_type` | string | How the base result was produced, normally boolean expression or LLM evaluation. |
| `result` | boolean | Final tag state after rule add/remove actions. |

### `boolean_results`

This is a dynamic object whose keys are configured tag or boolean-expression
IDs and whose values are booleans:

```json
{
  "uD": false,
  "uE": true
}
```

`true` means the configured boolean expression matched the article text. It
does not necessarily mean the tag remains active after rules run. Resolve IDs
to display names from the configuration version that was active when the blob
was written.

### `rules_trace[]`

| Field | Type | Meaning |
|---|---|---|
| `rule_id` | string | Identifier of the evaluated configured rule. |
| `actions_applied` | array of strings | Serialized actions that the rule applied. An empty list means no action was applied. |

### LLM fields

`llm_tokens` contains:

| Field | Type | Meaning |
|---|---|---|
| `input` | integer | Input/prompt tokens reported for the call. |
| `output` | integer | Generated/output tokens reported for the call. |

`llm_generation_config` is provider-dependent. Current Google-backed blobs can
include:

| Field | Type | Meaning |
|---|---|---|
| `temperature` | number | Sampling temperature sent to the provider. |
| `response_mime_type` | string | Requested model response media type, normally JSON. |
| `response_schema` | object | JSON schema given to the provider for structured output. Treat its nested fields as diagnostic provider configuration, not archive fields to map into the consuming application's domain model. |

Other providers or future versions may add or omit generation settings.

### `stage_timings[]`

| Field | Type | Meaning |
|---|---|---|
| `stage` | string | Pipeline stage name. |
| `duration_ms` | number | Stage duration in milliseconds. |
| `entered_at` | ISO-8601 string | Time the stage started. |
| `exited_at` | ISO-8601 string | Time the stage completed. |

The array only contains stages that executed. Do not assume a fixed number or
order of stages across pipeline versions.

### `inbound_data`

This object preserves the request/article fields received by the pipeline. It
can contain:

| Field | Type | Meaning |
|---|---|---|
| `article_id` | string | Original article ID. |
| `tracker_id` | string | Original tracker ID. |
| `config_id` | string or null | Explicit configuration ID, if supplied. |
| `headline` | string | Original headline. |
| `body` | string | Full article body used for evaluation. This can be large and can contain licensed or sensitive content. |
| `source` | string | Original publication/source. |
| `country` | string or null | Original country value. |
| `language` | string or null | Original language value. |
| `media_type` | string or null | Original media type. |
| `published_at` | ISO-8601 string | Original publication time. |
| `message_type` | string | Inbound message discriminator, normally `article`. |
| `origin` | string | Upstream processing origin, or `unknown` when none was supplied. |
| `metadata` | object | Flat upstream metadata. Values are scalars or lists of scalars. |
| `force_reprocess` | boolean | Whether this run was allowed to bypass normal deduplication. |
| `include_diagnostics` | boolean | Whether the caller requested extended diagnostic/rationale output. |
| `muckrack_link_id_tiny` | string or null | Muck Rack short/tiny link identifier when available. |
| `muckrack_link_id` | integer or null | Numeric Muck Rack link identifier when available. |

Known `inbound_data.metadata` keys include the following, but metadata is
source-dependent and is not a closed schema:

| Field | Typical type | Meaning |
|---|---|---|
| `url` | string | Article URL. |
| `authors` | array of strings | Upstream author names. |
| `snippet` | string | Upstream article excerpt. |
| `key_terms` | array of strings | Upstream-extracted key terms. |
| `audience` | number | Upstream audience/reach value. |
| `domain_authority` | number | Upstream domain-authority metric. |
| `scope` | string or array | Upstream scope/category information. |
| `sentiment` | string | Upstream metadata sentiment; do not confuse it with the Curation Engine's per-subject `sentiment`. |
| `clip_id` | string or number | Upstream clip identifier. |
| `has_publisher_restriction` | boolean | Whether publisher restrictions were reported upstream. |

## Fields shown in the screenshot's `rules_debug` block

These fields belong to the test endpoint's diagnostics response, not the
current archived blob schema.

### `parsed_rules[]`

The configuration rules parsed and sorted before evaluation. An empty array
means the configuration had no rules to evaluate.

| Field | Type | Meaning |
|---|---|---|
| `rule_id` | string | Configured rule ID. |
| `order` | integer | Evaluation order; lower-numbered rules run first. |
| `conditions` | array | Conditions that must match. |
| `conditions[].field` | string | State field examined by the condition. |
| `conditions[].operator` | string | Comparison operator. |
| `conditions[].value` | any | Configured comparison value. |
| `actions` | array | Actions to apply if the rule matches. |
| `actions[].action_type` | string | Action kind, such as adding/removing a tag or overriding a subject attribute. |
| `actions[].value` | any | Action target/value. |

### `initial_article_state`

This is a snapshot immediately before rules are applied:

| Field | Type | Meaning |
|---|---|---|
| `active_tags` | array of strings | Tag IDs initially active from boolean and LLM tag evaluation. |
| `boolean_results` | object | Boolean-expression results keyed by tag ID. |
| `subject_sentiments` | object | Initial LLM sentiment keyed by subject ID. |
| `subject_prominences` | object | Initial LLM prominence keyed by subject ID. |
| `subject_validations` | object | Initial LLM subject-validity boolean keyed by subject ID. This field is present in the screenshot/newer rules-engine response but may be absent from older responses. |
| `validation_result` | boolean | Overall validation result before rules execute. |

For example, keys such as `"106"` or `"uE"` are configuration IDs, not
human-readable names. The consuming application needs the applicable
configuration snapshot to resolve them.

## Consumer design recommendations

- Keep production and staging configuration separate; the account and suffix
  must always match.
- Request only **Storage Blob Data Reader** unless the application genuinely
  needs to modify the archive.
- Use correlation IDs as the application-level foreign key to blobs.
- Parse timestamps as timezone-aware ISO-8601 values.
- Treat nullable fields as normal and ignore unknown fields for forward
  compatibility.
- Prefer `subject_results` and `scored_attributes.tag_groups` for business
  output; treat raw LLM and debug fields as diagnostics.
- Avoid logging `inbound_data.body`, prompts, raw LLM responses, credentials, or
  whole blobs.
- Add retry with bounded exponential backoff for transient Azure errors, but do
  not retry `404` indefinitely: it can indicate that no audit blob was written.
- If durable, queryable history is required, ingest selected blob fields into a
  purpose-built database/index instead of repeatedly scanning the container.

## Implementation sources

The archive writer and schema are implemented in the Curation Engine pipeline:

- `curation-engine-pipeline/src/curation_engine/adapters/blob/audit_store.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/audit.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/article.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/diagnostics.py`
- `curation-engine-pipeline/src/curation_engine/domain/stages/scored_attributes.py`
- `curation-engine-pipeline/src/curation_engine/domain/stages/rules_engine.py`

The Configurator's existing reader is a useful reference implementation:

- `backend/app/noise_advisor/blob_client.py`

For the broader system data-store map, see
[`data-artefacts.md`](./data-artefacts.md).
