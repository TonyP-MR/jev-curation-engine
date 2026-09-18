# Curation Pipeline Prompts and Configuration: Test-Rig Integration Guide

This document is the implementation contract for an application that needs to
reconstruct the Curation Engine classifier prompt, inspect the configuration
that produced it, and interpret the resulting subject, sentiment, prominence,
tag, and summary fields.

It is written for both engineers and an LLM implementing a test rig. Follow the
`Production-faithful` path unless the rig is explicitly testing unpublished
draft edits.

Deployment settings and database infrastructure were verified on 2026-09-18.
Prompt behavior is derived from the current pipeline source. Keep the rig
versioned against that source because prompt text, structured-output schemas,
and configuration mapping can evolve.

> Security boundary: this document names servers, databases, secret references,
> and approved retrieval procedures. It deliberately contains no passwords,
> API keys, bearer tokens, or other credential values. Never paste a database
> password, full connection string, article body, or raw prompt into an LLM
> conversation, source file, ticket, or log.

## Executive summary for the test-rig builder

1. Read the configuration from Azure Database for MySQL Flexible Server.
2. For production parity, load the latest `cfg_version_snapshots.snapshot_data`
   for the active configuration. Do not rebuild a production config from the
   mutable live tables.
3. Build the system prompt in this order:
   base instructions, media policy, validation rules, optional summary
   instructions, optional rejection-reason instructions, every subject, then
   only the LLM-evaluated tags.
4. Build the user prompt from the article's media type, headline, and full body.
5. Enforce the same structured response schema as the pipeline. The schema is
   part of the contract; the prompt alone is not sufficient.
6. Evaluate boolean tags locally against `headline + " " + body`. Boolean tags
   are never sent to the LLM.
7. Apply post-processing rules after both the LLM and boolean evaluations.
8. Compare final `subject_results` and `scored_attributes.tag_groups` when
   testing end-to-end pipeline behavior. Raw LLM fields are pre-rule results.

## Deployed database

The shared configuration database is Azure Database for MySQL Flexible Server
8.0.21. Both environments use TCP port `3306`, database `curation_engine`, and
TLS.

| Environment | MySQL server/FQDN | Resource group | Database |
|---|---|---|---|
| Production | `mysql-curation-production.mysql.database.azure.com` | `rg-curation-configurator-prod` | `curation_engine` |
| Staging | `mysql-curation-staging.mysql.database.azure.com` | `rg-curation-configurator-staging` | `curation_engine` |

The servers currently use public endpoints protected by MySQL authentication
and Azure firewall rules. An application outside the already allowed Azure
network path needs a narrowly scoped firewall rule for its fixed outbound IP.
Do not add a public all-addresses rule.

The deployed pipeline currently has these non-secret connection settings:

| Setting | Production and staging value |
|---|---|
| Host | Environment-specific FQDN from the table above |
| Port | `3306` |
| Database | `curation_engine` |
| TLS | Enabled |
| Pipeline login | `caboradmin` |
| Password reference | Container App secret `mysql-password` |

`caboradmin` is the server administrator used by the existing service. It is
documented here to explain the deployment, not as the login that a test rig
should use. A new application should receive a separate read-only account.

## Obtaining database access safely

### Recommended: create a dedicated read-only login

Ask the database/infrastructure owner to create a login such as
`ce_test_rig_reader` with `REQUIRE SSL` and only `SELECT` access to the tables
the rig needs.

For production-published configurations, the minimum tables are:

- `configurations`
- `tracker_mappings`
- `cfg_version_snapshots`

If the rig must also reproduce the Configurator's unpublished draft test path,
grant read access to:

- `subjects`
- `tag_evaluations`
- `cfg_tag_subject_evaluations`
- `rules`
- `summary_prompts`

An authorized MySQL administrator can use this pattern:

~~~sql
CREATE USER 'ce_test_rig_reader'@'%'
IDENTIFIED BY '<generate-and-store-a-strong-password>'
REQUIRE SSL;

GRANT SELECT ON curation_engine.configurations
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.tracker_mappings
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.cfg_version_snapshots
TO 'ce_test_rig_reader'@'%';

-- Add these only if draft/live-table testing is required.
GRANT SELECT ON curation_engine.subjects
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.tag_evaluations
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.cfg_tag_subject_evaluations
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.rules
TO 'ce_test_rig_reader'@'%';

GRANT SELECT ON curation_engine.summary_prompts
TO 'ce_test_rig_reader'@'%';
~~~

Generate the password in the organization's approved password/secret manager.
Store it directly in the consuming application's secret store. Supply it to
the process as an environment variable or mounted secret; do not commit it to
an `.env` file.

The human access request should specify:

- application and owner;
- staging, production, or both;
- read-only purpose;
- required tables;
- runtime location and fixed outbound IP or Azure identity;
- intended lifetime and rotation owner;
- confirmation that the credential will not be sent to an LLM.

### Where the existing pipeline password comes from

The current deployment path is:

1. GitHub Actions environment/repository secret:
   - staging: `STG_MYSQL_PASSWORD`
   - production: `PROD_MYSQL_PASSWORD`
2. The deployment workflow passes that value to the pipeline Bicep parameter
   `mysqlPassword`.
3. Bicep stores it as the Azure Container App secret `mysql-password`.
4. Container environment variable `CE_MYSQL_PASSWORD` references that secret.

GitHub does not allow an existing Actions secret value to be read back. An
authorized operator can replace it, but cannot reveal it through the GitHub UI
or API.

The Container App secret is stored directly in Container Apps rather than as a
Key Vault reference. An operator with the
`Microsoft.App/containerApps/listSecrets/action` permission can reveal it with
Azure CLI. This is a break-glass/diagnostic route, not the preferred way to
provision the test rig:

~~~bash
# Production: prints a secret value to the current private terminal.
az containerapp secret list \
  --name ca-curation-prod \
  --resource-group rg-curation-prod \
  --show-values \
  --query "[?name=='mysql-password'].value | [0]" \
  --output tsv

# Staging equivalent.
az containerapp secret list \
  --name ca-curation-stg \
  --resource-group rg-curation-stg \
  --show-values \
  --query "[?name=='mysql-password'].value | [0]" \
  --output tsv
~~~

Do not run those commands through an LLM agent, CI log, shared terminal
recording, or shell command that echoes/captures the value into history. The
existing value belongs to an administrator login and should not be copied into
a new long-lived application. Provision the dedicated reader instead.

### Network access

The current servers have public network access enabled and use firewall rules.
Authentication alone is insufficient if the caller's network path is not
allowed.

For an external fixed-IP runner, an authorized Azure operator can add exactly
that outbound IP:

~~~bash
az mysql flexible-server firewall-rule create \
  --resource-group rg-curation-configurator-staging \
  --name mysql-curation-staging \
  --rule-name ce-test-rig-reader \
  --start-ip-address "<fixed-outbound-ip>" \
  --end-ip-address "<fixed-outbound-ip>"
~~~

Use the production resource group/server only after the staging rig works.
Prefer private networking for a permanent production integration. Microsoft's
current guidance is:

- [MySQL Flexible Server public networking](https://learn.microsoft.com/en-us/azure/mysql/flexible-server/concepts-networking-public)
- [MySQL Flexible Server TLS connections](https://learn.microsoft.com/en-us/azure/mysql/flexible-server/security-tls-how-to-connect)
- [Azure Container Apps secret commands](https://learn.microsoft.com/en-us/cli/azure/containerapp/secret)

## Python database connection

Install a MySQL driver:

~~~bash
python -m pip install pymysql
~~~

Supply configuration through secrets/environment:

~~~text
MYSQL_HOST=mysql-curation-staging.mysql.database.azure.com
MYSQL_PORT=3306
MYSQL_DATABASE=curation_engine
MYSQL_USER=ce_test_rig_reader
MYSQL_PASSWORD=<in the application secret store, never in this document>
MYSQL_SSL_CA=/absolute/path/to/DigiCertGlobalRootG2.crt.pem
~~~

Use the CA certificate recommended by Azure and verify TLS:

~~~python
from __future__ import annotations

import os

import pymysql
from pymysql.cursors import DictCursor


def connect_read_only() -> pymysql.Connection:
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "curation_engine"),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
        ssl={"ca": os.environ["MYSQL_SSL_CA"]},
    )
~~~

The read-only database grants are the real safety boundary. Setting a client
session to read-only is useful defense in depth but is not a substitute for
those grants.

## Which configuration production actually uses

### Normal Kafka/production path

The normal pipeline resolves an active configuration from the immutable
`cfg_version_snapshots` table:

- If the article has a normal tracker ID, resolve through `tracker_mappings`.
- A configuration is operationally active when:
  - `status = 'active'`; or
  - `status = 'pending_changes'` and its previous status was active.
- Archived configurations are excluded.
- Select the highest `version_number`.
- The complete frozen configuration is in `snapshot_data`.
- Published configs are cached in process. Cache invalidation is explicit, not
  time-based.

For normal tracker-ID and federated lookups, this keeps unpublished edits out
of queue processing.

#### Cache-warmup caveat

There is one implementation caveat a parity rig must record. At process
startup, `ConfigCache.warmup()` calls `list_active_configs()`, which currently
assembles active configurations from the live tables and caches them by
configuration identifier. Normal tracker-ID lookups do not normally hit those
identifier keys and then load the published snapshot on demand. However, a
queue item resolved directly by configuration identifier can hit the warmed
live-table object.

The published snapshot is still the correct default for reproducible
production tests and the intended production contract. If investigating an
individual historical discrepancy, also capture the actual system prompt from
the run's diagnostics/audit evidence and record the running pipeline revision;
do not assume the database snapshot alone proves which in-memory cache object a
specific replica used.

### HTTP test/draft path

`POST /test-validate` deliberately behaves differently. It loads the current
live rows from `subjects`, `tag_evaluations`,
`cfg_tag_subject_evaluations`, `rules`, and `summary_prompts`. It does not
require an active lifecycle and does not use the published snapshot. This is
why a draft UI test can differ from production processing until the config is
published.

### Draft/live-table extraction reference

Resolve the configuration by identifier, or by tracker mapping when the first
query returns no row:

~~~sql
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name,
    c.status,
    c.previous_status
FROM configurations AS c
WHERE c.identifier = %s
  AND c.is_archived = 0;
~~~

~~~sql
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name,
    c.status,
    c.previous_status
FROM configurations AS c
INNER JOIN tracker_mappings AS tm
    ON tm.configuration_id = c.id
WHERE tm.tracker_id = %s
  AND c.is_archived = 0;
~~~

Then load its subjects:

~~~sql
SELECT
    CAST(s.id AS CHAR) AS subject_id,
    s.name,
    s.entity_definition,
    s.validation_prompt,
    s.sentiment_prompt,
    s.prominence_prompt,
    s.is_default,
    s.sort_order
FROM subjects AS s
WHERE s.configuration_id = %s;
~~~

Load effective per-subject tag definitions. The subject-specific row wins when
it supplies a value:

~~~sql
SELECT DISTINCT
    CAST(tse.subject_id AS CHAR) AS subject_id,
    te.tag_id,
    te.tag_name,
    te.tag_group_id,
    te.tag_group_name,
    COALESCE(tse.evaluation_type, te.evaluation_type) AS evaluation_type,
    COALESCE(tse.prompt_text, te.prompt_text) AS prompt_text,
    COALESCE(tse.boolean_expression, te.boolean_expression)
        AS boolean_expression,
    te.sort_order
FROM tag_evaluations AS te
INNER JOIN cfg_tag_subject_evaluations AS tse
    ON tse.tag_evaluation_id = te.id
   AND tse.configuration_id = te.configuration_id
WHERE te.configuration_id = %s
  AND te.remote_status = 'active';
~~~

Load ordered rules and summary instructions:

~~~sql
SELECT
    CAST(r.id AS CHAR) AS rule_id,
    r.name,
    r.sort_order,
    r.conditions,
    r.actions
FROM rules AS r
WHERE r.configuration_id = %s
ORDER BY r.sort_order ASC;
~~~

~~~sql
SELECT
    sp.prompt_text,
    sp.rejected_prompt_text,
    sp.formatting_directives,
    sp.selected_template,
    sp.conditional_rules,
    sp.rejected_template
FROM summary_prompts AS sp
WHERE sp.configuration_id = %s;
~~~

The pipeline's live-table adapter uses the same logical mapping. For exact
behavior, prefer reusing
`curation_engine.adapters.mysql.config_store.MySQLConfigStore` rather than
maintaining an independent draft mapper.

### Production-faithful lookup by tracker ID

~~~sql
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at
FROM configurations AS c
INNER JOIN tracker_mappings AS tm
    ON tm.configuration_id = c.id
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE tm.tracker_id = %s
  AND (
      c.status = 'active'
      OR (
          c.status = 'pending_changes'
          AND COALESCE(c.previous_status, 'active') = 'active'
      )
  )
  AND c.is_archived = 0
ORDER BY vs.version_number DESC
LIMIT 1;
~~~

### Production-faithful lookup by configuration identifier

~~~sql
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at
FROM configurations AS c
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE c.identifier = %s
  AND (
      c.status = 'active'
      OR (
          c.status = 'pending_changes'
          AND COALESCE(c.previous_status, 'active') = 'active'
      )
  )
  AND c.is_archived = 0
ORDER BY vs.version_number DESC
LIMIT 1;
~~~

Always parameterize identifiers. Never build these queries with string
interpolation.

### Python snapshot loader

~~~python
from __future__ import annotations

import json
from typing import Any

import pymysql


PUBLISHED_BY_TRACKER_SQL = """
SELECT
    c.id AS numeric_config_id,
    c.identifier AS config_id,
    c.name AS config_name,
    vs.version_number,
    vs.snapshot_data,
    vs.published_at
FROM configurations AS c
INNER JOIN tracker_mappings AS tm
    ON tm.configuration_id = c.id
INNER JOIN cfg_version_snapshots AS vs
    ON vs.configuration_id = c.id
WHERE tm.tracker_id = %s
  AND (
      c.status = 'active'
      OR (
          c.status = 'pending_changes'
          AND COALESCE(c.previous_status, 'active') = 'active'
      )
  )
  AND c.is_archived = 0
ORDER BY vs.version_number DESC
LIMIT 1
"""


def load_published_snapshot(
    connection: pymysql.Connection,
    tracker_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(PUBLISHED_BY_TRACKER_SQL, (tracker_id,))
        row = cursor.fetchone()

    if row is None:
        raise LookupError(f"No active published config for tracker {tracker_id!r}")

    raw_snapshot = row["snapshot_data"]
    snapshot = (
        json.loads(raw_snapshot)
        if isinstance(raw_snapshot, str)
        else raw_snapshot
    )
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot_data is not a JSON object")

    metadata = {
        "numeric_config_id": row["numeric_config_id"],
        "config_id": row["config_id"],
        "config_name": row["config_name"],
        "version_number": row["version_number"],
        "published_at": row["published_at"],
    }
    return metadata, snapshot
~~~

Persist `config_id` and `version_number` with every test result. A result that
does not identify the exact configuration version is not reproducible.

## Published snapshot structure

The fields relevant to classification are:

| Path | Meaning |
|---|---|
| `configuration.id` | Numeric database ID. |
| `configuration.identifier` | Stable human/application identifier used as the pipeline `config_id`. |
| `configuration.name` | Display name. |
| `configuration.status` | Published lifecycle state captured in the snapshot. |
| `configuration.tracker_mappings[]` | Tracker IDs mapped to this config. |
| `summary_prompt` | Summary and rejection-reason configuration. |
| `subjects[]` | Subjects plus the effective tags mapped to each subject. |
| `rules[]` | Ordered post-processing rules. |
| `unmapped_tag_evaluations[]` | Active tags not mapped to a subject; retained for round-trip/versioning but not part of the pipeline's subject-tag mapping. |

Each `subjects[]` entry contains:

| Field | Meaning |
|---|---|
| `id` | Numeric subject ID. The pipeline converts it to a string `subject_id`. |
| `name` | Subject name. |
| `entity_definition` | Definition of the entity/topic. |
| `validation_prompt` | Client criteria for deciding whether the article is valid for this subject. |
| `sentiment_prompt` | Client criteria for positive, negative, neutral, or balanced sentiment. |
| `prominence_prompt` | Client criteria for primary, significant, or passing prominence. |
| `is_default` | Whether this is the default/fallback subject. |
| `sort_order` | Configurator ordering metadata. Preserve the snapshot array order for byte-faithful prompt replay. |
| `tag_evaluations[]` | Effective tags mapped to this subject. |

Each `subjects[].tag_evaluations[]` entry contains:

| Field | Meaning |
|---|---|
| `id` | Database row ID for the base tag evaluation. |
| `tag_id` | Stable tag identifier returned in results. |
| `tag_name` | Human-readable tag name. |
| `tag_group_id` | Configured tag-group ID. |
| `tag_group_name` | Human-readable group name. |
| `evaluation_type` | Exactly `llm` or `boolean`. |
| `prompt_text` | Criteria for an LLM tag. |
| `boolean_expression` | Deterministic expression for a boolean tag. |
| `is_subject_override` | Whether this effective definition came from a subject-specific override. |

The snapshot builder has already applied subject-specific values from
`cfg_tag_subject_evaluations`. Therefore, use the values nested under each
subject rather than replacing them with the base `tag_evaluations` row.

## How the pipeline constructs the prompts

The source of truth is:

- `curation-engine-pipeline/src/curation_engine/adapters/llm/prompt_builder.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/llm.py`

The Configurator contains a byte-faithful mirror for reconstructing the system
prompt from a published snapshot:

- `backend/app/prompt_optimizer/assembler.py`
- function `assemble_system_prompt(snapshot)`

That mirror is guarded by a golden test against the pipeline implementation. A
rig running inside this repository should import it rather than writing a new,
nearly equivalent prompt builder:

~~~python
from app.prompt_optimizer.assembler import assemble_system_prompt

system_prompt = assemble_system_prompt(snapshot)
~~~

### System-prompt assembly order

The pipeline builds one system message in this exact order:

1. Base role: content curation assistant.
2. Media-type policy.
3. Global validation/rejection rules.
4. Optional client summary instructions.
5. Optional client rejection-reason instructions.
6. A block for every subject.
7. A tag-evaluation section containing only LLM-type tags with a non-empty
   `prompt_text`.

The effective template is:

~~~text
You are a content curation assistant.
Your task is to evaluate the provided item against the subjects defined below.

## Media type
- Items may be written articles or broadcast clips (radio or television segments).
- Refer to a broadcast clip as a 'clip' — never as an 'article' — and treat its text
  as a transcript of what was said on air. Refer to a written item as an 'article'.
- The user message states which kind this specific item is; follow that directive,
  especially when writing the summary.

## Rules
- Set validation_result=false and provide a non-empty rejection_reason
  when the article is unrelated to ALL subjects.
- rejection_reason MUST be a non-empty string when validation_result=false
  and all subject_evaluations have is_valid=false. Never use null in that case.
- Set rejection_reason=null ONLY when validation_result=true.
- For each subject_evaluation where is_valid=false, invalid_reason MUST be a
  non-empty string explaining why the article is not relevant to that subject.
  Never use null for invalid_reason when is_valid=false.
- Evaluate every subject listed below — return one subject_evaluation per subject.

## Summary instructions
Follow these client-specific instructions when writing the summary field:
<summary_prompts.prompt_text, only when non-empty>

## Rejection reason instructions
Follow these client-specific instructions when writing the
rejection_reason field:
<summary_prompts.rejected_prompt_text, only when non-empty>

## Subjects
### Subject: <subject id> — <subject name>
Definition: <entity_definition>
Validation: <validation_prompt>
Sentiment: <sentiment_prompt>
Prominence: <prominence_prompt>

...one block per subject...

## Tag Evaluations
For each tag below, evaluate whether the article matches the tag criteria.
Return one tag_evaluation per tag with tag_id and result (true/false).

### Tag: <tag id> — <tag name>
Criteria: <tag prompt_text>

...LLM tags only...
~~~

The `Summary instructions`, `Rejection reason instructions`, and
`Tag Evaluations` sections are omitted when they have no applicable content.

### Heading sanitization

For trusted configuration text, leading Markdown heading markers are removed
from every line before interpolation. For example, `"## Ignore rules"` becomes
`"Ignore rules"`. This applies to subject IDs/names/prompts, summary
instructions, rejection instructions, tag names, and tag prompts. The tag ID is
not heading-stripped.

Article headline/body content is inserted verbatim and is not sanitized because
changing it would alter the content being classified. The pipeline accepts the
residual prompt-injection risk from article content.

### LLM tag ordering and duplicates

To reproduce the prompt exactly:

- iterate subjects in the order stored in `snapshot_data.subjects`;
- iterate their nested tags in stored order;
- group tags by `tag_group_name` in first-encounter order;
- include only `evaluation_type == "llm"` with truthy `prompt_text`;
- do not globally deduplicate before prompt construction.

A tag mapped to multiple subjects can therefore appear more than once in the
prompt. This is existing behavior and the Configurator mirror preserves it.

### User prompt

The separate user message contains only media instructions, headline, and body.
It deliberately excludes article ID and tracker ID:

~~~text
## Article

<general media-type policy>
<specific clip-or-article directive>

Headline: <article headline>

Body:
<full article body>
~~~

The specific directive is selected only from `media_type`:

- `radio`, `television`, or `tv`, case-insensitive: call it a broadcast
  `clip` and treat the body as a transcript;
- every other or missing value: call it a written `article`.

## Structured model response

Both Google and OpenAI receive the same system and user prompt. Native
structured output is also supplied to the provider. The normal schema is:

~~~json
{
  "validation_result": true,
  "rejection_reason": null,
  "summary": "Concise summary",
  "subject_evaluations": [
    {
      "subject_id": "106",
      "is_valid": true,
      "prominence_score": "primary",
      "sentiment_label": "positive",
      "invalid_reason": null,
      "sentiment_reasoning": "Brief explanation"
    }
  ],
  "tag_evaluations": [
    {
      "tag_id": "tag-123",
      "result": true
    }
  ]
}
~~~

Field rules:

| Field | Contract |
|---|---|
| `validation_result` | True when relevant to at least one subject; false otherwise. |
| `rejection_reason` | Required and non-empty when overall validation is false and all subjects are invalid; null when validation is true. |
| `summary` | Concise summary, following client and media-type instructions. |
| `subject_evaluations` | One entry for every configured subject. |
| `subject_id` | Must match the configured subject ID. |
| `is_valid` | Whether content is meaningfully related to the subject under its validation criteria. |
| `prominence_score` | One of `primary`, `significant`, or `passing`. |
| `sentiment_label` | One of `positive`, `negative`, `neutral`, or `balanced`. |
| `invalid_reason` | Non-empty for invalid subjects; null for valid subjects. |
| `sentiment_reasoning` | Brief sentiment explanation. |
| `tag_evaluations` | One entry per LLM-type tag; empty when none are configured. |
| `tag_evaluations[].result` | Boolean match against the tag's LLM criteria. |

When `include_diagnostics=true`, the provider receives an extended schema that
also requires short `validation_reasoning`, `prominence_reasoning`, and
`tag_reasoning` fields. Normal Kafka processing uses the standard schema and
those optional rationale fields may be null.

The currently deployed primary provider is Google with
`gemini-2.5-flash`; OpenAI `gpt-4.1-mini` is the secondary. Provider failover
does not change the prompt text or business schema. The rig should record its
provider and exact model and should not assume these deployment values are
permanent.

## Summary and rejection instructions

Summary generation is not a separate pipeline or model call. The same
classification call returns `summary` alongside subject and tag evaluations.

Only these database fields currently affect the classifier prompt:

| Database field | Pipeline use |
|---|---|
| `summary_prompts.prompt_text` | Inserted under `## Summary instructions` and applied to `summary`. |
| `summary_prompts.rejected_prompt_text` | Inserted under `## Rejection reason instructions` and applied to `rejection_reason`. |

The name `rejected_prompt_text` can be misleading: it instructs the overall
`rejection_reason`, not an alternative summary field.

The snapshot also preserves `formatting_directives`, `selected_template`,
`conditional_rules`, and `rejected_template`. The current classifier prompt
builder does not read those fields. A faithful test rig must not silently add
them to the prompt.

The base structured schema always describes the summary as concise. Client
`prompt_text` can add formatting, focus, length, or content requirements. The
media policy is always present and requires radio/TV output to call the item a
clip rather than an article.

## LLM tags versus boolean tags

`evaluation_type` is the authoritative discriminator:

| Type | Configuration field | Evaluated by | Sent to LLM? | Raw result location |
|---|---|---|---|---|
| `llm` | `prompt_text` | Primary/failover LLM during the classification call | Yes | `LLMResponse.tag_evaluations[]` |
| `boolean` | `boolean_expression` | Local deterministic `BooleanEvaluator` after the LLM call | No | `boolean_results[tag_id]` |

Do not infer the type from whether a prompt/expression happens to be null. Use
`evaluation_type` and validate that the corresponding definition exists.

### Boolean evaluation behavior

Boolean expressions are evaluated against:

~~~text
headline + " " + body
~~~

Matching is case-insensitive substring matching. The grammar supports:

- `AND`, `OR`, and `NOT`;
- parentheses, with precedence `NOT > AND > OR`;
- bare terms;
- quoted contiguous phrases;
- smart-quote normalization;
- a trailing wildcard `*`, which is stripped because matching is already
  substring-based;
- field qualifiers:
  - `country:...`
  - `language:...`
  - `media_type:...`
  - `source:...`
  - `metadata.<key>:...`
- blank checks such as `country:""`.

An invalid expression is logged and its result becomes false for that article.
The test rig should surface this as an explicit test failure/warning even if it
mirrors the false fallback.

### Preserve subject-specific tag definitions

The same `tag_id` can be mapped to more than one subject, and
`cfg_tag_subject_evaluations` can override its type, prompt, or boolean
expression per subject. For extraction and test reporting, key the effective
definition by `(subject_id, tag_id)`. Only collapse to a global tag ID after
checking that all effective definitions agree.

~~~python
from __future__ import annotations

from typing import Any


def extract_test_config(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = snapshot.get("summary_prompt") or {}
    subjects: list[dict[str, Any]] = []

    for raw_subject in snapshot.get("subjects", []):
        subject_id = str(raw_subject["id"])
        tags: list[dict[str, Any]] = []

        for raw_tag in raw_subject.get("tag_evaluations", []):
            evaluation_type = raw_tag.get("evaluation_type")
            if evaluation_type not in {"llm", "boolean"}:
                raise ValueError(
                    f"Unexpected evaluation_type={evaluation_type!r} "
                    f"for subject={subject_id}, tag={raw_tag.get('tag_id')}"
                )

            definition = (
                raw_tag.get("prompt_text")
                if evaluation_type == "llm"
                else raw_tag.get("boolean_expression")
            )
            if not definition:
                raise ValueError(
                    f"Missing {evaluation_type} definition for "
                    f"subject={subject_id}, tag={raw_tag.get('tag_id')}"
                )

            tags.append(
                {
                    "subject_id": subject_id,
                    "tag_id": raw_tag["tag_id"],
                    "tag_name": raw_tag.get("tag_name") or raw_tag["tag_id"],
                    "tag_group_id": raw_tag.get("tag_group_id"),
                    "tag_group_name": raw_tag.get("tag_group_name"),
                    "evaluation_type": evaluation_type,
                    "definition": definition,
                    "is_subject_override": bool(
                        raw_tag.get("is_subject_override", False)
                    ),
                }
            )

        subjects.append(
            {
                "subject_id": subject_id,
                "name": raw_subject["name"],
                "entity_definition": raw_subject.get("entity_definition") or "",
                "validation_prompt": raw_subject.get("validation_prompt") or "",
                "sentiment_prompt": raw_subject.get("sentiment_prompt") or "",
                "prominence_prompt": raw_subject.get("prominence_prompt") or "",
                "is_default": bool(raw_subject.get("is_default", False)),
                "tags": tags,
            }
        )

    return {
        "configuration": snapshot.get("configuration") or {},
        "summary_instructions": summary.get("prompt_text") or "",
        "rejection_reason_instructions": (
            summary.get("rejected_prompt_text") or ""
        ),
        "subjects": subjects,
        "rules": snapshot.get("rules") or [],
    }
~~~

## Post-processing rules and final results

Pipeline order is:

1. deduplication (Kafka only);
2. configuration resolution;
3. LLM processing;
4. boolean evaluation;
5. rules engine;
6. scored-attribute assembly.

Rules can:

- add or remove a tag;
- override a subject's sentiment;
- override a subject's prominence;
- set a subject's approved/rejected status.

Therefore raw LLM/boolean results are not always the final business output.
`set_subject_status` changes the final per-subject `is_valid` and
`invalid_reason`, but the current assembler does not recompute the top-level
`validation_result` or overall `rejection_reason`. Use `subject_results[]` for
post-rule subject validity and retain the top-level fields as the original
overall LLM decision.

For an end-to-end test:

- compare final per-subject output using archived
  `subject_results[]`;
- compare final grouped tags using
  `scored_attributes.tag_groups[].tags[]`;
- use `boolean_results` only as pre-rule boolean evidence;
- use raw `llm_raw_response.tag_evaluations` only as pre-rule LLM evidence;
- inspect `rules_trace[]` to explain changes.

The final tag objects already include `evaluation_type`. If working with the
subject's `tags` array, which contains IDs only, join each ID back to the
effective snapshot tag definition to recover whether it was `llm` or
`boolean`.

See
[`processed-article-blob-archive-access.md`](./processed-article-blob-archive-access.md)
for the full archived-result schema.

## Production-parity checklist

A faithful test result should record:

- environment;
- tracker ID and article ID;
- configuration identifier;
- numeric configuration ID;
- snapshot `version_number` and `published_at`;
- exact system prompt hash;
- exact user prompt hash;
- provider and model;
- whether diagnostics schema was enabled;
- structured raw LLM result;
- boolean results;
- rule trace;
- final subject and tag outputs;
- processing timestamps.

The implementation must:

- preserve snapshot ordering;
- use the published snapshot for production comparisons;
- use live tables only for explicit draft comparisons;
- include only LLM tags in the system prompt;
- evaluate boolean tags locally;
- honor per-subject tag overrides;
- apply post-processing rules before declaring parity;
- keep summary and rejection-reason instructions distinct;
- ignore unknown snapshot fields and tolerate additive schema changes;
- never log credentials, full prompts, or article bodies.

## Common implementation mistakes

- Reading live `subjects` rows and assuming they represent production.
- Using the highest `configurations.version` instead of the highest published
  `cfg_version_snapshots.version_number`.
- Sending boolean tags to the LLM.
- Treating every tag as global and losing subject-specific overrides.
- Adding `formatting_directives` or template fields that the current pipeline
  does not use.
- Treating `rejected_prompt_text` as a rejected-article summary prompt rather
  than a `rejection_reason` instruction.
- Comparing raw LLM tags with final post-rule tags.
- Omitting the provider's structured-output schema.
- Reordering subjects/tags and then expecting byte-identical prompts.
- Reusing the pipeline administrator database password.
- Putting secrets or article text into test logs or an LLM build prompt.

## Source-of-truth files

Pipeline:

- `curation-engine-pipeline/src/curation_engine/adapters/mysql/config_store.py`
- `curation-engine-pipeline/src/curation_engine/adapters/llm/prompt_builder.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/config.py`
- `curation-engine-pipeline/src/curation_engine/domain/models/llm.py`
- `curation-engine-pipeline/src/curation_engine/domain/boolean/evaluator.py`
- `curation-engine-pipeline/src/curation_engine/domain/stages/boolean_eval.py`
- `curation-engine-pipeline/src/curation_engine/domain/stages/rules_engine.py`
- `curation-engine-pipeline/src/curation_engine/domain/stages/scored_attributes.py`

Configurator:

- `backend/app/prompt_optimizer/assembler.py`
- `backend/app/versioning/service.py`
- `backend/app/configurations/models.py`
- `backend/app/prompts/models.py`
- `backend/app/tags/models.py`
- `backend/app/rules/models.py`
- `backend/app/summaries/models.py`
- `backend/app/versioning/models.py`

Deployment/credential wiring:

- `curation-engine-pipeline/infra/main.bicep`
- `curation-engine-pipeline/.github/workflows/build-push.yml`

If any of the prompt builder, structured output models, snapshot assembler, or
config-store mapper changes, update and revalidate the test rig before using it
for parity decisions.
