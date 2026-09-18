# TypeSafe Primitives Reference

TypeSafe provides three question primitives that evaluate against the provided `state`. All questions in a single request run in parallel and in isolation.

## 1. Noul (Boolean / Likelihood)

Evaluates whether a statement is true and returns a calibrated probability (0.0 to 1.0).

- **API Request Fields**:
  - `type`: `"noul"`
  - `instructions` (string | object | array): The yes/no statement or question.
  - `criteria` (optional object):
    - `true` (string): Description of what a "yes" (near 1.0) indicates.
    - `false` (string): Description of what a "no" (near 0.0) indicates.
- **Answer Shape**:
  - `type`: `"noul"`
  - `noul`: float between `0.0` (definitely false) and `1.0` (definitely true).

```json
{
  "type": "noul",
  "instructions": "Does this support message indicate payment failure?",
  "criteria": {
    "true": "User explicitly mentions failed checkout or credit card charge error",
    "false": "User asks about pricing or general account settings"
  }
}
```

## 2. Choice (Categorical Selection)

Selects the single best option from a discrete set of options.

- **API Request Fields**:
  - `type`: `"choice"`
  - `instructions` (string | object | array): Decision prompt.
  - `criteria` (map<string, string | null>): Map where keys are candidate choices, values are descriptions (or `null` if self-explanatory).
- **Answer Shape**:
  - `type`: `"choice"`
  - `choice`: string (highest-probability option).
  - `probabilities`: map<string, float> (distribution across all candidate options, summing to 1.0).
  - `confidence`: float (0.0 to 1.0, derived from distribution concentration).

```json
{
  "type": "choice",
  "instructions": "Which department should handle this ticket?",
  "criteria": {
    "billing": "Invoices, payment errors, chargebacks",
    "technical": "API errors, bugs, downtime",
    "general": null
  }
}
```

## 3. Score (Ordinal Rubric)

Rates content across an ordered list of descriptive levels (minimum 2 levels). Returns a probability-weighted expected score across the level indices.

- **API Request Fields**:
  - `type`: `"score"`
  - `instructions` (string | object | array): Rating prompt.
  - `criteria` (array<string | object>): Ordered list of levels (e.g., from lowest to highest).
- **Answer Shape**:
  - `type`: `"score"`
  - `score`: float (expected value across index levels, e.g. 1.83 on a 0-2 scale).
  - `legend`: map<string, string> (maps level indices "0", "1", ... to criteria text).
  - `probabilities`: map<string, float> (probability assigned to each level index).
  - `confidence`: float (0.0 to 1.0).

```json
{
  "type": "score",
  "instructions": "Rate customer frustration level",
  "criteria": [
    "Calm and polite inquiry",
    "Noticeably irritated or impatient",
    "Extremely angry, demanding escalation or threatening cancellation"
  ]
}
```

## Advanced: Structured Instructions & Criteria

Instead of flat strings, `instructions` and `criteria` values can be JSON objects or arrays. This allows passing complex domain schemas, rubric specifications, or multi-field validation criteria directly.

## Understanding Confidence

- `confidence` is reported on **Choice** and **Score** answers (not Noul).
- It measures how concentrated the probability distribution is compared to uniform spread.
  - Concentrated on one outcome $\rightarrow$ confidence near 1.0.
  - Spread evenly among options $\rightarrow$ confidence near 0.0.
- **Confidence vs. Probability**:
  - Probability tells you *what* the model believes.
  - Confidence tells you *how certain* the model is across the full distribution.
- Use confidence for thresholding and routing (e.g., auto-apply if `confidence >= 0.85`, else escalate to human review or fallback agent).
