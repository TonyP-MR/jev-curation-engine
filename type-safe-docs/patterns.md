# System One Architectural Patterns

Patterns for structuring software systems around TypeSafe's System One model.

## 1. Atomic Questions & Composition in Code

Instead of prompting an LLM to evaluate complex, multi-variable decisions in one step:
- Break complex judgments into distinct, single-focus questions evaluated in parallel.
- Combine and weight results deterministically in standard code.
- Enables changing business rules, weights, and thresholds instantly in application code without retraining or re-prompting.

## 2. Speculative Fan-Out

Because Jev parses the `state` once and evaluates questions in parallel with negligible incremental latency:
- Send all potentially relevant questions upfront in a single batch (e.g. classification, urgency, sentiment, intent, routing flags).
- Application code inspects only the questions relevant to downstream branching branches.
- Eliminates sequential round-trip API calls and keeps round-trip latency flat.

## 3. Confidence-Gated Routing

Use the 2-axis decision pattern:
- **Axis 1 (Prediction)**: The `choice` or `score`.
- **Axis 2 (Risk/Certainty)**: The `confidence` score.
- **Routing Rules**:
  - `confidence >= HIGH_THRESHOLD`: Fully automated execution (e.g., auto-refund, auto-tag).
  - `MID_THRESHOLD <= confidence < HIGH_THRESHOLD`: Tier 2 automated workflow or secondary verification.
  - `confidence < MID_THRESHOLD`: Escalate to human review queue or System Two generative fallback.

## 4. Intent Routing & Cascading

Route incoming requests to the most efficient tier:
- **Tier 1 (System One / Jev)**: Classify user query intent via `choice` with sub-100ms latency and high token efficiency.
- **Deterministic Action**: If intent matches a deterministic command (e.g. "check status", "reset password"), execute directly via internal microservices.
- **Tier 2 (System Two LLM)**: If intent is open-ended synthesis or creative generation, forward to a generative model with tailored system instructions.
- **Tier 3 (Human)**: Low confidence or sensitive intents route to human operators.

## 5. Composite Scoring

Compute multi-dimensional scores across distinct qualitative dimensions:
```python
# Example: Lead qualification
weights = {"budget": 0.4, "timeline": 0.3, "fit": 0.3}
total_score = sum(
    answers[dim].score * weights[dim]
    for dim in weights
)
```
- Decouples subjective evaluation (done by Jev per rubric) from mathematical calculation (done in Python/TypeScript).

## 6. Self-Consistency Verification

- For critical boundaries (e.g. moderation or fraud detection), include complementary or inverted check questions in the same request (e.g. asking "Is content compliant?" and "Does content violate policy X?").
- Compare probabilities in code to flag inconsistent edge cases for human audit.
