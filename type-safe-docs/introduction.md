# TypeSafe Introduction

TypeSafe's flagship model, **Jev**, is a "System One" model designed for software-consumed structured decisions rather than generative text output. Instead of generating and parsing text, Jev directly evaluates typed questions against arbitrary state and outputs structured, typed values and probability distributions.

## Primitives

All primitives evaluate questions against a given state and can be mixed in a single parallel API call without context rot or latency degradation:

| Question Type | Purpose | Return Value |
|---|---|---|
| **Choice** | Select an option from a list | `choice`, `probabilities`, `confidence` |
| **Score** | Score state against a rubric | `score`, `probabilities`, `confidence` |
| **Noul** | Evaluate if a statement is true | `noul` (0–1 float probability) |

## Core Architecture Pattern: Atomic Decomposition

- **Atomic gut-checks**: System One models perform best on isolated, well-scoped questions evaluated in parallel.
- **Compose in code**: Decompose multi-factor evaluations into separate single-factor questions. Aggregate and weight them programmatically in application logic rather than relying on complex single-prompt text generation.
