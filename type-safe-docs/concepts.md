# TypeSafe System One & Jev Overview

TypeSafe builds "System One" AI models designed to make fast, calibrated, structured decisions directly consumable by software, rather than generating freeform text.

## Core Concepts

- **System One vs. System Two (Generative LLMs)**: Generative LLMs generate text for human consumption; coercing them into structured decisions requires fragile prompt engineering, JSON schema formatting, and output parsing. System One directly evaluates a target `state` against typed questions and outputs typed values and probability distributions.
- **Flagship Model (`Jev`)**:
  - Model ID: `jev-1.13.0` (aliases: `jev-latest`, `jev-preview`).
  - Pricing: Charged per input token ($42 / Btok or $0.042 / Mtok). Output tokens are free.
  - Rate Limits: 250,000 tokens/sec, 1,200 req/min. Exceeding limits returns HTTP 429.
  - Context Limits: 64k tokens per request total; 32k tokens max for `state` plus the longest single question.
  - Input: Text, JSON objects, or arrays of text. No image/audio/video (must pre-process to text/metadata).
- **Training (RLCD)**: Trained with Reinforcement Learning from AI Feedback / Calibrated Decisions (RLCD) for calibrated probabilities. Not adapted per-customer via LoRA/fine-tuning.
- **Data Handling**: Zero data retention (ZDR) available for enterprise. Customer requests/responses are never used for model training.

## State Guidelines

`state` is the arbitrary content/context evaluated by Jev:
- Can be a string, JSON object, or list.
- Keep state relevant: irrelevant noise degrades accuracy. Pre-filter large documents.
- State is ingested once per request; every question evaluates against it in parallel without cross-question interference or context rot.

## Model Jaggedness & Failure Modes (Jev 1.13)

1. **Literal Reading**: Jev interprets instructions literally. Explicitly define criteria and boundary cases rather than assuming implied intent.
2. **Arithmetic & Numeric Calculation**: Jev struggles with exact arithmetic. Keep math in application code (e.g., let Jev classify categories or extract quantities, compute sums/totals in code).
3. **Date/Time Comparisons**: Don't ask Jev if Date A is before Date B. Have Jev extract or identify date strings, parse and compare via standard datetime libraries in code.
4. **Indirection & Multi-hop Reasoning**: High-hop relational lookups struggle. Point directly to relevant state fragments or decompose into chained evaluations.
5. **Large Irrelevant State**: Noise dilutes attention. Truncate, chunk, or BM25-filter state before passing to Jev.
6. **Adversarial Input**: Prompt injection in `state` can attempt to manipulate questions. Explicitly instruct criteria on handling untrusted user input.
