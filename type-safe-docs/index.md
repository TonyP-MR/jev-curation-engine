# TypeSafe Documentation Index

This directory contains condensed reference documentation scraped and synthesized from [docs.typesafe.ai](https://docs.typesafe.ai/).

## Overview of Reference Files

| File | Content & Focus |
|---|---|
| [`introduction.md`](./introduction.md) | High-level introduction to TypeSafe, Jev (System One), core primitives, and atomic composition. |
| [`concepts.md`](./concepts.md) | System One vs Generative LLMs, Jev model specs & limits (pricing, token context, rate limits), state structuring guidelines, and known model jaggedness / failure modes. |
| [`primitives.md`](./primitives.md) | Exhaustive reference for the three question types: **Noul** (boolean probability), **Choice** (categorical selection + distribution), and **Score** (ordinal rubric expected value), plus `confidence` behavior and structured criteria. |
| [`patterns.md`](./patterns.md) | Architectural blueprints: Speculative Fan-Out, Confidence-Gated Routing, Intent Routing & Cascading, Composite Scoring, and Self-Consistency Audits. |
| [`api_and_sdk.md`](./api_and_sdk.md) | HTTP REST API specifications (`POST /v1/systemone`, `GET /v1/models`), Python SDK (`typesafe-sdk`), TypeScript/JS SDK (`@typesafe-ai/sdk`), and Agent Skill integration. |
| [`cookbooks.md`](./cookbooks.md) | Summary of all 18 practical cookbooks (RAG filtering, citation checks, guardrails, entity alignment, function calling, re-ranking, etc.). |
