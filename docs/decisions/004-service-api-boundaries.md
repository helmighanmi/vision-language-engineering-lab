<!--
Path: docs/decisions/004-service-api-boundaries.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-004: The HTTP service is a single-process adapter

## Decision

Ship the HTTP API as an optional `[service]` extra under
`src/vlm_engineering/service`, with these boundaries:

- **Thin adapter.** Routes validate, resolve policy and call the same
  package APIs used by the CLI, notebooks and scenarios. Model loading,
  retrieval and evaluation logic stay outside the service layer.
- **One model per capability.** The service serves the configured Qwen or
  OpenAI-compatible provider. Per-request `model_size` / `model_id` /
  `model_path` overrides are rejected with `400` instead of triggering a
  reload or being silently ignored.
- **Synchronous for single-shot work, jobs for pipelines.** Describe,
  analyze, embed, rerank and query answer in the request. Ingest, index
  and evaluate run as in-process background jobs with polling and
  deletion.
- **Plain uvicorn, one process per replica.** No gunicorn worker fan-out
  and no distributed state. Authentication, rate limiting, request ids,
  logs, metrics and jobs are process-local.
- **Opt-in exposure.** Authentication is disabled only because no key is
  configured; remote images stay disabled unless explicitly allowed, and
  local media/output references are confined to their configured roots.

## Why

Keeping the service thin preserves a single tested implementation for every
capability and lets deployments swap the transport (CLI, HTTP, notebook)
without reimplementing retrieval logic. Rejecting per-request model
selection avoids GPU thrashing and makes latency predictable. Process-local
state is honest: it needs no Redis, queue or metrics backend, and the limits
are documented rather than hidden behind a fake cluster story.

## Consequences

- Jobs, buckets and counters reset on restart and are not shared across
  workers or replicas. Durable or horizontally scaled deployments need an
  external queue, gateway rate limiting and per-replica metric scraping.
- The request deadline protects the caller, not a synchronous GPU call that
  is already running in a worker thread.
- Scaling out means more replicas with independent caches and memory; it
  does not make the service a distributed system.
- Any future move to shared state (Redis jobs/limits, Prometheus histograms,
  a real queue) is an explicit architecture change, not an implementation
  detail.
