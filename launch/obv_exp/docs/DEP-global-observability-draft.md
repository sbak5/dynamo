# DEP (light): Global/cross-replica observability for KV cache and routing state

**Labels (proposed):** `dep:draft`, `dep:lightweight`, `router`
**Author:** Seonmyeong Bak (sbak@nvidia.com)

## Summary

Dynamo deployments run multiple frontend replicas sharing the same
prefill/decode workers. Observability today is split across five
non-integrated telemetry stacks, none of which give a global, cross-replica
view of KV cache state, worker load, or per-request lifecycle — so operators
can't see contention from *other* frontend replicas, and several stacks
duplicate the same data with incompatible models. This DEP opens discussion
on consolidating toward a global view, without prescribing an architecture.

## Motivation

- Multiple frontend replicas share the same observable prefill/decode workers.
- The per-request tracker has no visibility into contention from other frontends.
- Per-worker/per-frontend metrics give no global view of runtime state.
- Several existing observability stacks appear duplicative.

### Five non-integrated observability stacks

**A — Prometheus `/metrics` (per-process).** Central registry
`lib/runtime/src/metrics/prometheus_names.rs`; covers frontend pipeline
occupancy, backend/worker transport, runtime health, and router queue/load
metrics (`dynamo_router_*`, labeled by `router_id`,
`lib/llm/src/http/service/service_v2.rs:563-565,955-958`). Endpoint mounted
per-process (`lib/llm/src/http/service/metrics.rs:1955-1973`). **Strictly
per-process** — no in-Dynamo cross-replica aggregation.

**B — KV cache events + per-router radix-tree indexer.** Worker publishes
KV events over NATS (`lib/llm/src/kv_router/publisher/`); each router
independently subscribes to the full event fanout and rebuilds its own
radix tree (`lib/llm/src/kv_router/indexer/`). Default config
(`lib/kv-router/src/scheduling/config.rs:591-619`) is fully local:
`use_remote_indexer=false`, `serve_indexer=false`, `router_replica_sync=false`
— **N frontend replicas means N redundant copies of the same KV state.**
Opt-in alternatives exist but aren't default: `Indexer::Remote` (query one
served-indexer instance) and a standalone HTTP indexer service
(`lib/kv-router/src/services/indexer/`, exposes `/register /query /dump`) —
the closest existing building block to a global KV view, but not wired in,
and deliberately decoupled from `dynamo-runtime`'s discovery/event plane.

**C — Active-sequence / worker-load tracking + replica sync.**
`lib/kv-router/src/sequences/` (`ActiveSequences` authoritative locally,
`PromptRegistry` an advisory "torn read" projection,
`lib/kv-router/src/sequences/README.md`). `router_replica_sync` (default
`false`) optionally syncs `WorkerLoadSnapshot`s between router replicas over
NATS core pub/sub — no JetStream, no queue group, plain broadcast, and
routers self-filter their own echoed messages
(`replica_sync.rs:148-159`). **This is the one place that acknowledges
cross-replica contention at all**, but it's opt-in, best-effort, and
explicitly eventually-consistent/lossy by design.

**D — `RequestTracker` (per-request lifecycle).**
`lib/llm/src/protocols/common/timing.rs:83-182`. In-memory, per-request,
per-process — no etcd/NATS persistence, dies with the request's task.
Confirmed: no mechanism lets other frontend replicas see its state, directly
substantiating the stated gap. Feeds `request_trace`
(`lib/llm/src/request_trace/`), an in-process broadcast bus to local
JSONL/stderr sinks, gated by `DYN_REQUEST_TRACE`.

**E — Audit logging / OtelSink.** `lib/llm/src/audit/` captures full
request/response payloads (compliance, not perf). Own broadcast bus, fanned
to stderr/nats/jsonl/otel. Its `OtelSink` emits OTLP `LogRecord`s reusing the
same `OTEL_EXPORTER_OTLP_*` env vars as the runtime's own logging exporter —
two subsystems independently instantiating OTLP exporters against the same
collector.

*Shared plumbing:* `lib/llm/src/telemetry/bus.rs`'s generic `TelemetryBus<T>`
backs both D and E independently.

**KV router state:** each replica maintains its own independent radix-tree
view by default — no shared/global indexer in the default path.

**Frontend coordination:** etcd (`lib/runtime/src/discovery/`) is used
*exclusively* for service discovery/liveness — no cache-occupancy state
lives there. NATS carries KV events, request-plane dispatch, and one
bootstrap-only object-store snapshot (`radix-bucket`). The most promising
piggyback points for a future global store: that `radix-bucket` mechanism,
and the standalone indexer service's `/dump` endpoint (already aggregates
all workers in one process, but unused by default `kv_router` instances).

### Request traceability across process boundaries

Given stacks B/C are fragmented and eventually-consistent, how traceable is
one request end-to-end (frontend → prefill → decode)? **Request identity
itself is well-engineered; the artifacts needed to use it are the actual
gap.**

What works:
- One request ID, minted once at ingress (`openai.rs:500-539`), survives
  every process boundary via the pipeline `Context`/`Controller.id`
  (`lib/runtime/src/pipeline/context.rs:14-20,359-364`) — carried on NATS
  headers, rehydrated identically on the worker
  (`push_handler.rs:369-370`), reused across the prefill→decode handoff.
- Real distributed tracing (W3C `traceparent`, actual OTel spans, not just
  logs) is wired over the same NATS hop
  (`lib/runtime/src/logging.rs:657-701,1256-1378`,
  `addressed_router.rs:674`). With `OTEL_EXPORT_ENABLED` set consistently
  everywhere and one shared collector, a single `trace_id` genuinely spans
  frontend → prefill → decode today.

What's missing — the artifacts, not the ID:
1. **Routing "why" is DEBUG-only, never persisted** — the one log line
   tying `request_id + worker_id + overlap_blocks`
   (`push_router/selection.rs:170-181`) only exists if debug logging was on
   at that moment. Scoring against other candidates isn't tied to a
   request_id at all.
2. **`RequestTraceRecord` excludes the routing outcome by default**
   (`request_trace/record.rs:75`, `worker: None`).
3. **No worker-side `request_trace` exists** — it's constructed only in
   frontend/preprocessor code; the worker's only trace is whatever lands in
   raw logs.
4. **No cross-process joiner** — correlating one request across a frontend
   and two workers means manually grepping three log sources by hand.
5. **OTel stitching is all-or-nothing, silently** — if any process in the
   chain has export off, that hop just vanishes with no signal.

This refines duplication finding 5 below: identity is unified; the artifacts
keyed by it are DEBUG-gated, excluded by default, or absent on workers.

### Reproducer: what the routing-decision log actually looks like today

A Slurm reproducer (2 frontend replicas, 4 decode + 4 prefill vLLM workers,
disaggregated, `--router-mode kv`, `DYN_LOG=debug`) was run to check finding
1 concretely. Confirmed format and gaps:

**Format**: unstructured `tracing` text (not JSON, no metric), three
different call sites with three different field shapes, ANSI color codes
embedded inline (breaks naive `grep`/regex unless stripped first):
- `dynamo_llm::kv_router` → `[ROUTING_INPUT] request local hashes
  isl_tokens=... block_size=... local_hashes=[...]` (the request's own block
  hashes, before routing).
- `dynamo_kv_router::scheduling::selector` → one `Formula for worker_id=X
  ... = prefill_load_scale * adjusted_prefill_blocks + decode_blocks = A * B
  + C (raw_prefill_blocks: B, overlap_credit_blocks: N)` line *per
  candidate*, then one `Selected worker: worker_type=..., worker_id=X,
  logit=...` line for the winner — but the decode-worker variant of this
  line omits fields (`total blocks`) the prefill variant has, so each shape
  needs its own regex.
- `dynamo_llm::kv_router::push_router` → `[ROUTING] Best: worker_X ... with
  N/M blocks overlap`, `request_id`/`worker_id`/`overlap_blocks` as trailing
  span fields.

**Confirmed gap**: for one real request, all 4 prefill candidates showed
*identical* `raw_prefill_blocks=64.312, decode_blocks=0.000` → identical
`logit=64.312` (an exact tie), and separately all 4 decode candidates tied
at `logit=128.000`. These "load" terms are derived from the *request's own*
size (ISL/block_size), identical across every candidate being compared —
not a live measurement of any worker's current occupancy. **There is no
field anywhere in this log, at any level, that reflects a worker's current
queue depth/backlog** — so when candidates tie (which they did here, because
the synthetic benchmark's KV-overlap score also tied at 0), there is no way
to tell from the log why one specific worker won over its tied competitors.
This is a stronger version of finding 1: it's not just that the decision log
is DEBUG-gated and unpersisted — even with full debug logging on, the log
schema itself has no per-worker load field to explain a tied decision.

### Concrete duplication findings

1. KV hit-rate/overlap data is computed in three places with three
   different data models (radix tree node in B, per-request scalar in D's
   `RequestTracker.kv_overlap_blocks`/`cached_tokens`, Prometheus counter in
   A's `KvIndexerMetrics`) that only agree by convention.
2. D's and E's broadcast buses (both `TelemetryBus<T>`) serve nearly the
   same purpose, and E's new `OtelSink` duplicates an OTLP transport the
   runtime's own logging already uses against the same collector.
3. Worker load is tracked twice with different consistency models:
   Prometheus router-queue gauges (pull, per-replica, no reconciliation) vs.
   `ActiveSequences`/`PromptRegistry` (push, optionally synced but
   eventually-consistent/"torn reads").
4. Three KV-indexer topologies already exist (per-replica local [default],
   remote/served [opt-in], standalone HTTP service [feature-gated]) — a new
   proposal should position itself against the standalone service
   specifically, since it's the closest analog but was built for routing
   decisions, not observability.
5. Per-request identity isn't the gap it first appeared to be — it's
   unified end-to-end (see above). The real gap is that Prometheus (A)
   carries no per-request identity at all, and the artifacts that could
   carry the routing outcome per request (D on the frontend, nothing on the
   worker) are incomplete or missing.

## Proposal

Lightweight DEP — open discussion, not a locked design.

- **(a) Extend vs. build**: extend the standalone HTTP indexer service or
  `replica_sync` into a genuine global KV/load-state view, rather than
  building a third. Each has constraints today (the indexer service is
  deliberately decoupled from the discovery/event plane; `replica_sync` is
  lossy by design) a global-observability consumer may or may not tolerate.
- **(b) Shared request identity/store**: does cross-replica contention
  visibility need a shared store for `RequestTracker` state, and what
  consistency does it need (real-time vs. eventually consistent, push vs.
  pull)?
- **(c) Consolidate duplicate plumbing**: merge D's and E's
  `TelemetryBus`/OTLP-exporter usage into one export path.
- **(d) Close the request-traceability gap**: (i) promote the routing
  decision from a DEBUG log line to an always-recorded `RequestTraceRecord`
  field; (ii) add a worker-side `request_trace` equivalent; (iii) consider
  making OTel tracing default/required with an explicit "trace incomplete"
  signal instead of a silent gap.

Open questions:
- Real-time global view, or does eventually-consistent aggregation (periodic
  snapshot/dump) suffice?
- New standalone service, or extend the existing indexer service?
- Acceptable overhead/latency budget, given the current local-replay design
  was presumably chosen for low per-request latency?
- Should routing-decision recording and worker-side tracing be always-on, or
  opt-in with a lighter default than full DEBUG/OTel?

### Sketch: step-level instrumentation for efficiency

`RequestTracker` today has four coarse timestamps — enough for TTFT/E2E, not
enough to explain *why*, since it never records worker state (batch size,
queue depth, KV occupancy) at each point in time. Showing system-wide
efficiency needs timings aligned per request **and** per step, against
worker state at that instant. Illustrative schema:

```
RequestSpan {
  request_id, frontend_router_id,
  events: [
    StepEvent {
      request_id, worker_id, dp_rank, phase: Queued|Prefill|DecodeStep,
      step_index, t_start, t_end, tokens_in_step,
      // worker state captured inline, not scraped separately —
      // avoids clock-skew against a later join:
      batch_size_at_step, kv_cache_used_blocks, kv_cache_total_blocks,
      queue_depth_at_step, active_requests_on_worker,
    },
    ...
  ]
}
```

- **Emit from the worker**, not the frontend — closes gap 3 above; the
  worker is the only place that cheaply knows its own batch/KV state per
  scheduler iteration.
- **One event per decode iteration**, not per-token.
- **Reuse OTel spans** — make each `StepEvent` a child of the existing
  `kv_router.route_request` span rather than a new bus; correlation and a
  Gantt/waterfall view come for free wherever OTel export is already wired.

**Step 1 — per-request breakdown from the timestamps.** Each `StepEvent`'s
`t_start`/`t_end` decomposes one request's lifetime into named segments
instead of one opaque E2E number:

- `queue_wait = t_start(first Prefill event) - t_received`
- `prefill_compute = t_end(Prefill) - t_start(Prefill)`
- `decode_step_latency[i] = t_end(DecodeStep_i) - t_start(DecodeStep_i)`
  (this is ITL per step, per request)
- `inter_step_gap[i] = t_start(DecodeStep_i) - t_end(DecodeStep_{i-1})` —
  time the request's turn was *not* being served even though it was
  admitted, i.e. scheduler contention rather than compute cost

That last one is the key number: `compute` (actual GPU work) vs. `gap`
(waiting behind other requests on the same worker) are currently conflated
into a single per-request latency figure. Splitting them per-step is what
turns "this request was slow" into "this request was slow because it queued
behind N other sequences," attributable request-by-request.

**Step 2 — aggregating per-request breakdowns into worker/frontend
utilization.** Because every `StepEvent` also carries `worker_id` and
`batch_size_at_step`, the same records can be grouped the other way — by
worker instead of by request — to get a utilization view over a time window:

- `worker_busy_time = Σ (t_end - t_start)` over all StepEvents on that
  worker in the window → occupancy/utilization %.
- `worker_goodput = Σ tokens_in_step / worker_busy_time`, comparable against
  the theoretical max tokens/sec at the observed `batch_size_at_step` — the
  gap between the two is efficiency loss, and because it's derived from the
  same per-request records, it can be decomposed back down to *which
  requests* ate the difference.
- Grouping the same events by `frontend_router_id` instead gives per-frontend
  throughput and average `queue_wait`/`inter_step_gap` contributed to shared
  workers — i.e., how much load *this* frontend replica is putting on the
  pool, from the worker's own ground truth rather than each frontend's local
  (and possibly stale) view.

**Step 3 — attributing contention to other replicas.** Because
`active_requests_on_worker` is recorded by the worker itself at each step
(not inferred or synced from peers), a slow `inter_step_gap` on request R can
be explained directly: join R's gap window against the *other* concurrent
StepEvents on the same `worker_id` in that window, group those by
`frontend_router_id`, and attribute the gap proportionally to whichever
replica(s) had requests occupying the worker at the time. This answers the
original motivating question — contention from other frontend replicas —
per request and without relying on `replica_sync`'s lossy, opt-in broadcast
at all, since the worker's own step log is the single source of truth for
"who else was here."

**KV efficiency correlation**: joining `kv_overlap_blocks` (already captured
per-request in D) with `kv_cache_used_blocks_at_step` at the same timestamp
shows whether cache pressure/eviction hurt this request's hit rate — the two
numbers are currently disconnected (duplication finding 1).

Possible visualizations: a per-request waterfall/Gantt with segments colored
by the Step-1 breakdown (queue_wait / compute / inter_step_gap), with a
worker-utilization strip on the same time axis so a request's gap segments
visually line up with concurrent-request density from other frontends; and
an aggregate scatter of KV-overlap-ratio vs. TTFT colored by worker or
contending frontend to surface systemic hotspots.
