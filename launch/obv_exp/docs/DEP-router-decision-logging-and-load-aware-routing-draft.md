# DEP (light): Router decision logging and load-aware routing

**Labels (proposed):** `dep:draft`, `dep:lightweight`, `router`
**Author:** Seonmyeong Bak (sbak@nvidia.com)

## Summary

Two related, code-grounded findings from a live reproducer (2 frontend
replicas, disaggregated prefill/decode, `--router-mode kv`,
`--enforce-disagg`, multi-turn workload) plus source investigation of
`lib/kv-router/src/scheduling/selector.rs`:

1. **Routing decision logging** is real (per-request, per-candidate) but
   unstructured, DEBUG-gated, ANSI-polluted, and split across three
   incompatible line shapes — unusable for anything beyond ad hoc `grep`.
2. **Neither routing mode's "load" signal reflects genuine worker
   occupancy.** Both are router-local, advisory estimates. The infrastructure
   to fix this — real worker-reported queue depth/batch/step-latency — already
   exists in-tree (`ForwardPassMetrics`) and is proven in production by the
   planner/autoscaler, but is never wired into the router.

## Motivation

### Finding 1: routing decision logging is unstructured, DEBUG-gated, incomplete

Confirmed format, from a live reproducer run with `DYN_LOG=debug`: plain
`tracing` text (not JSON, no metric), three different call sites with three
different field shapes, ANSI color codes embedded inline (breaks naive
`grep`/regex unless stripped first — `sed 's/\x1b\[[0-9;]*m//g'`):

- `dynamo_llm::kv_router` → `[ROUTING_INPUT] request local hashes
  isl_tokens=... block_size=... local_hashes=[...]`
- `dynamo_kv_router::scheduling::selector` → one `Formula for worker_id=X
  ... = prefill_load_scale * adjusted_prefill_blocks + decode_blocks = A * B
  + C (raw_prefill_blocks: B, overlap_credit_blocks: N)` line *per
  candidate*, then one `Selected worker: worker_type=..., worker_id=X,
  logit=...` line for the winner — the decode variant of this line omits
  fields (`total blocks`) the prefill variant has.
- `dynamo_llm::kv_router::push_router` → `[ROUTING] Best: worker_X ... with
  N/M blocks overlap`, `request_id`/`worker_id`/`overlap_blocks` as trailing
  span fields.

Concrete problems this caused during the reproducer session:
- Reconstructing "which worker won and why" for one request required
  manually stripping ANSI codes, then grepping three different line shapes
  and joining them by `request_id` — no tool does this automatically.
- Correlating decisions to load, once we determined a real per-worker load
  signal existed (`ActiveSequences`/`decode_blocks`), still doesn't help: the
  log shows the *value used*, not why it was zero/stale/fallback for a given
  candidate, so a tied decision (see below) can't be explained after the
  fact even with full debug logging.
- No signal for candidates that were *not* selected beyond their own
  `Formula` line — no ranked comparison, no explicit "why worker X lost to
  worker Y" statement.

**Reproducer example of an unexplainable tie** (`--router-mode kv`, no
`--enforce-disagg`, single frontend, one incoming request, 4 prefill
candidates): all four showed identical `raw_prefill_blocks=64.312,
decode_blocks=0.000` → identical `logit=64.312`. The log offers no field
that explains why one specific worker won over its three tied competitors.

### Finding 2: neither routing mode's load signal reflects genuine worker occupancy

**`least-loaded` mode** (`lib/runtime/src/pipeline/network/egress/push_router.rs:984-1008`,
`least_loaded()`): reads `RoutingOccupancyState`
(`lib/runtime/src/component/client.rs:23-121`) — a `DashMap<instance_id,
AtomicU64>` incremented when *this router* dispatches a request to a worker
and decremented when *this router* observes that request's response stream
finish. Read live, no caching — but it is **not worker-reported**; it's an
estimate of what one router process has itself sent, invisible to other
frontend replicas sharing the same worker pool.

**`kv` (KV-aware) mode**'s `worker_logit`
(`lib/kv-router/src/scheduling/selector.rs:110-238`) combines two
independently-sourced per-worker signals:
- `overlap_credit_blocks` (KV cache hit credit) — **genuinely worker-reported**,
  fed by real KV-cache store/remove events published by backend workers into
  a prefix/radix tree (`lib/llm/src/kv_router/publisher/`,
  `lib/llm/src/kv_router/route_lookup.rs:95-133`).
- `decode_blocks`/`raw_prefill_blocks` (the "load" term) — sourced from
  `ActiveSequences`/`PromptRegistry`
  (`lib/kv-router/src/sequences/prompt_registry.rs:30-44,317-340`), which is
  **the same category of thing as `least-loaded`'s counter**: router-local
  bookkeeping. Increments on the router's own dispatch (`AddRequest`,
  `lib/kv-router/src/scheduling/queue.rs:743`), transitions when the router
  observes the first streamed token back
  (`lib/llm/src/kv_router/push_router/request_guard.rs:305-328`), decrements
  on the router's own request completion. Never a worker-published metric.

**Even where it is genuinely per-worker, block-count is not a good proxy
for compute workload.** Decode compute cost has two components that scale
with different things: attention cost scales with each sequence's KV-cache
length (summed across the batch, roughly proportional to total active
blocks — what `decode_blocks` tracks), while FFN/matmul cost scales with
batch size (concurrent sequence count) largely independent of context
length. A worker serving many *short* requests can show low `decode_blocks`
(little KV memory in use) while being compute-saturated on the FFN side from
high concurrency — invisible to the KV-aware load term.  Conversely,
`least-loaded`'s raw request count catches concurrency but is blind to
memory/attention-cost from long-context requests. Neither signal captures
both dimensions; neither is real GPU utilization, queue depth, or step
latency from the worker itself.

### Finding 3: the missing piece already exists in-tree — ForwardPassMetrics

Confirmed via source investigation: **`ForwardPassMetrics` (FPM)** is a real,
periodic (per scheduler iteration — every forward pass) worker→event-plane
stats push:
- Schema (`components/src/dynamo/common/forward_pass_metrics.py:153-190`):
  `wall_time` (real iteration latency), `scheduled_requests` (actual
  prefill/decode batch composition, token counts, prefix-cache-hit tokens),
  `queued_requests` (real **in-engine queue depth** — admitted but
  not-yet-scheduled requests).
- Producer: vLLM (`components/src/dynamo/vllm/instrumented_scheduler.py`,
  `InstrumentedScheduler`/`_FpmPublisherThread`); TRT-LLM/SGLang wiring
  appears partially present despite a `TODO` comment
  (`forward_pass_metrics.py:33`).
- Transport: engine subprocess → local ZMQ → Dynamo event plane (NATS/ZMQ) →
  any `FpmEventSubscriber`.
- **Consumers, confirmed by grep**: only the planner/autoscaler
  (`components/src/dynamo/planner/core/base.py`,`load_scaling.py`,
  `perf_model/*`) — genuinely uses live FPM data to drive scale-up/down and
  performance-model tuning. The experimental, unreleased
  `thunderagent_router` subscribes to the same channel but explicitly
  discards the live per-iteration fields, reading only one static capacity
  number.
- **Confirmed absent from the router**: zero references to
  `FpmEventSubscriber`/`forward_pass_metrics`/`ForwardPassMetrics` anywhere
  in `lib/kv-router/`, `lib/llm/src/kv_router/`, or the production
  router/frontend components. `selector.rs`'s imports contain no
  stats/telemetry types.

**This changes the shape of the ask**: this isn't "invent worker-side load
reporting" — that capability exists, is production-proven (the planner
already depends on it), and simply isn't plumbed into
`lib/kv-router/src/scheduling/selector.rs`'s `worker_loads` construction.

## Proposal

Lightweight DEP — open discussion, not a locked design.

- **(a) Structured, joinable routing-decision records.** Replace (or
  supplement) the three free-text log shapes with either: a single
  structured span/event per decision carrying all candidates' scores plus
  the winner (queryable via OTel, consistent with the tracing infrastructure
  already found to be well-engineered in the companion observability DEP),
  or a promotion of the routing outcome into `RequestTraceRecord` (Stack D
  from the companion DEP) so it's captured alongside per-request timing
  without a separate DEBUG-only code path. Should not be DEBUG-gated for the
  winner + top losing candidates, given the demonstrated cost of losing this
  data after the fact.
- **(b) Wire FPM into the router's load signal.** Extend
  `worker_loads`/`ActiveSequences`'s construction in
  `lib/kv-router/src/scheduling/` to also consume the existing
  `FpmEventSubscriber` stream (the same one the planner already reads),
  giving `worker_logit` access to real `queued_requests` (actual in-engine
  queue depth) and `scheduled_requests` (actual batch composition) instead
  of only router-local dispatch/completion bookkeeping. This would also
  naturally fix the per-frontend-replica visibility gap (Finding 2/Stack C
  in the companion DEP), since FPM is worker-published and consumable by
  every router replica identically, unlike `ActiveSequences`.
- **(c) Consider a combined workload term.** Given Finding 2's two-dimension
  critique (memory/attention-length vs. concurrency/batch-size), evaluate
  whether the logit formula should combine `decode_blocks` (memory/context
  proxy) with FPM's `scheduled_requests` count (concurrency/FFN proxy)
  rather than relying on either alone.

Open questions:
- Should structured decision logging be always-on by default (cost: log
  volume ~1 event/candidate/request) or opt-in with a lighter default than
  full `DYN_LOG=debug`?
- FPM is per-iteration (very high frequency) — should the router consume it
  directly per-decision, or maintain a locally-decayed/smoothed view to
  avoid over-reacting to single-iteration spikes?
- Does extending `worker_loads` with FPM change the KV-aware formula's
  tuning (`prefill_load_scale` and friends), and does that need
  re-validation against existing benchmarks?
- Should (b) apply to `least-loaded` mode too (replacing
  `RoutingOccupancyState` with FPM-derived queue depth), or is that
  intentionally kept simple/local as a fallback mode?
