# Lifecycle instrumentation implementation progress

Last updated: 2026-08-27

This is the working implementation record for native OpenTelemetry lifecycle
instrumentation in the Dynamo v1.4.1 runtime. M1 established timing boundaries;
M2 adds typed request identity and one-shot terminal outcomes. It complements:

- `LIFECYCLE_INTERACTIVE_DEVELOPMENT.md` — fresh-session setup and iterative mock.
- `LIFECYCLE_SPAN_DEVELOPMENT_RUNBOOK.md` — broader lifecycle-span operating notes.
- `/home/scratch.sbak_coreai/dynamo-observe/research/model-based-diagnosis/PROPOSAL-final-candidate.md` — target contract and milestones.

## Current scope

M1 and M2 deliberately stop short of detailed metrics. The runtime now provides
timing boundaries, a versioned lifecycle profile/mode, request/session/operation
identity, and a terminal result. It does not yet add queue-depth facts, token
counts, KV byte counts, direct engine events, or anomaly thresholds.

- Branch: `sbak/lifecycle-runtime-instrumentation`
- Development worktree: `~/scratch/dynamo-lifecycle-dev`
- Clean build checkout: `~/scratch/dynamo`
- Runtime/build image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1`
- Feature gate: `DYN_LIFECYCLE_TRACE_ENABLED=true`
- Pushed branch head: `7b9b55eca3` on `fork/sbak/lifecycle-runtime-instrumentation`

Do not modify the clean checkout's canonical real-workload launchers. The
development worktree holds the implementation and its development-only launch
copies.

## Implemented runtime boundaries

The existing frontend and router timing spans are joined with worker-runtime
operation spans. The retained real vLLM trace has this structure for every
request:

```text
http-request
└─ request.lifecycle
   ├─ request.preprocessing
   ├─ kv_router.select_worker #1 → kv_router.schedule → router.selection
   ├─ kv_router.route_request #1
   │  └─ handle_payload
   │     ├─ worker.admission
   │     ├─ request.dispatch
   │     ├─ worker.operation.prefill
   │     └─ response.streaming.prefill
   ├─ kv_router.select_worker #2 → kv_router.schedule → router.selection
   ├─ kv_router.route_request #2
   │  └─ handle_payload
   │     ├─ worker.admission
   │     ├─ request.dispatch
   │     ├─ worker.operation.decode
   │     └─ response.streaming.decode
   └─ response.streaming
```

The `worker.operation.prefill` and `worker.operation.decode` spans are
Rust-runtime worker-operation boundaries, selected by
`DYN_DISAGGREGATION_MODE`. They start around the worker's `generate()`
operation and remain active while the response is pumped. They may include
backend queueing and decode-side KV wait. The `engine.*` names are reserved for
future direct engine signals and must not be used for this coarse boundary.

### Source changes

| File | Progress |
| --- | --- |
| `lib/runtime/src/config/environment_names.rs` | Adds `DYN_DISAGGREGATION_MODE` to runtime lifecycle configuration. |
| `lib/runtime/src/telemetry.rs` | Adds role-specific worker streaming stages and helpers for `worker.operation.prefill` / `worker.operation.decode`; includes name-level unit coverage. |
| `lib/runtime/src/pipeline/network/ingress/push_handler.rs` | Wraps worker generation and response pumping with the role-selected operation spans. |
| `lib/kv-router/src/scheduling/queue.rs` | Preserves the sender's tracing context through the long-lived scheduler admission channel so router lifecycle spans remain parented to the request. |

Role-specific response names remove the ambiguity of the earlier three generic
`response.streaming` spans:

```text
response.streaming.prefill
response.streaming.decode
response.streaming          # frontend-to-client stream
response.streaming.worker   # role is unknown or aggregated
```

## M2: typed identity and terminal outcomes

M2 is implemented in the Rust runtime. Each retained `request.lifecycle` root
now carries the bounded schema needed to group a request's frontend, router,
and worker timing spans:

```text
dynamo.lifecycle.schema=v1
dynamo.lifecycle.profile=generic.v1     # configurable with DYN_LIFECYCLE_TRACE_PROFILE
dynamo.lifecycle.mode=core              # configurable with DYN_LIFECYCLE_TRACE_MODE
dynamo.request.id=<UUID>
dynamo.request.attempt=0
dynamo.session.id=<agent session or request-id fallback>
dynamo.session.source=<agent_context|request_id_fallback>
dynamo.operation.id=<per-operation UUID>
dynamo.operation.role=<frontend|prefill|decode|worker>
dynamo.request.terminal.outcome=<success|rejected|timed_out|failed|cancelled|unknown>
dynamo.request.terminal.error=<bool>
```

`request_id` is the grouping key for all P/D work for an end-user request.
`operation_id` distinguishes operation waves within that request. Each router
selection currently begins a distinct frontend-role operation wave. M3 will add
explicit `dynamo.operation.parent_id` and/or OTel links for P→D causality.
That relationship is intentionally not inferred from a shared request ID.

The terminal recorder is one-shot: concurrent completion, disconnect, timeout,
and backend-error paths cannot overwrite the first observed outcome. The
frontend maps HTTP 4xx responses to `rejected`, typed request-plane
`ResponseTimeout` errors to `timed_out`, client disconnects to `cancelled`,
and streamed backend errors to `failed`.

The timeout injection pauses a worker before its first response. It therefore
exercises `DYN_TCP_REQUEST_TIMEOUT` (five seconds by default), rather than the
post-SSE inactivity safety net controlled by
`DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`.

### M2 source changes

| File | M2 responsibility |
| --- | --- |
| `lib/runtime/src/telemetry.rs` | Versioned profile/mode, lifecycle identity, operation role, root terminal recorder, and the M3 propagation note. |
| `lib/llm/src/http/service/openai.rs` | Creates the frontend root, propagates identity, and records HTTP, timeout, streaming, and task-failure terminal paths. |
| `lib/llm/src/http/service/disconnect.rs` | Records `cancelled`, `timed_out`, and `failed` at the definitive stream/connection-monitor paths. |
| `lib/llm/src/http/service/metrics.rs` | Detects typed response-timeout errors without parsing error text. |
| `lib/runtime/src/pipeline/network/egress/tcp_client.rs` | Classifies an elapsed request deadline as `ResponseTimeout`, while retaining `CannotConnect` for connection failures. |
| `lib/llm/src/preprocessor.rs`, `lib/runtime/src/pipeline/network/ingress/push_handler.rs`, `lib/kv-router/src/scheduling/queue.rs`, `lib/backend-common/src/adapter.rs` | Preserve request-scoped lifecycle identity at frontend, router, and worker boundaries. |
| `lib/runtime/src/config/environment_names.rs` | Declares the profile/mode configuration names. |

### M2 build and terminal-path validation

The M2 override wheels were built from this worktree with the required base
image and validated with the frontend-only mock. The mock is development-only;
it does not modify the real P/D workload launchers.

| Item | Evidence |
| --- | --- |
| Build | Job `1968025`, `COMPLETED (0:0)` on `gh-nvl-203-compute02`; wheelhouse `~/dynamo_repro/wheelhouse-vllm-1.4.1-lifecycle-m2-terminal-fix-aarch64/`. |
| Rejected request | Unknown-model 404 exported a `request.lifecycle` root with `terminal.outcome=rejected`, `terminal.error=true`: `~/scratch/dynamo_repro/completed_runs/lifecycle_m2_terminal_matrix_fix_20260827_175000/logs/otel-traces.jsonl`. |
| Timed-out request | Job `1968118`, `COMPLETED (0:0)`. A paused request produced `timed_out` after 5.00 s: `~/scratch/dynamo_repro/completed_runs/lifecycle_m2_timeout_root_20260827_180600/logs/otel-traces.jsonl`. |
| Streaming backend error | Job `1968103`, `COMPLETED (0:0)`. The worker was killed after visible SSE data; the client received a structured error plus `[DONE]`, and the root was `failed`: `~/scratch/dynamo_repro/completed_runs/lifecycle_m2_stream_error_root_20260827_180200/logs/otel-traces.jsonl`. |
| Client cancellation | Earlier focused mock validation exported `cancelled`: `~/scratch/dynamo_repro/completed_runs/lifecycle_m2_cancel_fix_20260827_002013/logs/otel-traces.jsonl`. |

The fault-injection harness is `launch/lifecycle-runtime-mock.slurm`. Select
its independent probes with `LIFECYCLE_TEST_CASES=rejected`, `timed_out`,
`stream_error`, or `cancelled`. It stages logs on both success and injected
failure, then waits for frontend and Collector OTLP batches to flush before
teardown. Run timeout and worker-kill probes in separate mock allocations:
each intentionally makes the sole mock worker unavailable.

M2 commits on the pushed branch:

```text
b792535a6b chore(dev): add lifecycle terminal fault injection
7b9b55eca3 feat(runtime): add M2 lifecycle identity and terminal outcomes
```

## Development launch support

The following development-only launchers exist in the worktree:

| File | Purpose |
| --- | --- |
| `launch/lifecycle-runtime-mock.slurm` | Iterative frontend-only mock; it intentionally does not replace the real P/D workload. |
| `launch/otel-jsonl-collector.slurm` | Reachable OTLP/gRPC Collector Contrib sink that stages trace JSONL to the completed-run directory. |
| `launch/vllm-qwen36-pd-role-dp.slurm` | One P or D Dynamo worker backed by a configurable local vLLM data-parallel group. |
| `launch/vllm-qwen36-frontend-client-lifecycle-dev.slurm` | Development copy of the canonical frontend/client launcher; only change is `--force-reinstall` for the override wheels. |
| `launch/vllm-qwen36-pd-role-dp-lifecycle-dev.slurm` | Development copy of the DP worker launcher; only change is `--force-reinstall` for the override wheels. |

Force reinstall is required because the rebuilt wheels deliberately retain the
base image version `1.4.1`; without it, `pip` reports the wheel as already
installed and runs the stock runtime instead of the instrumentation.

## Validated build and real workload

The first ARM build identified and corrected a Rust type mismatch in the new
role selector. The successful replacement build was:

| Item | Value |
| --- | --- |
| Build job | `1947280` |
| Result | `COMPLETED`, exit `0:0`, elapsed `17m45s` |
| Builder | `g242-p33-0102` (`a100x2_aarch64`, IPP ARM) |
| Wheelhouse | `~/dynamo_repro/wheelhouse-lifecycle-vllm-v141-engine-operation-20260825-1235/` |

The validated real workload used a reachable JSONL collector and the
disaggregated Qwen 3.6 vLLM topology:

| Role | Job | Node | Configuration |
| --- | ---: | --- | --- |
| JSONL collector | `1947294` | `gh-nvl-203-compute06` | Completed cleanly. |
| Frontend/client | `1947482` | `gh-nvl-203-compute08` | One frontend GPU; cancelled only after client completion because of the known teardown wait. |
| Prefill | `1947483` | `gh-nvl-203-compute09` | Two GPUs, `DATA_PARALLEL_SIZE=2`; cancelled only after client completion because of the known teardown wait. |
| Decode | `1947484` | `gh-nvl-203-compute10` | Two GPUs, `DATA_PARALLEL_SIZE=2`; worker teardown exited nonzero after the client completed. |

Workload parameters:

```text
model: Qwen/Qwen3.6-35B-A3B-FP8
requests: 512
concurrency: 128
synthetic input tokens mean: 64
output tokens mean: 16
```

The client completed all 512 requests in 395.76 seconds. The jobs' final
teardown states are not a workload failure: the Collector had already flushed
and staged the trace artifact before the known frontend/prefill cleanup hang
was cancelled.

## Retained trace evidence

Trace artifact:

```text
~/scratch/dynamo_repro/completed_runs/
  engine_operation_dp2_jsonl_20260825_001740/
    logs/otel-traces.jsonl
```

The file is 17 MB and contains 188 OTLP JSONL batches. Each line is an OTLP
export object, not one flattened span. Span counts below are derived from
timestamps and parentage in the file:

| Span | Count |
| --- | ---: |
| `request.lifecycle` | 512 |
| `router.selection` | 1,024 |
| `worker.admission` | 1,024 |
| `worker.operation.prefill` | 512 |
| `worker.operation.decode` | 512 |
| `response.streaming.prefill` | 512 |
| `response.streaming.decode` | 512 |
| frontend `response.streaming` | 512 |

Timing-only analysis uses a within-run cohort and evaluates overlapping spans as
a timeline, never by summing children. The main observed timing results are:

| Boundary | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| `request.lifecycle` | 88.435 s | 130.431 s | 135.036 s |
| `worker.operation.prefill` | 383.444 ms | 11.514 s | 11.522 s |
| Prefill end → decode start | 115.141 ms | 187.557 ms | 199.845 ms |
| `worker.operation.decode` | 87.944 s | 121.326 s | 134.645 s |
| frontend `response.streaming` | 87.924 s | 121.286 s | 134.645 s |

For the 26 requests at or above the within-run lifecycle p95, the correlation
between request duration and `worker.operation.decode` is 0.988. This localizes the tail
to the observed decode-operation boundary, but does not prove decode compute is
the root cause because the boundary still includes uninstrumented backend work.

## Current proposal coverage and gaps

| Proposal property | Current status | Reason |
| --- | --- | --- |
| Frontend, routing selection, admission, P/D operation, and streaming timing | Observed | Structural timing coverage is complete for all 512 requests. |
| Router queue dwell | `UNKNOWN` | No `router.queue` span was retained; conditional applicability cannot be inferred from absence alone. |
| Engine queue dwell | `UNKNOWN` | No `engine.queue` span was retained. |
| Observed KV-transfer duration | `UNKNOWN` | No `kv.transfer` span was retained. vLLM metrics are not a replacement for request-correlated timing. |
| Lifecycle schema, identity, and terminal outcome | Observed | M2 exports versioned typed root attributes and one-shot outcomes. |
| Explicit P→D operation parent/link | Not yet implemented | M3 must propagate causality; request identity alone is not a parent relation. |
| Cohort residual or `VIOLATED` decision | `UNKNOWN` | Only one workload cohort exists; there is no versioned matched baseline. |
| Fine router subspans | Partial | One second-occurrence `kv_router.compute_seq_hashes` span is absent (511/512); this is auxiliary evidence, not a registered lifecycle stage. |

## Next implementation steps

1. Preserve this timing-only trace as the regression artifact for the current
   worker-operation milestone.
2. Add authoritative backend signals for `engine.queue`, `engine.prefill`,
   `engine.decode`, and observed NIXL `kv.transfer` completion. Do not infer
   them from metrics or from missing spans.
3. Add M3 operation causality: propagate parent operation identity and/or OTel
   links between prefill and decode, without overloading `request_id`.
4. Add bounded router facts and a backend capability table; retain `UNKNOWN`
   where direct backend evidence is unavailable.
5. Implement coverage and state-machine validation, including an explicit
   `UNKNOWN` result for sampled, dropped, unsupported, or conditionally absent
   evidence.
6. Build matched workload cohorts before assigning `VIOLATED` / `HOLDS` latency
   residuals; this single run only supports within-run timing localization.

## Repeat the validated real run

Use a fresh `RUN_ID`, submit the collector first, read its endpoint marker, and
pass the trace exporter variables to all three roles. Keep worker group size at
two GPUs while exposing one Dynamo worker per role:

```text
DATA_PARALLEL_SIZE=2
READY_WORKERS=1
CLIENT_REQUESTS=512
CLIENT_CONCURRENCY=128
DYN_LIFECYCLE_TRACE_ENABLED=true
OTEL_EXPORT_ENABLED=true
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
OTEL_TRACES_SAMPLE_RATIO=1.0
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://<collector-ip>:4317
```

Run the development-only force-reinstall frontend and DP launchers above. After
the client exits, verify the staged `otel-traces.jsonl` before cancelling any
known teardown hang.
