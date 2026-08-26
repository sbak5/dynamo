# Lifecycle instrumentation implementation progress

Last updated: 2026-08-25

This is the working implementation record for native, timing-only OpenTelemetry
lifecycle spans in the Dynamo v1.4.1 runtime. It complements:

- `LIFECYCLE_INTERACTIVE_DEVELOPMENT.md` — fresh-session setup and iterative mock.
- `LIFECYCLE_SPAN_DEVELOPMENT_RUNBOOK.md` — broader lifecycle-span operating notes.
- `/home/scratch.sbak_coreai/dynamo-observe/research/model-based-diagnosis/PROPOSAL-final-candidate.md` — target contract and milestones.

## Current scope

The implementation is deliberately limited to timing boundaries. It does not yet
add the proposal's typed lifecycle attributes, terminal outcomes, queue-depth
facts, token counts, KV byte counts, or anomaly thresholds.

- Branch: `sbak/lifecycle-runtime-instrumentation`
- Development worktree: `~/scratch/dynamo-lifecycle-dev`
- Clean build checkout: `~/scratch/dynamo`
- Runtime/build image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1`
- Feature gate: `DYN_LIFECYCLE_TRACE_ENABLED=true`

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
| Terminal outcome and finish reason | `UNKNOWN` | This timing-only stage adds no terminal schema/status. |
| Cohort residual or `VIOLATED` decision | `UNKNOWN` | Only one workload cohort exists; there is no versioned matched baseline. |
| Fine router subspans | Partial | One second-occurrence `kv_router.compute_seq_hashes` span is absent (511/512); this is auxiliary evidence, not a registered lifecycle stage. |

## Next implementation steps

1. Preserve this timing-only trace as the regression artifact for the current
   worker-operation milestone.
2. Add authoritative backend signals for `engine.queue`, `engine.prefill`,
   `engine.decode`, and observed NIXL `kv.transfer` completion. Do not infer
   them from metrics or from missing spans.
3. Add the proposal's typed schema: lifecycle profile/mode, operation identity,
   terminal outcome, bounded router facts, and backend capability table.
4. Implement coverage and state-machine validation, including an explicit
   `UNKNOWN` result for sampled, dropped, unsupported, or conditionally absent
   evidence.
5. Build matched workload cohorts before assigning `VIOLATED` / `HOLDS` latency
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
