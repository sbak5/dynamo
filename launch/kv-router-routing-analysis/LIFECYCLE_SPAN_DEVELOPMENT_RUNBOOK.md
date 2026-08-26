# Lifecycle Span Development Runbook

This is the operating record for timing-only OpenTelemetry instrumentation in
Dynamo v1.4.1. It supplements the shorter fresh-session guide in
`LIFECYCLE_INTERACTIVE_DEVELOPMENT.md`.

## Scope

- Development branch: `sbak/lifecycle-runtime-instrumentation`
- Isolated worktree: `~/scratch/dynamo-lifecycle-dev`
- Clean build checkout: `~/scratch/dynamo`
- Runtime image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1`
- Current milestone: timing-only spans. Do not add detailed product metrics
  or payload attributes.

## Implemented spans

Child spans overlap with their parents; their durations must not be summed.

| Stage | Span | Owner |
| --- | --- | --- |
| Request root | `request.lifecycle` | Rust frontend HTTP service |
| Preprocessing | `request.preprocessing` | Rust preprocessor |
| Router wait | `router.queue` | Rust KV-router scheduler |
| Router decision | `router.selection` | Rust KV-router scheduler |
| Backend receive | `worker.admission` | Rust runtime ingress |
| Backend dispatch | `request.dispatch` | Rust runtime ingress |
| Engine wrapper | `engine.generate` | Rust backend adapter |
| First engine output | `engine.queue` | Python vLLM handler via Rust OTEL context |
| Worker operation | `worker.operation.prefill` and `worker.operation.decode` | Rust runtime / backend adapter |
| Remote KV handoff | `kv.transfer` | Python vLLM decode handler |
| Output lifetime | `response.streaming` | Rust runtime ingress |

`engine.queue` ends at first engine output. In a remote-prefill decode request,
`kv.transfer` also ends at first output after the NIXL handoff.

## Changes made

The implementation is based on v1.4.1 commit `2112d6ba74`.

- `64496b576b`: lifecycle span registry and feature gate.
- `86d353acd6`: worker admission, dispatch, and response-streaming spans.
- `d2eae92d99`: frontend, preprocessing, and engine-stage spans.
- `6731e6b57d`: router queue and selection spans.
- `3cc824f796`: vLLM `engine.queue` and `kv.transfer` spans plus tests.
- `bd796ed071`: initial direct-Tempo mock collector.
- `7c5dae6129`: JSONL file collector replacing Tempo.

The main implementation files are:

```text
lib/runtime/src/telemetry.rs
lib/runtime/src/pipeline/network/ingress/push_handler.rs
lib/llm/src/http/service/openai.rs
lib/llm/src/preprocessor.rs
lib/backend-common/src/adapter.rs
lib/kv-router/src/scheduling/queue.rs
components/src/dynamo/common/backend/telemetry.py
components/src/dynamo/vllm/handlers.py
```

## Build override wheels

Build the isolated worktree with the maintained builder from the clean checkout.

```bash
BUILD_REPO=~/scratch/dynamo
DEV_REPO=~/scratch/dynamo-lifecycle-dev
WHEELHOUSE_NAME=lifecycle-runtime-v141-$(date +%Y%m%d-%H%M%S)

REPO_DIR="$DEV_REPO" \
IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1 \
WHEELHOUSE_NAME="$WHEELHOUSE_NAME" \
sbatch --nodelist=gh-nvl-203-compute02 \
  "$BUILD_REPO/launch/kv-router-routing-analysis/launch/build-qwen36-aarch64-wheel.slurm"
```

Verify it completed and contains both wheel files:

```bash
sacct -j <build-job-id> --format=JobID,State,ExitCode,Elapsed -X
ls -lh "$HOME/dynamo_repro/wheelhouse-$WHEELHOUSE_NAME/"
```

Validated build: `1935412`, wheelhouse
`lifecycle-engine-kv-v141-20260824-2205`.

## Export spans as JSONL

The frontend-only mock starts an OpenTelemetry Collector Contrib process locally.
It receives OTLP gRPC on port 4317 and writes:

```text
logs/otel-traces.jsonl
logs/otel-logs.jsonl
```

Each line in `otel-traces.jsonl` is an OTLP export batch in JSON, and may
contain several spans. It is not a one-span-per-line format.

The launcher expects the AArch64 collector binary at:

```text
~/dynamo_repro/tools/otelcol-contrib-0.122.1-linux-arm64/otelcol-contrib
```

Install it once if necessary:

```bash
TOOL_DIR="$HOME/dynamo_repro/tools/otelcol-contrib-0.122.1-linux-arm64"
mkdir -p "$TOOL_DIR"
curl -fL -o /tmp/otelcol-contrib.tar.gz \
  https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.122.1/otelcol-contrib_0.122.1_linux_arm64.tar.gz
tar -xzf /tmp/otelcol-contrib.tar.gz -C "$TOOL_DIR"
```

The collector configuration is
`launch/kv-router-routing-analysis/launch/lifecycle-otel-jsonl.yaml`.

## Run the frontend-only mock

The mock starts etcd, NATS, one frontend, one `dynamo.mocker` worker, and the
JSONL collector. It intentionally does not start vLLM prefill or decode roles.

```bash
cd ~/scratch/dynamo-lifecycle-dev
RUN_ID=lifecycle_jsonl_v141_$(date +%Y%m%d_%H%M%S)

RUN_ID="$RUN_ID" \
IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1 \
WHEELHOUSE_NAME="$WHEELHOUSE_NAME" \
DYN_LIFECYCLE_TRACE_ENABLED=true \
sbatch --output="$HOME/dynamo_repro/$RUN_ID.slurm.out" \
  --nodelist=gh-nvl-203-compute02 \
  launch/kv-router-routing-analysis/launch/lifecycle-runtime-mock.slurm
```

The launcher provides these export settings:

```text
OTEL_EXPORT_ENABLED=true
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
OTEL_TRACES_SAMPLE_RATIO=1.0
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://<allocated-node-ip>:4317
```

Validate the result:

```bash
RUN_DIR="$HOME/dynamo_repro/lifecycle_mock_runs/$RUN_ID"
sacct -j <mock-job-id> --format=JobID,State,ExitCode,Elapsed -X
sed -n '1,4p' "$RUN_DIR/logs/response.jsonl"
grep -aoE 'worker\.admission|request\.dispatch|response\.streaming' \
  "$RUN_DIR/logs/otel-traces.jsonl" | sort -u
```

The entire run is staged at:

```text
~/scratch/dynamo_repro/completed_runs/<run-id>/
```

Validated JSONL run: `lifecycle_jsonl_v141_20260824_231655`, Slurm job
`1936093`, completed successfully. Its trace JSONL contains
`worker.admission`, `request.dispatch`, and `response.streaming`.

## Validate the real vLLM P/D pipeline

The mock cannot generate P/D `worker.operation.*`, `engine.queue`, or
`kv.transfer` spans. Use the unmodified real launcher from
the clean checkout for that validation:

```bash
cd ~/scratch/dynamo/launch/kv-router-routing-analysis/launch
RUN_ID=lifecycle_engine_kv_v141_$(date +%Y%m%d_%H%M%S) \
FRONTEND_PARTITION=gh200 WORKER_PARTITION=gh200 \
FRONTEND_NODE=gh-nvl-203-compute01 \
PREFILL_NODE=gh-nvl-203-compute03 DECODE_NODE=gh-nvl-203-compute04 \
WHEELHOUSE_NAME="$WHEELHOUSE_NAME" \
CLIENT_REQUESTS=1 CLIENT_CONCURRENCY=1 \
DYN_LIFECYCLE_TRACE_ENABLED=true \
OTEL_EXPORT_ENABLED=true \
./submit-vllm-qwen36-three-job.sh
```

For JSONL in the three-job topology, run a collector on a host reachable from
frontend, prefill, and decode, and point
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` at it before submission. The mock's
collector is local to its one-node allocation and is not shared with P/D jobs.

Earlier P/D validation sent one successful request with zero AIPerf errors.
Frontend `1935555`, prefill `1935556`, and decode `1935557` were used.
Those launchers have a known teardown issue that can leave frontend and prefill
jobs allocated after a successful request.

## Next milestone

Run the real P/D topology with the JSONL collector reachable by every role,
then confirm a trace contains the intended timing-only engine and KV spans.

Before ending a session, check `git status --short --branch`, commit
intentional changes with DCO sign-off, and record job IDs in this file.
