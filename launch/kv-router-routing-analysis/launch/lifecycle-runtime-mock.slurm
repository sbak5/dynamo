#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Small iterative lifecycle-tracing smoke test. This intentionally does not use
# prefill/decode roles, SGLang, or vLLM inference. One aggregated dynamo.mocker
# worker runs beside the frontend and supplies a real request-plane backend.
#
# Submit from ~/scratch/dynamo, optionally pinning an AArch64/IP P node:
#   sbatch --nodelist=gh-nvl-203-compute02 \
#     launch/kv-router-routing-analysis/launch/lifecycle-runtime-mock.slurm
#
# Override WHEELHOUSE_NAME after an incremental wheel build. The default matches
# the vLLM 1.4.1 builder wheelhouse.
#SBATCH --job-name=dynamo-lifecycle-mock
#SBATCH --partition=gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1}"
WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-vllm-1.4.1-aarch64}"
RUN_ID="${RUN_ID:-lifecycle_mock_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-lifecycle-mocker}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
# Comma-separated, opt-in terminal-path probes: `rejected`, `timed_out`,
# `stream_error`, and/or `cancelled`. The legacy `failed` selector aliases
# `rejected` for compatibility with the original mock probe.
# The normal successful streaming request always runs first.
LIFECYCLE_TEST_CASES="${LIFECYCLE_TEST_CASES:-}"
LIFECYCLE_CANCEL_MAX_TIME_SECS="${LIFECYCLE_CANCEL_MAX_TIME_SECS:-0.25}"
MOCKER_SPEEDUP_RATIO="${MOCKER_SPEEDUP_RATIO:-100}"
FRONTEND_OTEL_FLUSH_DELAY_SECS="${FRONTEND_OTEL_FLUSH_DELAY_SECS:-6}"
HOST_HOME="$HOME"
RUN_DIR="$HOST_HOME/dynamo_repro/lifecycle_mock_runs/$RUN_ID"
LOGDIR="$RUN_DIR/logs"
OTELCOL_BIN="${OTELCOL_BIN:-$HOST_HOME/dynamo_repro/tools/otelcol-contrib-0.122.1-linux-arm64/otelcol-contrib}"
OTELCOL_TEMPLATE="${OTELCOL_TEMPLATE:-$HOST_HOME/scratch/dynamo-lifecycle-dev/launch/kv-router-routing-analysis/launch/lifecycle-otel-jsonl.yaml}"
OTELCOL_CONFIG="$RUN_DIR/otelcol.yaml"
OTEL_TRACES_FILE="$LOGDIR/otel-traces.jsonl"
OTEL_LOGS_FILE="$LOGDIR/otel-logs.jsonl"
mkdir -p "$LOGDIR"

export IMAGE WHEELHOUSE_NAME RUN_ID MODEL_NAME MODEL_PATH HOST_HOME RUN_DIR LOGDIR
export LIFECYCLE_TEST_CASES LIFECYCLE_CANCEL_MAX_TIME_SECS MOCKER_SPEEDUP_RATIO
export FRONTEND_OTEL_FLUSH_DELAY_SECS
export DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS="${DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS:-}"
# The `timed_out` probe pauses a worker before its first response, exercising
# the request-plane timeout rather than the post-SSE inactivity monitor.
export DYN_TCP_REQUEST_TIMEOUT="${DYN_TCP_REQUEST_TIMEOUT:-5}"
export DYN_LIFECYCLE_TRACE_ENABLED="${DYN_LIFECYCLE_TRACE_ENABLED:-true}"
export DYN_LOGGING_SPAN_EVENTS="${DYN_LOGGING_SPAN_EVENTS:-true}"
export OTEL_EXPORT_ENABLED="${OTEL_EXPORT_ENABLED:-true}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="${OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:-grpc}"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE:-true}"
export OTEL_TRACES_SAMPLE_RATIO="${OTEL_TRACES_SAMPLE_RATIO:-1.0}"

stage_logs() {
  local dest="$HOST_HOME/scratch/dynamo_repro/completed_runs/$RUN_ID"
  mkdir -p "$dest"
  rsync -a "$RUN_DIR/" "$dest/" || true
  printf 'run_id=%s\nstaged_at_utc=%s\nsource=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" "$RUN_DIR" > "$dest/STAGED.txt"
}

cleanup() {
  local status=$?
  if [ -n "${OTELCOL_PID:-}" ]; then
    kill "$OTELCOL_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$OTELCOL_PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$OTELCOL_PID" 2>/dev/null && kill -KILL "$OTELCOL_PID" 2>/dev/null || true
    wait "$OTELCOL_PID" 2>/dev/null || true
  fi
  stage_logs
  exit "$status"
}
trap cleanup EXIT INT TERM

INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
[ -n "$INFRA_IP" ] || { echo "could not resolve node IP" >&2; exit 2; }
[ -x "$OTELCOL_BIN" ] || { echo "missing OTel Collector binary: $OTELCOL_BIN" >&2; exit 2; }
[ -f "$OTELCOL_TEMPLATE" ] || { echo "missing OTel Collector template: $OTELCOL_TEMPLATE" >&2; exit 2; }
sed -e "s|__OTEL_TRACES_FILE__|$OTEL_TRACES_FILE|g" -e "s|__OTEL_LOGS_FILE__|$OTEL_LOGS_FILE|g" "$OTELCOL_TEMPLATE" > "$OTELCOL_CONFIG"
"$OTELCOL_BIN" --config="$OTELCOL_CONFIG" > "$LOGDIR/otelcol.log" 2>&1 &
OTELCOL_PID=$!
until curl -sf http://127.0.0.1:13133/ >/dev/null 2>&1; do
  kill -0 "$OTELCOL_PID" 2>/dev/null || { cat "$LOGDIR/otelcol.log" >&2; exit 1; }
  sleep 1
done
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://$INFRA_IP:4317"
echo "[lifecycle-mock] OTel Collector pid=$OTELCOL_PID otlp=$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT traces=$OTEL_TRACES_FILE"

# Preserve collector flush/staging even when an injected fault deliberately
# makes the inner test fail. The final exit status still reflects that test.
set +e
srun --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="$HOST_HOME:$HOST_HOME" --container-remap-root --export=ALL \
  bash -s <<'INNER'
set -euo pipefail

STAGING_LOGDIR="/tmp/dynamo-lifecycle-logs-$SLURM_JOB_ID"
mkdir -p "$STAGING_LOGDIR"

wheelhouse="$HOST_HOME/dynamo_repro/wheelhouse-$WHEELHOUSE_NAME"
[ -d "$wheelhouse" ] || { echo "missing wheelhouse: $wheelhouse" >&2; exit 2; }
pip install --break-system-packages --force-reinstall --no-deps \
  "$wheelhouse"/ai_dynamo_runtime*.whl "$wheelhouse"/ai_dynamo-*any.whl

cleanup() {
  status=$?
  for pid in "${MOCKER_PID:-}" "${FRONTEND_PID:-}" "${NATS_PID:-}" "${ETCD_PID:-}"; do
    [ -z "$pid" ] || kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  cp -a "$STAGING_LOGDIR"/. "$LOGDIR"/ 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

export ETCD_ENDPOINTS="http://127.0.0.1:2379"
export NATS_SERVER="nats://127.0.0.1:4222"
export DYN_LIFECYCLE_TRACE_ENABLED DYN_LOGGING_SPAN_EVENTS
export DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS
export DYN_TCP_REQUEST_TIMEOUT
export OTEL_EXPORT_ENABLED OTEL_EXPORTER_OTLP_TRACES_ENDPOINT OTEL_EXPORTER_OTLP_TRACES_PROTOCOL
export OTEL_EXPORTER_OTLP_TRACES_INSECURE OTEL_TRACES_SAMPLE_RATIO

etcd --data-dir "/tmp/dynamo-lifecycle-etcd-$SLURM_JOB_ID" \
  --listen-client-urls http://127.0.0.1:2379 \
  --advertise-client-urls http://127.0.0.1:2379 >"$STAGING_LOGDIR/etcd.log" 2>&1 &
ETCD_PID=$!
nats-server -js -a 127.0.0.1 -p 4222 >"$STAGING_LOGDIR/nats.log" 2>&1 &
NATS_PID=$!

python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin \
  >"$STAGING_LOGDIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

python3 -m dynamo.mocker --model-name "$MODEL_NAME" --model-path "$MODEL_PATH" \
  --speedup-ratio "$MOCKER_SPEEDUP_RATIO" --request-plane tcp >"$STAGING_LOGDIR/mocker.log" 2>&1 &
MOCKER_PID=$!

for _ in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/v1/models | grep -q "$MODEL_NAME"; then
    break
  fi
  kill -0 "$FRONTEND_PID" 2>/dev/null || { cat "$STAGING_LOGDIR/frontend.log" >&2; exit 1; }
  kill -0 "$MOCKER_PID" 2>/dev/null || { cat "$STAGING_LOGDIR/mocker.log" >&2; exit 1; }
  sleep 2
done
curl -sf http://127.0.0.1:8000/v1/models | grep -q "$MODEL_NAME" || {
  echo "mocker did not register within 180 seconds" >&2
  exit 1
}

curl --fail --silent --show-error --no-buffer \
  -H 'content-type: application/json' \
  http://127.0.0.1:8000/v1/chat/completions \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":4,\"stream\":true}" \
  >"$STAGING_LOGDIR/response.jsonl"

grep -q 'data:' "$STAGING_LOGDIR/response.jsonl"

# An unknown model reaches the frontend request lifecycle but is rejected
# before a backend response is delivered. It exercises the `rejected` path.
case ",$LIFECYCLE_TEST_CASES," in
  *,rejected,*|*,failed,*)
    rejection_status=$(curl --silent --show-error --output "$STAGING_LOGDIR/rejected-response.json" \
      --write-out '%{http_code}' \
      -H 'content-type: application/json' \
      http://127.0.0.1:8000/v1/chat/completions \
      -d '{"model":"lifecycle-missing-model","messages":[{"role":"user","content":"fail"}],"max_tokens":4,"stream":false}')
    case "$rejection_status" in
      4*) echo "[lifecycle-mock] injected rejected request: HTTP $rejection_status" ;;
      *) echo "expected injected rejection, got HTTP $rejection_status" >&2; exit 1 ;;
    esac
    ;;
esac

# Stop an already registered worker before issuing a valid request. Its listener
# stays open but produces no first response, so the typed TCP request timeout is
# the authoritative cause of terminal `timed_out`.
case ",$LIFECYCLE_TEST_CASES," in
  *,timed_out,*)
    case "${DYN_TCP_REQUEST_TIMEOUT:-}" in
      ""|0) echo "timed_out probe requires DYN_TCP_REQUEST_TIMEOUT>0" >&2; exit 1 ;;
    esac
    kill -STOP "$MOCKER_PID"
    set +e
    curl --silent --show-error --no-buffer --max-time 8 \
      -H 'content-type: application/json' \
      http://127.0.0.1:8000/v1/chat/completions \
      -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"timeout\"}],\"max_tokens\":4,\"stream\":true}" \
      >"$STAGING_LOGDIR/timed-out-response.jsonl"
    timeout_rc=$?
    set -e
    kill -CONT "$MOCKER_PID"
    [ "$timeout_rc" -ne 28 ] || {
      echo "frontend inactivity timeout did not close the request" >&2
      exit 1
    }
    sleep 1
    echo "[lifecycle-mock] injected timed-out request"
    ;;
esac

# A deliberately short-lived streaming client disconnects while a long mock
# response is in flight. Slow the mock with MOCKER_SPEEDUP_RATIO when needed.
case ",$LIFECYCLE_TEST_CASES," in
  *,cancelled,*)
    set +e
    curl --silent --show-error --no-buffer --max-time "$LIFECYCLE_CANCEL_MAX_TIME_SECS" \
      -H 'content-type: application/json' \
      http://127.0.0.1:8000/v1/chat/completions \
      -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"cancel\"}],\"max_tokens\":4096,\"stream\":true}" \
      >"$STAGING_LOGDIR/cancelled-response.jsonl"
    cancel_rc=$?
    set -e
    # Very fast mockers can complete before the client deadline. Treat that as
    # an inconclusive cancellation probe rather than aborting unrelated
    # terminal-case validation; the cancellation-specific run uses a slower
    # backend configuration.
    if [ "$cancel_rc" -ne 28 ]; then
      echo "[lifecycle-mock] cancellation probe completed before client deadline (curl rc=$cancel_rc)" >&2
    fi
    # Allow the frontend disconnect monitor to observe the closed client.
    sleep 2
    echo "[lifecycle-mock] injected cancelled request"
    ;;
esac

# Kill the worker after the first visible SSE chunk. This preserves HTTP 200 and
# exercises a genuine backend stream error rather than a request-start failure.
case ",$LIFECYCLE_TEST_CASES," in
  *,stream_error,*)
    curl --silent --show-error --no-buffer \
      -H 'content-type: application/json' \
      http://127.0.0.1:8000/v1/chat/completions \
      -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"stream error\"}],\"max_tokens\":4096,\"stream\":true}" \
      >"$STAGING_LOGDIR/stream-error-response.jsonl" &
    stream_error_curl_pid=$!
    for _ in $(seq 1 100); do
      grep -q 'data:' "$STAGING_LOGDIR/stream-error-response.jsonl" 2>/dev/null && break
      kill -0 "$stream_error_curl_pid" 2>/dev/null || break
      sleep 0.05
    done
    grep -q 'data:' "$STAGING_LOGDIR/stream-error-response.jsonl" || {
      echo "stream-error probe did not receive its first chunk" >&2
      exit 1
    }
    kill -KILL "$MOCKER_PID"
    set +e
    wait "$stream_error_curl_pid"
    stream_error_rc=$?
    set -e
    grep -q '"error"' "$STAGING_LOGDIR/stream-error-response.jsonl" || {
      echo "stream-error probe did not receive an SSE error frame (curl rc=$stream_error_rc)" >&2
      exit 1
    }
    sleep 1
    echo "[lifecycle-mock] injected streaming backend error"
    ;;
esac

# Give the frontend's own OTLP batch processor time to export completed roots
# before the inner cleanup sends it SIGTERM. (The outer delay only protects the
# collector and cannot recover spans from an already terminated frontend.)
sleep "$FRONTEND_OTEL_FLUSH_DELAY_SECS"
echo "[lifecycle-mock] request completed; logs: $LOGDIR"
INNER
inner_status=$?
set -e
sleep "${OTEL_FLUSH_DELAY_SECS:-3}"
echo "[lifecycle-mock] OTLP trace JSONL staged at $OTEL_TRACES_FILE"
exit "$inner_status"
