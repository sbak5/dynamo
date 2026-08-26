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
  --speedup-ratio 100 --request-plane tcp >"$STAGING_LOGDIR/mocker.log" 2>&1 &
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
echo "[lifecycle-mock] request completed; logs: $LOGDIR"
INNER

sleep "${OTEL_FLUSH_DELAY_SECS:-3}"
echo "[lifecycle-mock] OTLP trace JSONL staged at $OTEL_TRACES_FILE"
