#!/bin/bash
#SBATCH --job-name=qwen36-vllm-frontend-client
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Job 1 of 3: frontend and AIPerf share an ARM+IPP node. Submit this first.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
RUN_ID="${RUN_ID:-qwen36_vllm_$(date +%Y%m%d_%H%M%S)}"
case "$RUN_ID" in *[!A-Za-z0-9_.-]*|'') echo "invalid RUN_ID" >&2; exit 2;; esac
HOST_HOME="$HOME"
# Keep active logs off the quota-limited home filesystem.  `RUNS_ROOT` must
# be shared by the frontend and both worker allocations.
RUNS_ROOT="${RUNS_ROOT:-$HOST_HOME/scratch/dynamo_repro/runs}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
SHARED="$RUN_DIR/shared"
LOGDIR="$RUN_DIR/logs"
WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-vllm-1.4.1-aarch64}"
# Keep development wheels on scratch: the home filesystem is quota-limited,
# while this run root is already mounted by every validation container.
WHEELHOUSE_ROOT="${WHEELHOUSE_ROOT:-$RUNS_ROOT/wheelhouses}"
STAGE_ROOT="${STAGE_ROOT:-$HOME/scratch/dynamo_repro/completed_runs}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-1}"
CLIENT_REQUESTS="${CLIENT_REQUESTS:-4}"
# Empty by default: production-comparable validation continues to run AIPerf.
# An opt-in probe is used only for lifecycle negative-path development.
CLIENT_PROBE_SCRIPT="${CLIENT_PROBE_SCRIPT:-}"
# When set, materialize a bounded one-request-per-worker policy on the shared
# run directory. This is used with CLIENT_PROBE_SCRIPT to exercise scheduler
# queue rejection without changing the ordinary routing workload.
ROUTER_QUEUE_LIMIT_PER_WORKER="${ROUTER_QUEUE_LIMIT_PER_WORKER:-}"
# Slurm copies batch scripts to its spool directory, so `$0` cannot locate
# companion collector/nginx files. The submitter supplies this stable source
# directory for lifecycle development runs.
SCRIPT_DIR="${LIFECYCLE_LAUNCH_DIR:-$HOST_HOME/scratch/dynamo-lifecycle-dev/launch/kv-router-routing-analysis/launch}"
MODEL_SOURCE_IMAGE="${MODEL_SOURCE_IMAGE:-nginx:1.27}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-/home/scratch.sbak_coreai/model-cache/Qwen/Qwen3.6-35B-A3B-FP8}"
MODEL_SOURCE_CONFIG="${MODEL_SOURCE_CONFIG:-$SCRIPT_DIR/nginx-model-source.conf}"
MODEL_SOURCE_PORT="${MODEL_SOURCE_PORT:-18080}"
OTELCOL_BIN="${OTELCOL_BIN:-$HOST_HOME/dynamo_repro/tools/otelcol-contrib-0.122.1-linux-arm64/otelcol-contrib}"
OTELCOL_TEMPLATE="${OTELCOL_TEMPLATE:-$SCRIPT_DIR/lifecycle-otel-jsonl.yaml}"
OTELCOL_CONFIG="$RUN_DIR/otelcol.yaml"
OTEL_TRACES_FILE="$LOGDIR/otel-traces.jsonl"
OTEL_LOGS_FILE="$LOGDIR/otel-logs.jsonl"
DYN_LIFECYCLE_TRACE_ENABLED="${DYN_LIFECYCLE_TRACE_ENABLED:-true}"
DYN_LIFECYCLE_TRACE_MODE="${DYN_LIFECYCLE_TRACE_MODE:-investigation}"
case "${DYN_LIFECYCLE_TRACE_MODE,,}" in core|investigation) ;; *) echo "invalid DYN_LIFECYCLE_TRACE_MODE=$DYN_LIFECYCLE_TRACE_MODE" >&2; exit 2;; esac
model_source_pid=
collector_pid=
mkdir -p "$SHARED" "$LOGDIR"
rm -f "$SHARED"/{infra_ip.txt,frontend_ready,prefill_workers_ready,decode_workers_ready,prefill_workers_done,decode_workers_done,model_source_ready,model_source_endpoint.txt,model_source_files.tsv,otel_collector_ready,otel_collector_endpoint.txt,stop_workers}
if [ -n "$CLIENT_PROBE_SCRIPT" ]; then
  test -f "$CLIENT_PROBE_SCRIPT"
  # $SHARED is under RUNS_ROOT, which every client container explicitly mounts.
  # The worktree itself need not be mounted into a Slurm-spooled client step.
  cp "$CLIENT_PROBE_SCRIPT" "$SHARED/lifecycle-client-probe.sh"
  CLIENT_PROBE_SCRIPT="$SHARED/lifecycle-client-probe.sh"
fi
export IMAGE MODEL RUN_ID HOST_HOME RUNS_ROOT RUN_DIR SHARED LOGDIR WHEELHOUSE_NAME WHEELHOUSE_ROOT STAGE_ROOT CLIENT_CONCURRENCY CLIENT_REQUESTS CLIENT_PROBE_SCRIPT ROUTER_QUEUE_LIMIT_PER_WORKER MODEL_SOURCE_IMAGE MODEL_SOURCE_DIR MODEL_SOURCE_CONFIG MODEL_SOURCE_PORT OTELCOL_BIN OTELCOL_TEMPLATE OTELCOL_CONFIG OTEL_TRACES_FILE OTEL_LOGS_FILE DYN_LIFECYCLE_TRACE_ENABLED DYN_LIFECYCLE_TRACE_MODE

stage_logs() {
  local dest="$STAGE_ROOT/$RUN_ID"
  mkdir -p "$dest"
  rsync -a "$RUN_DIR/" "$dest/" || true
  printf 'run_id=%s\nstaged_at_utc=%s\nsource=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" "$RUN_DIR" > "$dest/STAGED.txt"
}
stop_model_source() {
  if test -n "$model_source_pid"; then
    kill "$model_source_pid" 2>/dev/null || true
    wait "$model_source_pid" 2>/dev/null || true
  fi
}
stop_collector() {
  if test -n "$collector_pid"; then
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
}
trap 'stop_model_source; stop_collector; stage_logs' EXIT

test -f "$MODEL_SOURCE_DIR/config.json"
test -f "$MODEL_SOURCE_CONFIG"
test -x "$OTELCOL_BIN"
test -f "$OTELCOL_TEMPLATE"
find "$MODEL_SOURCE_DIR" -maxdepth 1 -type f -printf '%f\t%s\n' | sort >"$SHARED/model_source_files.tsv"
test -s "$SHARED/model_source_files.tsv"

# Do not terminate the producer early: with `pipefail`, `head -1` can make
# `ip` fail with SIGPIPE before the frontend has written its ready marker.
INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ && !found {split($4, address, "/"); print address[1]; found=1}')
printf '%s\n' "$INFRA_IP" > "$SHARED/infra_ip.txt"
export INFRA_IP
echo "[vllm-frontend-client] run=$RUN_ID node=$SLURMD_NODENAME infra_ip=$INFRA_IP"

sed -e "s|__OTEL_TRACES_FILE__|$OTEL_TRACES_FILE|g" \
    -e "s|__OTEL_LOGS_FILE__|$OTEL_LOGS_FILE|g" \
    "$OTELCOL_TEMPLATE" > "$OTELCOL_CONFIG"
"$OTELCOL_BIN" --config="$OTELCOL_CONFIG" >"$LOGDIR/otelcol.log" 2>&1 &
collector_pid=$!
until curl -sf http://127.0.0.1:13133/ >/dev/null; do
  kill -0 "$collector_pid" 2>/dev/null || { cat "$LOGDIR/otelcol.log" >&2; exit 1; }
  sleep 1
done
OTEL_ENDPOINT="http://$INFRA_IP:4317"
printf '%s\n' "$OTEL_ENDPOINT" > "$SHARED/otel_collector_endpoint.txt"
touch "$SHARED/otel_collector_ready"
export OTEL_ENDPOINT OTEL_EXPORT_ENABLED="${OTEL_EXPORT_ENABLED:-true}" \
  OTEL_EXPORTER_OTLP_PROTOCOL=grpc OTEL_EXPORTER_OTLP_INSECURE=true \
  OTEL_EXPORTER_OTLP_ENDPOINT="$OTEL_ENDPOINT" \
  OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc OTEL_EXPORTER_OTLP_TRACES_INSECURE=true \
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$OTEL_ENDPOINT" \
  OTEL_TRACES_SAMPLE_RATIO="${OTEL_TRACES_SAMPLE_RATIO:-1.0}"

if [ -n "$ROUTER_QUEUE_LIMIT_PER_WORKER" ]; then
  case "$ROUTER_QUEUE_LIMIT_PER_WORKER" in *[!0-9]*|'') echo "invalid ROUTER_QUEUE_LIMIT_PER_WORKER" >&2; exit 2;; esac
  cat >"$SHARED/router-policy.yaml" <<EOF
default_policy_family: lifecycle_fixture
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: lifecycle_fixture
    policy_family: lifecycle_fixture
    cache_bucket: all
    queue_policy: fcfs
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: $ROUTER_QUEUE_LIMIT_PER_WORKER
EOF
  export DYN_ROUTER_POLICY_CONFIG="$SHARED/router-policy.yaml"
fi

srun --overlap --nodes=1 --ntasks=1 --container-image="$MODEL_SOURCE_IMAGE" \
  --container-mounts="$MODEL_SOURCE_DIR:/models:ro,$MODEL_SOURCE_CONFIG:/etc/nginx/nginx.conf:ro" \
  --container-remap-root --export=ALL /usr/sbin/nginx -g 'daemon off;' >"$LOGDIR/model-source-nginx.log" 2>&1 &
model_source_pid=$!
until curl -sf "http://127.0.0.1:$MODEL_SOURCE_PORT/config.json" >/dev/null; do
  kill -0 "$model_source_pid" 2>/dev/null || { cat "$LOGDIR/model-source-nginx.log" >&2; exit 1; }
  sleep 1
done
printf 'http://%s:%s\n' "$INFRA_IP" "$MODEL_SOURCE_PORT" >"$SHARED/model_source_endpoint.txt"
touch "$SHARED/model_source_ready"

srun --overlap --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="$HOST_HOME:$HOST_HOME,$RUNS_ROOT:$RUNS_ROOT,$MODEL_SOURCE_DIR:/models:ro" --container-remap-root --export=ALL \
  bash -s <<'FRONTEND' &
set -euo pipefail
wh="$WHEELHOUSE_ROOT/$WHEELHOUSE_NAME"
pip install --break-system-packages --force-reinstall --no-deps "$wh"/ai_dynamo_runtime*.whl "$wh"/ai_dynamo-*any.whl
export ETCD_ENDPOINTS="$INFRA_IP:2379" NATS_SERVER="nats://$INFRA_IP:4222"
etcd --data-dir "/tmp/etcd-$SLURM_JOB_ID" --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls "http://$INFRA_IP:2379" >"$LOGDIR/etcd.log" 2>&1 & ETCD_PID=$!
nats-server -js -a 0.0.0.0 -p 4222 >"$LOGDIR/nats.log" 2>&1 & NATS_PID=$!
# Workers advertise their run-scoped /raid path, which is intentionally local
# to their node.  Mount the checkpoint here so the discovery watcher can use
# this same-basename directory as its --model-path fallback for model cards.
python -m dynamo.frontend --http-port 8000 --router-mode "${ROUTER_MODE:-kv}" --enforce-disagg --model-path /models >"$LOGDIR/frontend.log" 2>&1 & FRONTEND_PID=$!
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do kill -0 "$FRONTEND_PID" 2>/dev/null || exit 1; sleep 2; done
touch "$SHARED/frontend_ready"
while [ ! -e "$SHARED/stop_workers" ]; do kill -0 "$FRONTEND_PID" 2>/dev/null || exit 1; sleep 3; done
# Keep discovery services alive until both worker jobs have completed their
# graceful shutdown; otherwise their lease teardown sees a spurious etcd loss.
for marker in "$SHARED/prefill_workers_done" "$SHARED/decode_workers_done"; do
  until [ -e "$marker" ]; do sleep 1; done
done
kill "$FRONTEND_PID" "$ETCD_PID" "$NATS_PID" 2>/dev/null || true
wait 2>/dev/null || true
FRONTEND
frontend_pid=$!

until [ -e "$SHARED/frontend_ready" ]; do kill -0 "$frontend_pid" 2>/dev/null || { wait "$frontend_pid"; exit 1; }; sleep 2; done

set +e
srun --overlap --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="$HOST_HOME:$HOST_HOME,$RUNS_ROOT:$RUNS_ROOT" --container-remap-root --export=ALL \
  bash -s <<'CLIENT'
set -euo pipefail
python -m pip install --break-system-packages --quiet --target /tmp/aiperf-site "aiperf==${AIPERF_VERSION:-0.11.0}"
aiperf() { PYTHONPATH=/tmp/aiperf-site python -m aiperf "$@"; }
until [ -e "$SHARED/prefill_workers_ready" ] && [ -e "$SHARED/decode_workers_ready" ]; do sleep 2; done
until curl -sf "http://$INFRA_IP:8000/v1/models" | grep -q "$MODEL"; do sleep 3; done
mkdir -p "$LOGDIR/aiperf"
if [ -n "$CLIENT_PROBE_SCRIPT" ]; then
  test -f "$CLIENT_PROBE_SCRIPT"
  bash "$CLIENT_PROBE_SCRIPT" >"$LOGDIR/client.log" 2>&1
else
  aiperf profile --model "$MODEL" --url "http://$INFRA_IP:8000" --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" --request-count "$CLIENT_REQUESTS" --synthetic-input-tokens-mean 64 --output-tokens-mean 16 --artifact-dir "$LOGDIR/aiperf" >"$LOGDIR/client.log" 2>&1
fi
CLIENT
client_status=$?
set -e
touch "$SHARED/stop_workers"
for marker in "$SHARED/prefill_workers_done" "$SHARED/decode_workers_done"; do until [ -e "$marker" ]; do sleep 2; done; done
wait "$frontend_pid"; frontend_status=$?
echo "[vllm-frontend-client] logs staged at $STAGE_ROOT/$RUN_ID"
[ "$client_status" -eq 0 ] && [ "$frontend_status" -eq 0 ]
