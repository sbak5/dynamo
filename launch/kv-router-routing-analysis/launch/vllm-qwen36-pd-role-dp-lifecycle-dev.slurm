#!/bin/bash
#SBATCH --job-name=qwen36-vllm-role
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# DATA_PARALLEL_SIZE=2 requires both GH200 GPUs allocated to this role.
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00

# One vLLM P/D role node. DATA_PARALLEL_SIZE local workers share its GPUs.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
RUN_ID="${RUN_ID:?RUN_ID must match frontend job}"
case "$RUN_ID" in *[!A-Za-z0-9_.-]*|'') echo "invalid RUN_ID" >&2; exit 2;; esac
ROLE="${ROLE:?ROLE must be prefill or decode}"
case "$ROLE" in prefill|decode) ;; *) echo "invalid ROLE=$ROLE" >&2; exit 2;; esac
HOST_HOME="$HOME"
# Match the frontend's shared-scratch run root; worker-local model copies
# remain under /raid and are removed by the cleanup trap below.
RUNS_ROOT="${RUNS_ROOT:-$HOST_HOME/scratch/dynamo_repro/runs}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
SHARED="$RUN_DIR/shared"
LOGDIR="$RUN_DIR/logs"
WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-vllm-1.4.1-aarch64}"
# Keep development wheels under the shared run root, which is mounted in the
# worker container and avoids the quota-limited home filesystem.
WHEELHOUSE_ROOT="${WHEELHOUSE_ROOT:-$RUNS_ROOT/wheelhouses}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
case "$DATA_PARALLEL_SIZE" in *[!0-9]*|'') echo "invalid DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE" >&2; exit 2;; esac
# Count top-level Dynamo workers here, not vLLM's internal data-parallel
# replicas. A single Dynamo worker with DATA_PARALLEL_SIZE=2 still emits one
# `VllmWorker ... initialized` line, so keep READY_WORKERS=1 for that layout.
READY_WORKERS="${READY_WORKERS:-1}"
case "$READY_WORKERS" in *[!0-9]*|'') echo "invalid READY_WORKERS=$READY_WORKERS" >&2; exit 2;; esac
MODEL_STAGE_ROOT="${MODEL_STAGE_ROOT:-/raid/tmp/dynamo-model-cache-${USER}}"
MODEL_STAGE_PARALLELISM="${MODEL_STAGE_PARALLELISM:-8}"
case "$MODEL_STAGE_PARALLELISM" in *[!0-9]*|'') echo "invalid MODEL_STAGE_PARALLELISM=$MODEL_STAGE_PARALLELISM" >&2; exit 2;; esac
MODEL_NAME="${MODEL##*/}"
LOCAL_MODEL_RUN_DIR="$MODEL_STAGE_ROOT/$RUN_ID"
LOCAL_MODEL_DIR="$LOCAL_MODEL_RUN_DIR/$MODEL_NAME"
DYN_LIFECYCLE_TRACE_ENABLED="${DYN_LIFECYCLE_TRACE_ENABLED:-true}"
DYN_LIFECYCLE_TRACE_MODE="${DYN_LIFECYCLE_TRACE_MODE:-investigation}"
case "${DYN_LIFECYCLE_TRACE_MODE,,}" in core|investigation) ;; *) echo "invalid DYN_LIFECYCLE_TRACE_MODE=$DYN_LIFECYCLE_TRACE_MODE" >&2; exit 2;; esac
export IMAGE MODEL RUN_ID ROLE HOST_HOME RUNS_ROOT RUN_DIR SHARED LOGDIR WHEELHOUSE_NAME WHEELHOUSE_ROOT GPU_MEMORY_UTILIZATION MAX_MODEL_LEN DATA_PARALLEL_SIZE READY_WORKERS MODEL_STAGE_ROOT MODEL_STAGE_PARALLELISM MODEL_NAME LOCAL_MODEL_RUN_DIR LOCAL_MODEL_DIR DYN_LIFECYCLE_TRACE_ENABLED DYN_LIFECYCLE_TRACE_MODE

until [ -s "$SHARED/infra_ip.txt" ]; do sleep 2; done
INFRA_IP=$(cat "$SHARED/infra_ip.txt")
until [ -e "$SHARED/model_source_ready" ] && [ -s "$SHARED/model_source_endpoint.txt" ] && [ -s "$SHARED/model_source_files.tsv" ]; do sleep 2; done
MODEL_SOURCE_ENDPOINT=$(cat "$SHARED/model_source_endpoint.txt")
until [ -e "$SHARED/otel_collector_ready" ] && [ -s "$SHARED/otel_collector_endpoint.txt" ]; do sleep 2; done
OTEL_ENDPOINT=$(cat "$SHARED/otel_collector_endpoint.txt")
export INFRA_IP MODEL_SOURCE_ENDPOINT OTEL_ENDPOINT ETCD_ENDPOINTS="$INFRA_IP:2379" NATS_SERVER="nats://$INFRA_IP:4222" \
  OTEL_EXPORT_ENABLED="${OTEL_EXPORT_ENABLED:-true}" \
  OTEL_EXPORTER_OTLP_PROTOCOL=grpc OTEL_EXPORTER_OTLP_INSECURE=true \
  OTEL_EXPORTER_OTLP_ENDPOINT="$OTEL_ENDPOINT" \
  OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc OTEL_EXPORTER_OTLP_TRACES_INSECURE=true \
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$OTEL_ENDPOINT" \
  OTEL_TRACES_SAMPLE_RATIO="${OTEL_TRACES_SAMPLE_RATIO:-1.0}"

set +e
srun --container-image="$IMAGE" --container-mounts="$HOST_HOME:$HOST_HOME,$RUNS_ROOT:$RUNS_ROOT,/raid:/raid" \
  --container-remap-root --export=ALL bash -s <<'INNER'
set -euo pipefail
wh="$WHEELHOUSE_ROOT/$WHEELHOUSE_NAME"
pip install --break-system-packages --force-reinstall --no-deps "$wh"/ai_dynamo_runtime*.whl "$wh"/ai_dynamo-*any.whl
cleanup_local_model() {
  status=$?
  case "$LOCAL_MODEL_RUN_DIR" in
    /raid/tmp/dynamo-model-cache-*/*) rm -rf -- "$LOCAL_MODEL_RUN_DIR" ;;
    *) echo "refusing unsafe local-model cleanup path: $LOCAL_MODEL_RUN_DIR" >&2 ;;
  esac
  exit "$status"
}
trap cleanup_local_model EXIT INT TERM
mkdir -p "$LOCAL_MODEL_DIR"
mapfile -t model_urls < <(awk -F '\t' -v endpoint="$MODEL_SOURCE_ENDPOINT" '{print endpoint "/" $1}' "$SHARED/model_source_files.tsv")
curl --fail --location --retry 5 --retry-all-errors --continue-at - \
  --parallel --parallel-max "$MODEL_STAGE_PARALLELISM" --remote-name-all \
  --output-dir "$LOCAL_MODEL_DIR" "${model_urls[@]}"
while IFS=$'\t' read -r file expected_size; do
  actual_size=$(stat -c %s "$LOCAL_MODEL_DIR/$file")
  test "$actual_size" = "$expected_size" || { echo "size mismatch: $file" >&2; exit 1; }
done < "$SHARED/model_source_files.tsv"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$LOCAL_MODEL_RUN_DIR/hf-cache"
until curl -sf "http://${INFRA_IP}:8000/health" >/dev/null 2>&1; do sleep 2; done
args=(--model "$LOCAL_MODEL_DIR" --served-model-name "$MODEL" --tensor-parallel-size 1 --data-parallel-size "$DATA_PARALLEL_SIZE" --disaggregation-mode "$ROLE" --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" --enforce-eager --block-size 128 --enable-prefix-caching --no-enable-log-requests)
env DYN_LOG=info DYN_DISAGGREGATION_MODE="$ROLE" NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo UCX_NET_DEVICES=all UCX_IB_GPU_DIRECT_RDMA=no VLLM_SSM_CONV_STATE_LAYOUT=DS \
  python3 -m dynamo.vllm "${args[@]}" >"$LOGDIR/${ROLE}.log" 2>&1 &
worker_pid=$!
until worker_count=$(grep -c "VllmWorker for .* has been initialized" "$LOGDIR/${ROLE}.log" 2>/dev/null || true) && worker_count="${worker_count:-0}" && [ "$worker_count" -ge "$READY_WORKERS" ]; do
  kill -0 "$worker_pid" 2>/dev/null || { wait "$worker_pid"; exit 1; }
  sleep 3
done
touch "$SHARED/${ROLE}_workers_ready"
while [ ! -e "$SHARED/stop_workers" ]; do
  kill -0 "$worker_pid" 2>/dev/null || { wait "$worker_pid"; exit 1; }
  sleep 5
done
kill "$worker_pid" 2>/dev/null || true
wait "$worker_pid" 2>/dev/null || true
INNER
status=$?
set -e
touch "$SHARED/${ROLE}_workers_done"
exit "$status"
