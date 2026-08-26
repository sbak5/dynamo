#!/bin/bash
#SBATCH --job-name=qwen36-vllm-role
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00

# One vLLM P/D role node. DATA_PARALLEL_SIZE local workers share its GPUs.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
RUN_ID="${RUN_ID:?RUN_ID must match frontend job}"
ROLE="${ROLE:?ROLE must be prefill or decode}"
case "$ROLE" in prefill|decode) ;; *) echo "invalid ROLE=$ROLE" >&2; exit 2;; esac
HOST_HOME="$HOME"
RUN_DIR="$HOST_HOME/dynamo_repro/runs/$RUN_ID"
SHARED="$RUN_DIR/shared"
LOGDIR="$RUN_DIR/logs"
WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-vllm-1.4.1-aarch64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
case "$DATA_PARALLEL_SIZE" in *[!0-9]*|'') echo "invalid DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE" >&2; exit 2;; esac
# One Dynamo worker can own several vLLM internal data-parallel replicas.
READY_WORKERS="${READY_WORKERS:-1}"
case "$READY_WORKERS" in *[!0-9]*|'') echo "invalid READY_WORKERS=$READY_WORKERS" >&2; exit 2;; esac
export IMAGE MODEL RUN_ID ROLE HOST_HOME RUN_DIR SHARED LOGDIR WHEELHOUSE_NAME GPU_MEMORY_UTILIZATION MAX_MODEL_LEN DATA_PARALLEL_SIZE READY_WORKERS

until [ -s "$SHARED/infra_ip.txt" ]; do sleep 2; done
INFRA_IP=$(cat "$SHARED/infra_ip.txt")
export INFRA_IP ETCD_ENDPOINTS="$INFRA_IP:2379" NATS_SERVER="nats://$INFRA_IP:4222"

set +e
srun --container-image="$IMAGE" --container-mounts="$HOST_HOME:$HOST_HOME" \
  --container-remap-root --export=ALL bash -s <<'INNER'
set -euo pipefail
wh="$HOST_HOME/dynamo_repro/wheelhouse-$WHEELHOUSE_NAME"
pip install --break-system-packages --no-deps "$wh"/ai_dynamo_runtime*.whl "$wh"/ai_dynamo-*any.whl
export HF_HOME="/tmp/dynamo-hf-cache-${USER}"
mkdir -p "$HF_HOME"
until curl -sf "http://${INFRA_IP}:8000/health" >/dev/null 2>&1; do sleep 2; done
args=(--model "$MODEL" --served-model-name "$MODEL" --tensor-parallel-size 1 --data-parallel-size "$DATA_PARALLEL_SIZE" --disaggregation-mode "$ROLE" --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" --enforce-eager --block-size 128 --enable-prefix-caching --no-enable-log-requests)
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
