#!/bin/bash
#SBATCH --job-name=qwen36-vllm-frontend-client
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# Job 1 of 3: frontend and AIPerf share an ARM+IPP node. Submit this first.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
RUN_ID="${RUN_ID:-qwen36_vllm_$(date +%Y%m%d_%H%M%S)}"
HOST_HOME="$HOME"
RUN_DIR="$HOST_HOME/dynamo_repro/runs/$RUN_ID"
SHARED="$RUN_DIR/shared"
LOGDIR="$RUN_DIR/logs"
WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-vllm-1.4.1-aarch64}"
STAGE_ROOT="${STAGE_ROOT:-$HOME/scratch/dynamo_repro/completed_runs}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-1}"
CLIENT_REQUESTS="${CLIENT_REQUESTS:-4}"
mkdir -p "$SHARED" "$LOGDIR"
rm -f "$SHARED"/{infra_ip.txt,frontend_ready,prefill_workers_ready,decode_workers_ready,prefill_workers_done,decode_workers_done,stop_workers}
export IMAGE MODEL RUN_ID HOST_HOME RUN_DIR SHARED LOGDIR WHEELHOUSE_NAME STAGE_ROOT CLIENT_CONCURRENCY CLIENT_REQUESTS

stage_logs() {
  local dest="$STAGE_ROOT/$RUN_ID"
  mkdir -p "$dest"
  rsync -a "$RUN_DIR/" "$dest/" || true
  printf 'run_id=%s\nstaged_at_utc=%s\nsource=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" "$RUN_DIR" > "$dest/STAGED.txt"
}
trap stage_logs EXIT

INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
printf '%s\n' "$INFRA_IP" > "$SHARED/infra_ip.txt"
export INFRA_IP
echo "[vllm-frontend-client] run=$RUN_ID node=$SLURMD_NODENAME infra_ip=$INFRA_IP"

srun --overlap --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="$HOST_HOME:$HOST_HOME" --container-remap-root --export=ALL \
  bash -s <<'FRONTEND' &
set -euo pipefail
wh="$HOST_HOME/dynamo_repro/wheelhouse-$WHEELHOUSE_NAME"
pip install --break-system-packages --force-reinstall --no-deps "$wh"/ai_dynamo_runtime*.whl "$wh"/ai_dynamo-*any.whl
export ETCD_ENDPOINTS="$INFRA_IP:2379" NATS_SERVER="nats://$INFRA_IP:4222"
etcd --data-dir "/tmp/etcd-$SLURM_JOB_ID" --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls "http://$INFRA_IP:2379" >"$LOGDIR/etcd.log" 2>&1 & ETCD_PID=$!
nats-server -js -a 0.0.0.0 -p 4222 >"$LOGDIR/nats.log" 2>&1 & NATS_PID=$!
python -m dynamo.frontend --http-port 8000 --router-mode "${ROUTER_MODE:-kv}" --enforce-disagg >"$LOGDIR/frontend.log" 2>&1 & FRONTEND_PID=$!
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do kill -0 "$FRONTEND_PID" 2>/dev/null || exit 1; sleep 2; done
touch "$SHARED/frontend_ready"
while [ ! -e "$SHARED/stop_workers" ]; do kill -0 "$FRONTEND_PID" 2>/dev/null || exit 1; sleep 3; done
kill "$FRONTEND_PID" "$ETCD_PID" "$NATS_PID" 2>/dev/null || true
wait 2>/dev/null || true
FRONTEND
frontend_pid=$!

until [ -e "$SHARED/frontend_ready" ]; do kill -0 "$frontend_pid" 2>/dev/null || { wait "$frontend_pid"; exit 1; }; sleep 2; done

set +e
srun --overlap --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="$HOST_HOME:$HOST_HOME" --container-remap-root --export=ALL \
  bash -s <<'CLIENT'
set -euo pipefail
python -m pip install --break-system-packages --quiet --target /tmp/aiperf-site "aiperf==${AIPERF_VERSION:-0.11.0}"
aiperf() { PYTHONPATH=/tmp/aiperf-site python -m aiperf "$@"; }
until [ -e "$SHARED/prefill_workers_ready" ] && [ -e "$SHARED/decode_workers_ready" ]; do sleep 2; done
until curl -sf "http://$INFRA_IP:8000/v1/models" | grep -q "$MODEL"; do sleep 3; done
mkdir -p "$LOGDIR/aiperf"
aiperf profile --model "$MODEL" --url "http://$INFRA_IP:8000" --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" --request-count "$CLIENT_REQUESTS" --synthetic-input-tokens-mean 64 --output-tokens-mean 16 --artifact-dir "$LOGDIR/aiperf" >"$LOGDIR/client.log" 2>&1
CLIENT
client_status=$?
set -e
touch "$SHARED/stop_workers"
for marker in "$SHARED/prefill_workers_done" "$SHARED/decode_workers_done"; do until [ -e "$marker" ]; do sleep 2; done; done
wait "$frontend_pid"; frontend_status=$?
echo "[vllm-frontend-client] logs staged at $STAGE_ROOT/$RUN_ID"
[ "$client_status" -eq 0 ] && [ "$frontend_status" -eq 0 ]
