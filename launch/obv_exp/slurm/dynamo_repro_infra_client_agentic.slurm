#!/bin/bash
#SBATCH --job-name=dynamo-repro-infra-client-agentic
#SBATCH --partition=h100x4_aarch64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#
# Reproducer, job A variant: etcd + NATS + 2 Dynamo frontend replicas, same
# as dynamo_repro_infra_client.slurm, but the client load is aiperf's
# Agentic Code Dataset Generator instead of the flat synthetic multi-turn
# config -- kept as a SEPARATE script (not a flag on the original) so the
# two client configs stay independently reproducible side by side.
#
# What the agentic generator models that the flat synthetic config doesn't:
#   - layered shared-prompt reuse: L1 (global tools/system prompt, shared
#     across ALL sessions), L1.5 (group-shared repo instructions), L2
#     (session-specific starting context), L3 (conversation history that
#     grows incrementally turn over turn)
#   - repository context sampled around a much larger mean (~15k tokens,
#     capped here via --max-isl to fit the model's context window)
#   - probabilistic session resets/retires/restart-continuations, which
#     split a session while preserving cache reuse across the split --
#     directly relevant to the KV-router load/overlap work this reproducer
#     exists for
#   - inter-turn delays sampled from an agentic/human delay mixture instead
#     of a flat Gaussian
#
# Two-step workflow (see aiperf docs: Agentic Code Dataset Generator):
#   1. `aiperf synthesize agentic-code` generates a Mooncake-trace JSONL
#      dataset offline (no server needed).
#   2. `aiperf profile --custom-dataset-type mooncake_trace --input-file`
#      replays it against a live endpoint.
# Both steps run inside the same container as the frontends, right before
# the replay step, so no separate job/dataset-staging step is needed.
#
# Everything else (storage layout, $HOME archival, wheel install, CPU-time
# instrumentation) is identical to dynamo_repro_infra_client.slurm -- see
# that file's header comments for why.
#
# Usage:
#   export RUN_ID=$(date +%Y%m%d_%H%M%S)   # or let this script pick one
#   sbatch dynamo_repro_infra_client_agentic.slurm
#   sbatch dynamo_repro_workers.slurm       # after this job is RUNNING, same RUN_ID

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

BASE_DIR="${HOME}/dynamo_repro"
SCRATCH_ARCHIVE="/home/scratch.sbak_coreai/dynamo_repro/archived-runs"
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"
SHARED="${SHARED:-${RUN_DIR}/shared}"
export LOGDIR="${LOGDIR:-${RUN_DIR}/logs}"
export DATASET_DIR="${RUN_DIR}/agentic-dataset"

# Move all but the newest run directory under $HOME/dynamo_repro/runs/ to
# scratch once $HOME crosses 80% used -- see dynamo_repro_infra_client.slurm.
archive_old_runs_if_needed() {
  local used size pct
  read -r used size < <(df --output=used,size "$HOME" 2>/dev/null | tail -1)
  [ -z "$used" ] && return 0
  pct=$(( used * 100 / size ))
  if [ "$pct" -ge 80 ]; then
    echo "[repro-A-agentic] \$HOME at ${pct}% used, archiving old run dirs to ${SCRATCH_ARCHIVE}..."
    mkdir -p "$SCRATCH_ARCHIVE"
    find "${BASE_DIR}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r d; do
      mv "$d" "${SCRATCH_ARCHIVE}/" && echo "[repro-A-agentic] archived $(basename "$d")"
    done
  fi
}
archive_old_runs_if_needed

mkdir -p "$SHARED" "$LOGDIR" "$DATASET_DIR"

export HOST_HOME="$HOME"
export WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-latest}"

export INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
echo -n "$INFRA_IP" > "$SHARED/infra_ip.txt"
echo "[repro-A-agentic] node=${SLURMD_NODENAME} ip=${INFRA_IP} run_id=${RUN_ID}"
echo "[repro-A-agentic] wrote ${SHARED}/infra_ip.txt for job B to read"
echo "[repro-A-agentic] logs in ${LOGDIR}, dataset in ${DATASET_DIR}"

export ETCD_ENDPOINTS="${INFRA_IP}:2379"
export NATS_SERVER="nats://${INFRA_IP}:4222"

# Session/size knobs, overridable from the submitting shell. Kept modest
# (10 sessions, same as the flat synthetic config).
#
# AGENTIC_CONFIG points at a CUSTOM generator config, not aiperf's bundled
# "default" one -- the bundled default is tuned for ~167k-token-context
# models (max_prompt_tokens=167000, cache.layer1_tokens=32000 alone). Any
# --max-isl override we tried (4000-16000) was still smaller than that fixed
# 32000-token layer1 pool, so every session was born already over budget and
# got force-retired on turn 1 (confirmed empirically: 100% forced-retire
# rate at every tested --max-isl). --max-isl only overrides max_prompt_tokens
# -- it does NOT shrink the cache layer sizes baked into the config, so it
# can't fix this alone.
#
# dynamo_repro_agentic_config.json (copied to $HOME, see below, so this
# script stays scratch-independent like the rest of this reproducer) shrinks
# layer1_tokens 32000->500 and layer2 mean 15000->3000, with
# max_prompt_tokens=12000 -- sized for this reproducer's 32768-context model
# with real headroom for turn growth. Verified empirically: 10 sessions / 62
# total turns (avg 6.2 turns/session, range 2-9) instead of 10 turns total
# (1 per session, 100% immediate forced-retire) with the bundled default.
export NUM_SESSIONS="${NUM_SESSIONS:-10}"
export AGENTIC_CONFIG="${AGENTIC_CONFIG:-${HOST_HOME}/dynamo_repro/agentic-config.json}"

srun --container-image="$IMAGE" \
     --container-mounts="${HOST_HOME}:${HOST_HOME}" \
     --export=ALL,ETCD_ENDPOINTS,NATS_SERVER,MODEL,INFRA_IP,LOGDIR,DATASET_DIR,HOST_HOME,WHEELHOUSE_NAME,NUM_SESSIONS,AGENTIC_CONFIG \
     bash -s <<'INNER'
set -euo pipefail

WHEELHOUSE="${HOST_HOME}/dynamo_repro/wheelhouse-${WHEELHOUSE_NAME}"
if [ -d "$WHEELHOUSE" ]; then
  echo "[repro-A-agentic] installing custom dynamo wheel from ${WHEELHOUSE}"
  pip install --no-deps "${WHEELHOUSE}"/ai_dynamo_runtime*.whl "${WHEELHOUSE}"/ai_dynamo-*any.whl
  pip show ai-dynamo ai-dynamo-runtime 2>/dev/null | grep -E "^Name|^Version"
else
  echo "[repro-A-agentic] no wheelhouse found at ${WHEELHOUSE}, using stock image dynamo install"
fi

nohup etcd --data-dir "/tmp/etcd-${SLURM_JOB_ID}" \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls "http://${INFRA_IP}:2379" \
  > ${LOGDIR}/etcd.log 2>&1 &
ETCD_PID=$!

nohup nats-server -js -a 0.0.0.0 -p 4222 > ${LOGDIR}/nats.log 2>&1 &
NATS_PID=$!
sleep 5

DYN_LOG="${DYN_LOG:-info}"
echo "[repro-A-agentic] DYN_LOG=${DYN_LOG}"

DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE:-0}"
echo "[repro-A-agentic] DYN_REQUEST_TRACE=${DYN_REQUEST_TRACE}"

DYN_SYSTEM_PORT=8091 DYN_LOG="${DYN_LOG}" DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE}" \
DYN_REQUEST_TRACE_OUTPUT_PATH=${LOGDIR}/request-trace-0.jsonl \
  python -m dynamo.frontend --http-port 8000 \
    --router-mode kv --router-reset-states --enforce-disagg \
    > ${LOGDIR}/frontend-0.log 2>&1 &
FRONTEND0_PID=$!

DYN_SYSTEM_PORT=8092 DYN_LOG="${DYN_LOG}" DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE}" \
DYN_REQUEST_TRACE_OUTPUT_PATH=${LOGDIR}/request-trace-1.jsonl \
  python -m dynamo.frontend --http-port 8001 \
    --router-mode kv --enforce-disagg \
    > ${LOGDIR}/frontend-1.log 2>&1 &
FRONTEND1_PID=$!

echo "[repro-A-agentic] waiting for both frontends to report healthy..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done
until curl -sf http://localhost:8001/health >/dev/null 2>&1; do sleep 2; done

echo "[repro-A-agentic] frontends healthy, waiting for a worker to register model '${MODEL}'..."
until curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; do sleep 5; done
echo "[repro-A-agentic] model registered by a worker"

# Generate two independent agentic-code datasets (different seeds), one per
# frontend -- mirrors the flat synthetic config sending independently
# randomized sessions to each frontend rather than replaying the identical
# trace against both.
echo "[repro-A-agentic] generating agentic-code dataset for frontend-0 (seed=1)..."
aiperf synthesize agentic-code --num-sessions "$NUM_SESSIONS" --config "$AGENTIC_CONFIG" \
  --seed 1 --output "${DATASET_DIR}/frontend-0" > ${LOGDIR}/synthesize-frontend0.log 2>&1
DATASET0=$(find "${DATASET_DIR}/frontend-0" -name "dataset.jsonl" | head -1)

echo "[repro-A-agentic] generating agentic-code dataset for frontend-1 (seed=2)..."
aiperf synthesize agentic-code --num-sessions "$NUM_SESSIONS" --config "$AGENTIC_CONFIG" \
  --seed 2 --output "${DATASET_DIR}/frontend-1" > ${LOGDIR}/synthesize-frontend1.log 2>&1
DATASET1=$(find "${DATASET_DIR}/frontend-1" -name "dataset.jsonl" | head -1)

echo "[repro-A-agentic] dataset0=${DATASET0}"
echo "[repro-A-agentic] dataset1=${DATASET1}"
echo "[repro-A-agentic] replaying agentic-code traces through BOTH frontends"

read_cpu_ticks() {
  awk '{print $14+$15}' "/proc/$1/stat" 2>/dev/null || echo 0
}
CLK_TCK=$(getconf CLK_TCK)
CPU0_BEFORE=$(read_cpu_ticks "$FRONTEND0_PID")
CPU1_BEFORE=$(read_cpu_ticks "$FRONTEND1_PID")

aiperf profile --model "$MODEL" --url http://localhost:8000 \
  --endpoint-type chat --streaming --concurrency 8 \
  --input-file "$DATASET0" --custom-dataset-type mooncake_trace \
  > ${LOGDIR}/client-frontend0.log 2>&1 &
CLIENT0_PID=$!

aiperf profile --model "$MODEL" --url http://localhost:8001 \
  --endpoint-type chat --streaming --concurrency 8 \
  --input-file "$DATASET1" --custom-dataset-type mooncake_trace \
  > ${LOGDIR}/client-frontend1.log 2>&1 &
CLIENT1_PID=$!

wait "$CLIENT0_PID" "$CLIENT1_PID"

CPU0_AFTER=$(read_cpu_ticks "$FRONTEND0_PID")
CPU1_AFTER=$(read_cpu_ticks "$FRONTEND1_PID")
CPU0_SECONDS=$(awk -v a="$CPU0_BEFORE" -v b="$CPU0_AFTER" -v hz="$CLK_TCK" 'BEGIN { printf "%.3f", (b - a) / hz }')
CPU1_SECONDS=$(awk -v a="$CPU1_BEFORE" -v b="$CPU1_AFTER" -v hz="$CLK_TCK" 'BEGIN { printf "%.3f", (b - a) / hz }')
echo "[repro-A-agentic] DYN_LOG=${DYN_LOG} frontend-0 CPU time during client run: ${CPU0_SECONDS}s"
echo "[repro-A-agentic] DYN_LOG=${DYN_LOG} frontend-1 CPU time during client run: ${CPU1_SECONDS}s"

echo "[repro-A-agentic] client finished, shutting down frontends/etcd/nats..."
kill "$FRONTEND0_PID" "$FRONTEND1_PID" "$ETCD_PID" "$NATS_PID" 2>/dev/null || true
wait 2>/dev/null || true
INNER

echo "[repro-A-agentic] client finished. Logs in ${LOGDIR}"
