#!/bin/bash
#SBATCH --job-name=dynamo-repro-infra-client
#SBATCH --partition=h100x4_aarch64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#
# Reproducer, job A of 2: etcd + NATS + 2 Dynamo frontend replicas + aiperf
# client, all on ONE node. Submitted as a separate job from
# dynamo_repro_workers.slurm (job B) because this cluster's Slurm topology
# config rejects any job spanning more than one node on this class of
# partition (isolated per-node switches, no shared parent in topology.conf)
# — so each role gets its own single-node job instead of one multi-node
# allocation.
#
# Storage lives under $HOME, not scratch: $HOME is NFS-mounted on every
# partition, while /home/scratch.sbak_coreai is only mounted on "ipp"-tagged
# nodes -- using $HOME means this job (and dynamo_repro_workers.slurm) can
# run on ANY partition, including GPU-heavy ones without scratch access.
# pyxis does NOT auto-forward $HOME into containers either, so it's still
# bind-mounted explicitly via --container-mounts below.
#
# $HOME has only a 5GB quota, so once a run's logs land here this script
# archives all but the newest run directory off to scratch
# (/home/scratch.sbak_coreai/dynamo_repro/archived-runs/) whenever $HOME
# crosses 80% usage -- see archive_old_runs_if_needed() below. Archived runs
# are only browsable from scratch-having nodes afterward, which is fine since
# they're just historical logs at that point.
#
# Hand-off to job B: this script resolves its own node's routable IP and
# writes it to $SHARED/infra_ip.txt; job B polls for that file to learn where
# to reach etcd/NATS/the frontends. Both jobs must be submitted with the same
# RUN_ID in their environment so they agree on $SHARED and $LOGDIR.
#
# Everything for one run lives under $HOME/dynamo_repro/runs/<RUN_ID>/
# {shared,logs}/ (RUN_ID defaults to a YYYYMMDD_HHMMSS timestamp if not set).
# The wheel to install comes from $HOME/dynamo_repro/wheelhouse-latest/,
# populated by dynamo_repro_build_wheel.slurm (which itself still builds
# using a persistent scratch-side build-cache -- see that file).
#
# Client uses a multi-turn conversation workload (not flat single-turn
# synthetic requests) so the KV-aware router actually has real cache overlap
# to differentiate workers on: --shared-system-prompt-length gives every
# session an identical prefix, --conversation-turn-mean gives each session
# several turns building on shared history, and
# --connection-reuse-strategy sticky-user-sessions pins a session's turns to
# one connection (tests sticky/session-aware routing specifically).
#
# Usage:
#   export RUN_ID=$(date +%Y%m%d_%H%M%S)   # or let this script pick one
#   sbatch dynamo_repro_infra_client.slurm
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

# Move all but the newest run directory under $HOME/dynamo_repro/runs/ to
# scratch once $HOME crosses 80% used -- keeps the 5GB $HOME quota from being
# exhausted by accumulated run logs (this is what previously happened when
# everything lived directly under $HOME -- see git history).
archive_old_runs_if_needed() {
  local used size pct
  read -r used size < <(df --output=used,size "$HOME" 2>/dev/null | tail -1)
  [ -z "$used" ] && return 0
  pct=$(( used * 100 / size ))
  if [ "$pct" -ge 80 ]; then
    echo "[repro-A] \$HOME at ${pct}% used, archiving old run dirs to ${SCRATCH_ARCHIVE}..."
    mkdir -p "$SCRATCH_ARCHIVE"
    find "${BASE_DIR}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r d; do
      mv "$d" "${SCRATCH_ARCHIVE}/" && echo "[repro-A] archived $(basename "$d")"
    done
  fi
}
archive_old_runs_if_needed

mkdir -p "$SHARED" "$LOGDIR"

# Exported explicitly (not relied on as the container's own $HOME) because
# the srun heredoc below is single-quoted -- $HOME inside it would otherwise
# resolve via the container's environment, which need not match the host
# path used for --container-mounts.
export HOST_HOME="$HOME"

# Matches dynamo_repro_build_wheel.slurm's WHEELHOUSE_NAME -- lets two
# builds (e.g. "baseline" vs "latest") coexist under $HOME for A/B
# comparisons; pick which one this run installs from.
export WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-latest}"

export INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
echo -n "$INFRA_IP" > "$SHARED/infra_ip.txt"
echo "[repro-A] node=${SLURMD_NODENAME} ip=${INFRA_IP} run_id=${RUN_ID}"
echo "[repro-A] wrote ${SHARED}/infra_ip.txt for job B to read"
echo "[repro-A] logs in ${LOGDIR}"

export ETCD_ENDPOINTS="${INFRA_IP}:2379"
export NATS_SERVER="nats://${INFRA_IP}:4222"

# Inner script piped via stdin (heredoc), not a `bash -c` string, so nothing
# needs escaping against the outer script — values cross over as exported
# env vars instead.
srun --container-image="$IMAGE" \
     --container-mounts="${HOST_HOME}:${HOST_HOME}" \
     --export=ALL,ETCD_ENDPOINTS,NATS_SERVER,MODEL,INFRA_IP,LOGDIR,HOST_HOME,WHEELHOUSE_NAME \
     bash -s <<'INNER'
set -euo pipefail

# Install the custom-built wheel (dynamo_repro_build_wheel.slurm output),
# if present, before starting anything -- overrides the NGC image's stock
# dynamo install with whatever DEP code changes were just built. `pip`/
# `python` here resolve via the image's baked-in PATH (same one
# `python -m dynamo.frontend` already relies on below), so no --system flag
# or explicit venv path needed.
WHEELHOUSE="${HOST_HOME}/dynamo_repro/wheelhouse-${WHEELHOUSE_NAME}"
if [ -d "$WHEELHOUSE" ]; then
  echo "[repro-A] installing custom dynamo wheel from ${WHEELHOUSE}"
  pip install --no-deps "${WHEELHOUSE}"/ai_dynamo_runtime*.whl "${WHEELHOUSE}"/ai_dynamo-*any.whl
  pip show ai-dynamo ai-dynamo-runtime 2>/dev/null | grep -E "^Name|^Version"
else
  echo "[repro-A] no wheelhouse found at ${WHEELHOUSE}, using stock image dynamo install"
fi

nohup etcd --data-dir "/tmp/etcd-${SLURM_JOB_ID}" \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls "http://${INFRA_IP}:2379" \
  > ${LOGDIR}/etcd.log 2>&1 &
ETCD_PID=$!

nohup nats-server -js -a 0.0.0.0 -p 4222 > ${LOGDIR}/nats.log 2>&1 &
NATS_PID=$!
sleep 5

# DYN_LOG defaults to info (routing-decision logging in
# lib/kv-router/src/scheduling/selector.rs runs at INFO level
# unconditionally: "Routing formula" once per selector at startup,
# "Routing decision" once per request). Override with
# `export DYN_LOG=warn` before submitting to measure routing-decision
# logging overhead -- see the CPU-time instrumentation below.
DYN_LOG="${DYN_LOG:-info}"
echo "[repro-A] DYN_LOG=${DYN_LOG}"

# DYN_REQUEST_TRACE=1 turns on Stack D's per-request JSONL trace -- separate
# feature from routing-decision logging, but it does its own per-request
# file I/O, which would confound a DYN_LOG=info-vs-warn CPU-time comparison.
# Off by default; only turn on when actually inspecting Stack D's trace.
DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE:-0}"
echo "[repro-A] DYN_REQUEST_TRACE=${DYN_REQUEST_TRACE}"

# Only replica 0 gets --router-reset-states (must run on exactly one replica).
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

echo "[repro-A] waiting for both frontends to report healthy..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done
until curl -sf http://localhost:8001/health >/dev/null 2>&1; do sleep 2; done

echo "[repro-A] frontends healthy, waiting for a worker to register model '${MODEL}'..."
until curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; do sleep 5; done
echo "[repro-A] model registered by a worker, sending multi-turn client load through BOTH frontends"

# CPU time (utime+stime, in clock ticks) consumed by each frontend process
# so far -- snapshotted right before the client load starts, compared
# against another snapshot right after it finishes. This isolates the
# frontend/router's own CPU cost from end-to-end request latency (which is
# dominated by GPU inference time and wouldn't show a routing-logging-sized
# effect either way).
read_cpu_ticks() {
  awk '{print $14+$15}' "/proc/$1/stat" 2>/dev/null || echo 0
}
CLK_TCK=$(getconf CLK_TCK)
CPU0_BEFORE=$(read_cpu_ticks "$FRONTEND0_PID")
CPU1_BEFORE=$(read_cpu_ticks "$FRONTEND1_PID")

# Synthetic multi-turn conversation workload: 10 sessions, ~5 turns each
# (mean, +-1 stddev), each session sharing a 200-token system prompt across
# all turns and all sessions. Switched back to this from --public-dataset
# sharegpt because that path only replayed 1 turn per sampled conversation
# (Request Count == --conversation-num exactly) -- no real multi-turn
# replay happened despite --connection-reuse-strategy sticky-user-sessions.
# The synthetic generator gives direct, verified control over turn count.
aiperf profile --model "$MODEL" --url http://localhost:8000 \
  --endpoint-type chat --streaming --concurrency 8 \
  --conversation-num 10 --conversation-turn-mean 5 --conversation-turn-stddev 1 \
  --shared-system-prompt-length 200 \
  --connection-reuse-strategy sticky-user-sessions \
  --synthetic-input-tokens-mean 1000 --output-tokens-mean 200 \
  > ${LOGDIR}/client-frontend0.log 2>&1 &
CLIENT0_PID=$!

aiperf profile --model "$MODEL" --url http://localhost:8001 \
  --endpoint-type chat --streaming --concurrency 8 \
  --conversation-num 10 --conversation-turn-mean 5 --conversation-turn-stddev 1 \
  --shared-system-prompt-length 200 \
  --connection-reuse-strategy sticky-user-sessions \
  --synthetic-input-tokens-mean 1000 --output-tokens-mean 200 \
  > ${LOGDIR}/client-frontend1.log 2>&1 &
CLIENT1_PID=$!

# Wait only for the two client processes, not the long-lived
# etcd/nats/frontend servers (those never exit on their own -- a plain
# `wait` here would just block until the job hits its walltime limit).
wait "$CLIENT0_PID" "$CLIENT1_PID"

CPU0_AFTER=$(read_cpu_ticks "$FRONTEND0_PID")
CPU1_AFTER=$(read_cpu_ticks "$FRONTEND1_PID")
CPU0_SECONDS=$(awk -v a="$CPU0_BEFORE" -v b="$CPU0_AFTER" -v hz="$CLK_TCK" 'BEGIN { printf "%.3f", (b - a) / hz }')
CPU1_SECONDS=$(awk -v a="$CPU1_BEFORE" -v b="$CPU1_AFTER" -v hz="$CLK_TCK" 'BEGIN { printf "%.3f", (b - a) / hz }')
echo "[repro-A] DYN_LOG=${DYN_LOG} frontend-0 CPU time during client run: ${CPU0_SECONDS}s"
echo "[repro-A] DYN_LOG=${DYN_LOG} frontend-1 CPU time during client run: ${CPU1_SECONDS}s"

echo "[repro-A] client finished, shutting down frontends/etcd/nats..."
kill "$FRONTEND0_PID" "$FRONTEND1_PID" "$ETCD_PID" "$NATS_PID" 2>/dev/null || true
wait 2>/dev/null || true
INNER

echo "[repro-A] client finished. Logs in ${LOGDIR}"
