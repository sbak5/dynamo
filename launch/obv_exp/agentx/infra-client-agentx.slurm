#!/bin/bash
#SBATCH --job-name=dynamo-repro-infra-client-fpm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=0
#SBATCH --nodelist=ipp2-0037
#SBATCH --time=02:00:00
#
# Reproducer, job A of 2: etcd + NATS + Tempo + 2 Dynamo frontend replicas +
# AIPerf.  This tracing variant is intentionally self-contained: it preserves
# the raw AIPerf JSONL, FPM, routing logs, and Tempo trace cache needed by the
# dashboard in the sibling repository.
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
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"
SHARED="${SHARED:-${RUN_DIR}/shared}"
export LOGDIR="${LOGDIR:-${RUN_DIR}/logs}"
export SHARED

# Do not access scratch here: frontend/client jobs may run on any node. Runs
# remain under $HOME until explicitly staged from an ipp node after completion.

mkdir -p "$SHARED" "$LOGDIR"
rm -f "$SHARED/stop_workers" "$SHARED/prefill_workers_done" "$SHARED/decode_workers_done" \
  "$SHARED/prefill_workers_ready" "$SHARED/decode_workers_ready" \
  "$SHARED/prefill_page_size" "$SHARED/decode_page_size"

# Exported explicitly (not relied on as the container's own $HOME) because
# the srun heredoc below is single-quoted -- $HOME inside it would otherwise
# resolve via the container's environment, which need not match the host
# path used for --container-mounts.
export HOST_HOME="$HOME"
# AIPerf's per-request profile JSONL can be large for the production workload.
# Frontend/client jobs run on ipp nodes, so spool these artifacts directly to
# scratch during the run and merge them into the completed run's logs/ at
# staging time.  Keeping them under $RUN_DIR would exhaust the 5-GiB home
# quota before the client finishes.
export AIPERF_ARTIFACT_ROOT="${AIPERF_ARTIFACT_ROOT:-/home/scratch.sbak_coreai/dynamo_repro/inflight_runs/${RUN_ID}/logs}"
mkdir -p "$AIPERF_ARTIFACT_ROOT"
# Pyxis mounts $HOME explicitly but does not automatically expose scratch.
# AIPerf writes its raw JSONL under AIPERF_ARTIFACT_ROOT, so bind the ipp-only
# scratch path at the identical container path; otherwise records disappear
# when the frontend container exits.
export IPP_SCRATCH="/home/scratch.sbak_coreai"
# Latest Dynamo resolves model-card metadata in the frontend as well as in
# workers. Keep the lightweight HF cache on the NFS-mounted host path so the
# frontend can use config/tokenizer metadata without a fresh Hub API call.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOST_HOME}/dynamo_repro/hf-hub}"

# Matches dynamo_repro_build_wheel.slurm's WHEELHOUSE_NAME -- lets two
# builds (e.g. "baseline" vs "latest") coexist under $HOME for A/B
# comparisons; pick which one this run installs from.
# This wheel is built from the current selector/candidate logging commit.
# It is x86_64 because this job runs on an ipp node.
export WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-selector-log-fix-x86}"

# The default preserves the KV-router baseline.  For the requested
# conversation-aware comparison submit with:
#   ROUTER_MODE=random ROUTER_SESSION_AFFINITY_TTL_SECS=900
# This makes AIPerf emit a stable, distinct x-dynamo-session-id for every
# multi-turn conversation (rather than merely reusing its HTTP connection).
export ROUTER_MODE="${ROUTER_MODE:-kv}"
export ROUTER_SESSION_AFFINITY_TTL_SECS="${ROUTER_SESSION_AFFINITY_TTL_SECS:-}"
# This is a vLLM launcher.  The SGLang-only page-size marker is not produced.
export WAIT_FOR_PAGE_SIZE="${WAIT_FOR_PAGE_SIZE:-0}"
export CLIENT_SESSION_HEADER="${CLIENT_SESSION_HEADER:-x-dynamo-session-id}"
# JSONL is explicitly enabled below; 0.11 remains the current client version.
export AIPERF_VERSION="${AIPERF_VERSION:-0.11.0}"
# Optional AgentX client environment prepared under scratch.  When unset, the
# existing pinned PyPI client is used for the synthetic workload.
export AIPERF_CLIENT_SITE="${AIPERF_CLIENT_SITE:-}"
export CLIENT_WORKLOAD="${CLIENT_WORKLOAD:-synthetic}"
export AGENTX_DATASET="${AGENTX_DATASET:-semianalysis_cc_traces_weka_with_subagents_060826_256k}"
export AGENTX_MAX_CONTEXT_LENGTH="${AGENTX_MAX_CONTEXT_LENGTH:-131072}"
export AGENTX_DURATION="${AGENTX_DURATION:-900}"

# Per frontend. Defaults create 256 concurrent sessions and 1,000
# conversations across the two frontends. This sustains enough arrival
# pressure to observe engine queueing under the previous multi-turn shape.
export CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-128}"
export CLIENT_CONVERSATIONS="${CLIENT_CONVERSATIONS:-500}"
export CLIENT_CONVERSATION_TURN_MEAN="${CLIENT_CONVERSATION_TURN_MEAN:-5}"
export CLIENT_CONVERSATION_TURN_STDDEV="${CLIENT_CONVERSATION_TURN_STDDEV:-1}"
export CLIENT_SHARED_SYSTEM_PROMPT_LENGTH="${CLIENT_SHARED_SYSTEM_PROMPT_LENGTH:-200}"
export CLIENT_SYNTHETIC_INPUT_TOKENS_MEAN="${CLIENT_SYNTHETIC_INPUT_TOKENS_MEAN:-1000}"
export CLIENT_OUTPUT_TOKENS_MEAN="${CLIENT_OUTPUT_TOKENS_MEAN:-200}"

export INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
echo -n "$INFRA_IP" > "$SHARED/infra_ip.txt"
echo "[repro-A] node=${SLURMD_NODENAME} ip=${INFRA_IP} run_id=${RUN_ID}"
echo "[repro-A] wrote ${SHARED}/infra_ip.txt for job B to read"
echo "[repro-A] logs in ${LOGDIR}"

export ETCD_ENDPOINTS="${INFRA_IP}:2379"
export NATS_SERVER="nats://${INFRA_IP}:4222"

# Direct Tempo ingestion avoids a separate Collector/Loki deployment.  Tempo
# runs in this allocation and is queried before it is stopped, so the staged
# run contains durable per-request trace JSON rather than a transient URL.
export OTEL_EXPORT_ENABLED=true
export DYN_LOGGING_JSONL=true
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://${INFRA_IP}:4317"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
export OTEL_TRACES_SAMPLE_RATIO=1.0
export TEMPO_CONFIG="${HOST_HOME}/dynamo_repro/runs/${RUN_ID}/shared/tempo.yaml"
cp "/home/scratch.sbak_coreai/dynamo-benchmark-perf-dashboard/launch/tempo.yaml" "$TEMPO_CONFIG"
sed -i "s|__TEMPO_DATA_DIR__|${RUN_DIR}/tempo|g" "$TEMPO_CONFIG"
mkdir -p "${RUN_DIR}/tempo"
TEMPO_BIN="/home/scratch.sbak_coreai/dynamo_repro/tools/tempo-2.8.2/tempo"
if [ ! -x "$TEMPO_BIN" ]; then
  echo "[repro-A] missing $TEMPO_BIN" >&2
  exit 1
fi
"$TEMPO_BIN" -config.file="$TEMPO_CONFIG" > "${LOGDIR}/tempo.log" 2>&1 &
TEMPO_PID=$!
echo "[repro-A] Tempo started pid=${TEMPO_PID}, OTLP=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"
until curl -sf "http://localhost:3200/ready" >/dev/null 2>&1; do sleep 1; done

# Inner script piped via stdin (heredoc), not a `bash -c` string, so nothing
# needs escaping against the outer script — values cross over as exported
# env vars instead.
srun --container-image="$IMAGE" \
     --container-mounts="${HOST_HOME}:${HOST_HOME},${IPP_SCRATCH}:${IPP_SCRATCH}" \
     --export=ALL,ETCD_ENDPOINTS,NATS_SERVER,MODEL,INFRA_IP,LOGDIR,HOST_HOME,HF_HUB_CACHE,WHEELHOUSE_NAME,ROUTER_MODE,ROUTER_SESSION_AFFINITY_TTL_SECS,CLIENT_SESSION_HEADER,AIPERF_VERSION,AIPERF_CLIENT_SITE,CLIENT_WORKLOAD,AGENTX_DATASET,AGENTX_MAX_CONTEXT_LENGTH,AGENTX_DURATION,CLIENT_CONCURRENCY,SHARED,WAIT_FOR_PAGE_SIZE,AIPERF_ARTIFACT_ROOT,OTEL_EXPORT_ENABLED,DYN_LOGGING_JSONL,OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,OTEL_EXPORTER_OTLP_TRACES_INSECURE,OTEL_TRACES_SAMPLE_RATIO \
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

# Capture every ForwardPassMetrics event from both worker groups. The
# collector subscribes before workers register; discovery adds publishers as
# they appear. It adds an absolute receive timestamp and writes one JSON
# object per event, preserving the per-worker counter_id for gap detection.
FPM_COLLECTOR="${HOST_HOME}/dynamo_repro/tools/collect_fpm.py"
if [ ! -f "$FPM_COLLECTOR" ]; then
  echo "[repro-A] missing FPM collector: $FPM_COLLECTOR" >&2
  exit 1
fi
python "$FPM_COLLECTOR" --output-dir "$LOGDIR" \
  > "${LOGDIR}/fpm-collector.log" 2>&1 &
FPM_COLLECTOR_PID=$!
echo "[repro-A] FPM collector started pid=${FPM_COLLECTOR_PID}"

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

# The runtime image carries AIPerf 0.8.0, which has no --session-header and
# therefore cannot send Dynamo's per-conversation affinity header. Install a
# pinned current client in a separate venv so this does not mutate the
# frontend's Python environment or its rebuilt Dynamo wheel.
if [ -n "$AIPERF_CLIENT_SITE" ]; then
  aiperf_client() {
    PYTHONPATH="${AIPERF_CLIENT_SITE}${PYTHONPATH:+:${PYTHONPATH}}" python -m aiperf "$@"
  }
else
  python -m pip install --quiet --target /tmp/aiperf-site "aiperf==${AIPERF_VERSION}"
  aiperf_client() {
    PYTHONPATH=/tmp/aiperf-site python -m aiperf "$@"
  }
fi
echo "[repro-A] client=$(aiperf_client --version)"

# Only replica 0 gets --router-reset-states (must run on exactly one replica).
ROUTER_ARGS=(--router-mode "$ROUTER_MODE" --enforce-disagg)
if [ -n "$ROUTER_SESSION_AFFINITY_TTL_SECS" ]; then
  ROUTER_ARGS+=(--router-session-affinity-ttl-secs "$ROUTER_SESSION_AFFINITY_TTL_SECS")
fi
ROUTER_REPLICA_SYNC="${ROUTER_REPLICA_SYNC:-1}"
if [ "$ROUTER_REPLICA_SYNC" = "1" ]; then
  ROUTER_ARGS+=(--router-replica-sync)
fi
echo "[repro-A] router_mode=${ROUTER_MODE} session_affinity_ttl_secs=${ROUTER_SESSION_AFFINITY_TTL_SECS:-disabled} router_replica_sync=${ROUTER_REPLICA_SYNC}"

DYN_SYSTEM_PORT=8091 OTEL_SERVICE_NAME=dynamo-frontend-0 DYN_LOG="${DYN_LOG}" DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE}" \
DYN_REQUEST_TRACE_OUTPUT_PATH=${LOGDIR}/request-trace-0.jsonl \
  python -m dynamo.frontend --http-port 8000 \
    "${ROUTER_ARGS[@]}" --router-reset-states \
    > ${LOGDIR}/frontend-0.log 2>&1 &
FRONTEND0_PID=$!

DYN_SYSTEM_PORT=8092 OTEL_SERVICE_NAME=dynamo-frontend-1 DYN_LOG="${DYN_LOG}" DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE}" \
DYN_REQUEST_TRACE_OUTPUT_PATH=${LOGDIR}/request-trace-1.jsonl \
  python -m dynamo.frontend --http-port 8001 \
    "${ROUTER_ARGS[@]}" \
    > ${LOGDIR}/frontend-1.log 2>&1 &
FRONTEND1_PID=$!

echo "[repro-A] waiting for both frontends to report healthy..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 2; done
until curl -sf http://localhost:8001/health >/dev/null 2>&1; do sleep 2; done

if [ "$WAIT_FOR_PAGE_SIZE" = "1" ]; then
  echo "[repro-A] frontends healthy, waiting for both P/D worker roles to validate page-size agreement..."
  until [ -e "$SHARED/prefill_workers_ready" ] && [ -e "$SHARED/decode_workers_ready" ]; do sleep 2; done
  echo "[repro-A] P/D page-size agreement: prefill=$(cat "$SHARED/prefill_page_size"), decode=$(cat "$SHARED/decode_page_size")"
else
  echo "[repro-A] frontends healthy; skipping SGLang-only page-size readiness gate"
fi
echo "[repro-A] waiting for a worker to register model '${MODEL}'..."
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

# Synthetic multi-turn conversation workload: 500 sessions per frontend, ~5 turns each
# (mean, +-1 stddev), each session sharing a 200-token system prompt across
# all turns and all sessions. Switched back to this from --public-dataset
# sharegpt because that path only replayed 1 turn per sampled conversation
# (Request Count == --conversation-num exactly) -- no real multi-turn
# replay happened despite --connection-reuse-strategy sticky-user-sessions.
# The synthetic generator gives direct, verified control over turn count.
# When affinity is enabled, map its per-conversation correlation ID to
# Dynamo's session-affinity header; it stays identical for every turn of one
# conversation and differs across conversations.
SESSION_AIPERF_ARGS=()
if [ -n "$ROUTER_SESSION_AFFINITY_TTL_SECS" ]; then
  SESSION_AIPERF_ARGS=(--session-header "$CLIENT_SESSION_HEADER")
fi
mkdir -p "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend0" "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend1"
if [ "$CLIENT_WORKLOAD" = agentx ]; then
  AGENTX_ARGS=(
    --scenario inferencex-agentx-mvp
    --max-context-length "$AGENTX_MAX_CONTEXT_LENGTH"
    --public-dataset "$AGENTX_DATASET"
    --benchmark-duration "$AGENTX_DURATION"
    --use-server-token-count
  )
  aiperf_client profile --model "$MODEL" --url http://localhost:8000 \
    --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" \
    "${SESSION_AIPERF_ARGS[@]}" "${AGENTX_ARGS[@]}" \
    --server-metrics-formats jsonl json csv \
    --artifact-dir "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend0" \
    > ${LOGDIR}/client-frontend0.log 2>&1 &
  CLIENT0_PID=$!

  aiperf_client profile --model "$MODEL" --url http://localhost:8001 \
    --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" \
    "${SESSION_AIPERF_ARGS[@]}" "${AGENTX_ARGS[@]}" \
    --server-metrics-formats jsonl json csv \
    --artifact-dir "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend1" \
    > ${LOGDIR}/client-frontend1.log 2>&1 &
  CLIENT1_PID=$!
else
aiperf_client profile --model "$MODEL" --url http://localhost:8000 \
  --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" \
  --conversation-num "$CLIENT_CONVERSATIONS" \
  --conversation-turn-mean "$CLIENT_CONVERSATION_TURN_MEAN" \
  --conversation-turn-stddev "$CLIENT_CONVERSATION_TURN_STDDEV" \
  --shared-system-prompt-length "$CLIENT_SHARED_SYSTEM_PROMPT_LENGTH" \
  --connection-reuse-strategy sticky-user-sessions \
  "${SESSION_AIPERF_ARGS[@]}" \
  --synthetic-input-tokens-mean "$CLIENT_SYNTHETIC_INPUT_TOKENS_MEAN" \
  --output-tokens-mean "$CLIENT_OUTPUT_TOKENS_MEAN" \
  --server-metrics-formats jsonl json csv \
  --artifact-dir "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend0" \
  > ${LOGDIR}/client-frontend0.log 2>&1 &
CLIENT0_PID=$!

aiperf_client profile --model "$MODEL" --url http://localhost:8001 \
  --endpoint-type chat --streaming --concurrency "$CLIENT_CONCURRENCY" \
  --conversation-num "$CLIENT_CONVERSATIONS" \
  --conversation-turn-mean "$CLIENT_CONVERSATION_TURN_MEAN" \
  --conversation-turn-stddev "$CLIENT_CONVERSATION_TURN_STDDEV" \
  --shared-system-prompt-length "$CLIENT_SHARED_SYSTEM_PROMPT_LENGTH" \
  --connection-reuse-strategy sticky-user-sessions \
  "${SESSION_AIPERF_ARGS[@]}" \
  --synthetic-input-tokens-mean "$CLIENT_SYNTHETIC_INPUT_TOKENS_MEAN" \
  --output-tokens-mean "$CLIENT_OUTPUT_TOKENS_MEAN" \
  --server-metrics-formats jsonl json csv \
  --artifact-dir "${AIPERF_ARTIFACT_ROOT}/aiperf-frontend1" \
  > ${LOGDIR}/client-frontend1.log 2>&1 &
CLIENT1_PID=$!
fi

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
sleep 2
kill "$FPM_COLLECTOR_PID" 2>/dev/null || true
wait "$FPM_COLLECTOR_PID" 2>/dev/null || true
kill "$FRONTEND0_PID" "$FRONTEND1_PID" "$ETCD_PID" "$NATS_PID" 2>/dev/null || true
wait 2>/dev/null || true
INNER

touch "$SHARED/stop_workers"
echo "[repro-A] wrote ${SHARED}/stop_workers for worker jobs"
echo "[repro-A] waiting for prefill/decode jobs to finish writing logs..."
for marker in "$SHARED/prefill_workers_done" "$SHARED/decode_workers_done"; do
  until [ -e "$marker" ]; do sleep 2; done
done

# The frontend/client job runs on an ipp node, so stage the finished run here
# without a separate Slurm staging job.
DEST="/home/scratch.sbak_coreai/dynamo_repro/completed_runs/${RUN_ID}"
mkdir -p "$DEST"
rsync -a --delete "$RUN_DIR/" "$DEST/"
# Keep AIPerf's raw records.  The dashboard consumes JSONL; its JSON/CSV
# summaries cannot provide the x_request_id/time-series joins.
rsync -a "$AIPERF_ARTIFACT_ROOT/" "$DEST/artifacts/"
cat "$AIPERF_ARTIFACT_ROOT"/aiperf-frontend*/profile_export.jsonl > "$DEST/profile_export.jsonl"
# AIPerf <=0.8 exported raw JSONL metric samples.  Newer AIPerf releases
# export only JSON/CSV summaries, which cannot drive the occupancy time series.
# Preserve those summaries under artifacts but do not fail the serving run.
SERVER_METRICS_LINE=""
if [ -f "$AIPERF_ARTIFACT_ROOT/aiperf-frontend0/server_metrics_export.jsonl" ]; then
  cp "$AIPERF_ARTIFACT_ROOT/aiperf-frontend0/server_metrics_export.jsonl" "$DEST/server_metrics_export.jsonl"
  # Worker-side streams carry vllm:kv_cache_usage_perc.  They are written on
  # the GB200 nodes under $HOME and included in RUN_DIR before this stage.
  if compgen -G "$DEST/logs/worker-metrics-*.jsonl" > /dev/null; then
    cat "$DEST"/logs/worker-metrics-*.jsonl >> "$DEST/server_metrics_export.jsonl"
  else
    echo "[repro-A] worker metrics streams absent; KV-util/free-block panels will be omitted" >&2
  fi
  SERVER_METRICS_LINE="  server_metrics: ${DEST}/server_metrics_export.jsonl"
else
  echo "[repro-A] AIPerf did not export raw server_metrics JSONL; occupancy panel will be omitted" >&2
fi
cat > "$DEST/dashboard.yaml" <<EOF
name: ${RUN_ID}
description: "Qwen3-32B-FP8, vLLM P/D 4x4 GB200, KV router, replica sync, no session affinity"
mode: disagg
framework: vllm
topology:
  block_size: 16
  workers:
    prefill: {parallelism: dep, rank: 1, worker_count: 4}
    decode: {parallelism: dep, rank: 1, worker_count: 4}
sources:
  aiperf_profile: ${DEST}/profile_export.jsonl
${SERVER_METRICS_LINE}
  trace_query_url: http://localhost:3200
EOF
# Tempo remains live while the dashboard performs its one-time exact request-ID
# scrape.  Its JSON cache is then staged with the rest of the run.
srun --overlap --nodes=1 --ntasks=1 --container-image="$IMAGE" \
  --container-mounts="${HOST_HOME}:${HOST_HOME},/home/scratch.sbak_coreai:/home/scratch.sbak_coreai" \
  python /home/scratch.sbak_coreai/dynamo-benchmark-perf-dashboard/dashboard.py "$DEST/dashboard.yaml" \
  > "$DEST/dashboard-build.log" 2>&1 || echo "[repro-A] dashboard build failed; raw sources retained" >&2
printf 'run_id=%s\nstaged_at_utc=%s\nsource=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" "$RUN_DIR" > "$DEST/STAGED.txt"
kill "$TEMPO_PID" 2>/dev/null || true
wait "$TEMPO_PID" 2>/dev/null || true
rm -rf "$AIPERF_ARTIFACT_ROOT"
rm -rf "$RUN_DIR"
echo "[repro-A] staged completed run to ${DEST} and removed temporary ${RUN_DIR}"
