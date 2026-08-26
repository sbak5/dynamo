#!/bin/bash
#SBATCH --job-name=qwen36-otel-jsonl-collector
#SBATCH --partition=gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=03:00:00

# External OTLP JSONL collector for the unmodified Qwen 3.6 three-job launcher.
# Submit this job first with the same RUN_ID as frontend, prefill, and decode.
set -eu

RUN_ID=$RUN_ID
HOST_HOME=$HOME
RUN_DIR=$HOST_HOME/dynamo_repro/runs/$RUN_ID
SHARED=$RUN_DIR/shared
LOGDIR=$RUN_DIR/logs
STAGE_ROOT=$HOST_HOME/scratch/dynamo_repro/completed_runs
OTELCOL_BIN=$HOST_HOME/dynamo_repro/tools/otelcol-contrib-0.122.1-linux-arm64/otelcol-contrib
OTELCOL_TEMPLATE=$HOST_HOME/scratch/dynamo-lifecycle-dev/launch/kv-router-routing-analysis/launch/lifecycle-otel-jsonl.yaml
OTELCOL_CONFIG=$RUN_DIR/otelcol.yaml
OTEL_TRACES_FILE=$LOGDIR/otel-traces.jsonl
OTEL_LOGS_FILE=$LOGDIR/otel-logs.jsonl
COLLECTOR_PID=

mkdir -p "$SHARED" "$LOGDIR"

stage_jsonl() {
  dest=$STAGE_ROOT/$RUN_ID/logs
  mkdir -p "$dest"
  for file in otel-traces.jsonl otel-logs.jsonl otelcol.log; do
    [ ! -f "$LOGDIR/$file" ] || cp "$LOGDIR/$file" "$dest/$file"
  done
}

cleanup() {
  status=$?
  if [ -n "$COLLECTOR_PID" ]; then
    kill "$COLLECTOR_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$COLLECTOR_PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$COLLECTOR_PID" 2>/dev/null && kill -KILL "$COLLECTOR_PID" 2>/dev/null || true
    wait "$COLLECTOR_PID" 2>/dev/null || true
  fi
  stage_jsonl
  touch "$SHARED/otel_collector_done"
  exit "$status"
}
trap cleanup EXIT INT TERM

[ -x "$OTELCOL_BIN" ] || { echo "missing OTel Collector: $OTELCOL_BIN" >&2; exit 2; }
[ -f "$OTELCOL_TEMPLATE" ] || { echo "missing collector config: $OTELCOL_TEMPLATE" >&2; exit 2; }

INFRA_IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^docker|^br-|^virbr/ {print $4}' | cut -d/ -f1 | head -1)
[ -n "$INFRA_IP" ] || { echo "could not resolve collector node IP" >&2; exit 2; }
sed -e "s|__OTEL_TRACES_FILE__|$OTEL_TRACES_FILE|g" \
    -e "s|__OTEL_LOGS_FILE__|$OTEL_LOGS_FILE|g" \
    "$OTELCOL_TEMPLATE" > "$OTELCOL_CONFIG"

"$OTELCOL_BIN" --config="$OTELCOL_CONFIG" > "$LOGDIR/otelcol.log" 2>&1 &
COLLECTOR_PID=$!
until curl -sf http://127.0.0.1:13133/ >/dev/null 2>&1; do
  kill -0 "$COLLECTOR_PID" 2>/dev/null || { cat "$LOGDIR/otelcol.log" >&2; exit 1; }
  sleep 1
done

printf 'http://%s:4317\n' "$INFRA_IP" > "$SHARED/otel_collector_endpoint.txt"
touch "$SHARED/otel_collector_ready"
echo "[otel-jsonl-collector] run=$RUN_ID endpoint=http://$INFRA_IP:4317"

while [ ! -e "$SHARED/stop_workers" ]; do
  kill -0 "$COLLECTOR_PID" 2>/dev/null || { cat "$LOGDIR/otelcol.log" >&2; exit 1; }
  sleep 3
done

# Let role shutdown flush final spans before stopping the collector.
sleep 10
