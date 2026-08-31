#!/bin/bash
# Real P/D lifecycle negative-path probe. Invoked only by the optional
# CLIENT_PROBE_SCRIPT path in vllm-qwen36-frontend-client-lifecycle-dev.slurm.
set -euo pipefail

endpoint="http://${INFRA_IP}:8000/v1/chat/completions"
prompt=""
# Keep the active prefill busy longer than the frontend disconnect monitor's
# polling interval.  The closure run supplies a 16384-token worker context;
# this prompt plus its intentionally short output stays below that limit.
for ((index = 0; index < 5000; index++)); do
  prompt+="queue pressure token "
done
payload() {
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":16,"stream":true}' "$MODEL" "$prompt"
}

# Submit a burst.  Arrival order at the frontend is deliberately unspecified:
# one request becomes active, one occupies the one-entry scheduler queue, and
# one receives the overload response.  Identify the rejected connection from
# its actual status, then close both remaining connections immediately.  Thus
# the queued request is cancelled before it can be admitted regardless of
# which client process the scheduler happened to enqueue.
for index in 1 2 3; do
  curl --silent --show-error --no-buffer \
    --output "$LOGDIR/queue-pressure-$index.jsonl" \
    --write-out '%{http_code}\n' \
    -H 'content-type: application/json' "$endpoint" \
    -d "$(payload)" >"$LOGDIR/queue-pressure-$index.status" &
  pids[$index]=$!
done

rejected_index=""
for _ in $(seq 1 100); do
  for index in 1 2 3; do
    if grep -qx '529' "$LOGDIR/queue-pressure-$index.status" 2>/dev/null; then
      rejected_index="$index"
      break 2
    fi
  done
  sleep 0.05
done
if [ -z "$rejected_index" ]; then
  echo "expected one queue rejection HTTP 529" >&2
  exit 1
fi
echo "[queue-negative-client] injected queue rejection: HTTP 529"

for index in 1 2 3; do
  [ "$index" = "$rejected_index" ] && continue
  kill -TERM "${pids[$index]}" 2>/dev/null || true
done
for index in 1 2 3; do
  wait "${pids[$index]}" 2>/dev/null || true
done
sleep 5
echo "[queue-negative-client] injected cancellation while queued"
