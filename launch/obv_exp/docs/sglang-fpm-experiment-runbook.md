# SGLang FPM KV-router experiment runbook

This is the canonical, reproducible layout for the Qwen3.6 SGLang
prefill/decode experiments.

## Canonical locations

| Purpose | Location |
| --- | --- |
| Launch scripts | `~/scratch/dynamo/kv-router-routing-analysis/launch/` |
| Analysis scripts | `~/scratch/dynamo/kv-router-routing-analysis/` |
| Temporary active-run logs | `~/dynamo_repro/runs/<RUN_ID>/` |
| Completed-run logs and plots | `/home/scratch.sbak_coreai/dynamo_repro/completed_runs/<RUN_ID>/` |

Do not create experiment scripts in other directories. Do not submit an
automatic staging job or make frontend/worker jobs depend on each other.

## Scripts

Use only these launch scripts:

- `launch/infra-client-fpm.slurm`: etcd, NATS, two frontend replicas, FPM
  collector, and AIPerf client. This is the established client configuration:
  128 concurrency and 500 conversations **per frontend**, five turns per
  conversation, 200 shared-system-prompt tokens, 1000 fresh input tokens, and
  200 output tokens.
- `launch/sglang-qwen36-pd-role.slurm`: one SGLang worker process per GPU.
  Submit it once with `ROLE=prefill` and once with `ROLE=decode`, each on a
  separate GB200 node with four GPUs.

## Required observability configuration

Frontend submission:

```bash
DYN_LOG=info
DYN_REQUEST_TRACE=0
ROUTER_MODE=kv
ROUTER_REPLICA_SYNC=1
```

`DYN_LOG=info` emits `Selected worker` records, including candidate scores and
the chosen worker. `DYN_REQUEST_TRACE` is intentionally off: it adds per-request
frontend file I/O and is not required for router/FPM correlation.

The worker launcher must set:

```bash
DYN_FPM_TRACE=1
--enable-metrics
```

For the SGLang 0.5.11 runtime used here, `DYN_FPM_TRACE=1` is the required
Dynamo switch. It enables SGLang's internal FPM producer and the Dynamo relay
onto the `forward-pass-metrics` event-plane topic. Do **not** pass
`--enable-forward-pass-metrics`: this runtime's SGLang CLI does not recognise
that newer flag. `--enable-metrics` separately enables SGLang's metrics path.
The frontend's collector subscribes to prefill and decode FPM topics and writes
`logs/fpm-prefill.jsonl` and `logs/fpm-decode.jsonl`.

For Qwen3.6 SGLang P/D on the 0.5.11 runtime, both roles must use
`--page-size 1`. Prefill resolves to page size 1 on this path; allowing decode
to use page size 16 makes the first KV transfer fail. The launcher validates
the resolved value in every worker log before it remains available to serve.

For the session-affinity comparison, add:

```bash
ROUTER_SESSION_AFFINITY_TTL_SECS=900
```

Use the same model, wheelhouses, topology, and client defaults for both A/B
runs. The only A/B difference is the affinity TTL.

## Submission and completion procedure

1. Submit the frontend/client job and both role jobs independently, using one
   `RUN_ID`. Do not use Slurm dependencies between them. Workers poll
   `shared/infra_ip.txt` written by the frontend.
2. Monitor the client job and these readiness signals before considering the
   run valid:
   - each worker logs `FPM relay for dp_rank=...`;
   - the frontend collector logs that it is listening for
     `forward-pass-metrics`;
   - `fpm-prefill.jsonl` and `fpm-decode.jsonl` become nonempty;
   - frontend logs contain `Selected worker` records.
3. When the client completes, cancel the prefill and decode jobs explicitly.
4. From an ipp node, copy the completed run to scratch. This is a manual
   finalization step, not a staging job:

```bash
RUN_ID=<run-id>
srun --partition=general --nodelist=<ipp-node> --ntasks=1 --time=00:20:00 \
  bash -lc 'set -e; src="$HOME/dynamo_repro/runs/'"$RUN_ID"'"; \
  dst="/home/scratch.sbak_coreai/dynamo_repro/completed_runs/'"$RUN_ID"'"; \
  mkdir -p "$dst"; rsync -a "$src/" "$dst/"'
```

Do not delete the temporary source until the scratch copy is checked.

## Analysis

Run analysis only against the completed scratch directory:

```bash
RUN_DIR=/home/scratch.sbak_coreai/dynamo_repro/completed_runs/<RUN_ID>
python3 ~/scratch/dynamo/kv-router-routing-analysis/plot_routing.py
python3 ~/scratch/dynamo/kv-router-routing-analysis/plot_cross_frontend.py
python3 ~/scratch/dynamo/kv-router-routing-analysis/plot_router_vs_fpm.py
```

Expected outputs in `<RUN_ID>/analysis/`:

- `1_*` through `4_*`: router decisions, candidate costs, and selected workers.
- `7_*` through `9_*`: cross-frontend collision timelines, summaries, and
  sampled decision pairs.
- `10_router_vs_fpm_prefill.png` and `10_router_vs_fpm_decode.png`: router
  load versus engine FPM load. These require nonempty FPM JSONL files.

The log parser must accept the current `Selected worker` prefix (and may retain
compatibility with the previous `Routing decision` prefix).
