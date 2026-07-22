# AgentX / Dynamo experiment session — 2026-07-22

## Goal

Run SemiAnalysis AgentX (Weka agentic-session replay) against Dynamo vLLM
prefill/decode disaggregation while retaining router decisions, FPM, worker KV
metrics, AIPerf JSONL, and OTEL tracing.

## AgentX client validation

- Source: `ai-dynamo/aiperf`, branch `ajc/agentx`.
- Isolated AIPerf environment: `/home/scratch.sbak_coreai/aiperf-agentx/venv`.
- Hugging Face cache: `/home/scratch.sbak_coreai/aiperf-agentx/hf`.
- Corpus used: `semianalysis_cc_traces_weka_with_subagents_060826_256k`.
- Corpus validation completed: 391 traces; a sampled trace used 64-token KV
  blocks, and another trace contained 600 requests with 6 subagent markers.
- AgentX scenario: `--scenario inferencex-agentx-mvp`; it locks agentic replay,
  `first_turn_prefix` cache busting, trace timing, and has a 900-second minimum.

## Intended serving configuration

- Model: `Qwen/Qwen3-32B-FP8`.
- Topology: four single-GPU prefill workers and four single-GPU decode workers
  on separate nodes.
- Router: `--router-mode kv --router-replica-sync` with session affinity off.
- Client: two frontend replicas, AgentX, concurrency 16 per frontend,
  900-second duration.
- Collected data: frontend routing logs, FPM JSONL, worker Prometheus/KV JSONL,
  raw AIPerf server-metric JSONL, and Tempo/OTEL traces.

## Context constraint

`Qwen3-32B-FP8` declares a native 40,960-token context. Against the 256k
AgentX corpus this retains only 1/391 traces; 131,072 would retain 137/391 and
262,144 retains all 391. Do not claim a native-context run is broad AgentX
coverage.

An attempted `--rope-scaling` extension was removed. The current
`dynamo.vllm` entrypoint accepts `--max-model-len` but rejects
`--rope-scaling`; no subsequent scripts contain that option.

## Launch scripts

Dedicated AgentX copies:

```text
~/scratch/dynamo/launch/agentx/infra-client-agentx.slurm
~/scratch/dynamo/launch/agentx/worker-vllm-agentx-role.slurm
```

The worker script supports an optional `VLLM_MAX_MODEL_LEN`; leave it unset
for the Qwen native-context validation run.

The dashboard-source copies remain at:

```text
/home/scratch.sbak_coreai/dynamo-benchmark-perf-dashboard/launch/
```

## Attempts and outcomes

| Jobs | Placement | Outcome |
| --- | --- | --- |
| 1627357–1627359 | GB200 | Worker launch rejected unsupported `--rope-scaling`; frontend timed out waiting. |
| 1632601, 1632609, 1632733 | GB200 prefill + GB300 decode | Cancelled after decode placement delay; do not use GB300. |
| 1632940–1632942 | GB300 | Repeated `NODE_FAIL` / launch staging failures; cancelled. |
| 1634089–1634091 | GB200 `gb-nvl-147-compute08/09` | Both worker jobs externally terminated before the batch script: `RaisedSignal:53 (Real-time_signal_19)`, `DerivedExitCode=0:0`; no worker logs were written. |

No AgentX request has successfully been sent to Dynamo in this session.

## Placement policy for the next retry

- Do **not** use GB300.
- Use two healthy GB200 nodes, or 8xH100 / GH200 nodes if available.
- Keep prefill and decode on separate nodes.
- Start the frontend/client independently (no Slurm dependency), but begin
  AIPerf only after all prefill and decode workers have registered.
- Check the Slurm batch stdout/stderr and worker registration immediately after
  allocation; if a job has `RaisedSignal:53` before script output, select a
  different node pair rather than changing Dynamo/AIPerf settings.
