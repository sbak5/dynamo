# FlexKV experiment runbook for the Dynamo KV-router reproducer

## Purpose

Measure whether FlexKV external KV reuse lets Dynamo route prefill requests for load balance rather than repeatedly routing to the worker already known to hold a prefix.

This runbook is intentionally staged. First prove local FlexKV integration. Only then add distributed remote reuse, which needs additional services and RDMA/Mooncake support.

## Scope decision: FlexKV, not KVBM

This experiment uses **FlexKV as the sole hierarchical/distributed cache substrate**. Do not enable `DynamoConnector`/KVBM alongside it and do not build a KVBM promotion service for this run. Both systems overlap in role, but FlexKV already exposes the required vLLM remote-reuse flow: global metadata snapshots, remote matching, leases, Mooncake transfer, target-slot loading, and Dynamo KV events. KVBM is not the baseline path and is out of scope unless FlexKV proves insufficient.

## What changes relative to the current reproducer

The current worker job uses NIXL only:

```json
{"kv_connector":"NixlConnector","kv_role":"kv_both"}
```

For FlexKV disaggregated serving:

- Decode workers stay NIXL-only.
- Prefill workers use Dynamo's `PdConnector`, with FlexKV first and NIXL second.
- The frontend remains `--router-mode kv --enforce-disagg`.
- Prefill workers retain the ZMQ KV-event publisher.

`DYNAMO_USE_FLEXKV=1` enables FlexKV's Dynamo KV-event bridge. It is necessary but not sufficient: the FlexKV connector and a CPU cache allocation are also required.

## Stage 0 — preflight; do not start a workload yet

Run this inside the exact runtime image used by `dynamo_repro_workers.slurm` (`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1` unless overridden):

```bash
python3 - <<'PY'
import importlib.util
print("flexkv package:", importlib.util.find_spec("flexkv"))
print("vLLM FlexKV connector:", importlib.util.find_spec(
    "vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector"))
PY
python3 -c 'import vllm; print("vLLM version:", vllm.__version__)'
```

Expected: both module lookups resolve. If either is missing, stop. The stock reproducer image has not been verified to contain the FlexKV package/connector combination; build or obtain a compatible image first. Do not interpret an import failure as a routing failure.

Also record the image digest, vLLM version, Dynamo commit/wheel version, GPU type, and block size in the experiment log.

## Stage 1 — local FlexKV offload/reuse, single prefill worker

Goal: verify that a prefill worker can PUT completed prefixes into FlexKV's local CPU tier and GET them on a later request.

Use one prefill worker and one decode worker initially. Start the prefill worker with:

```bash
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20200 \
CUDA_VISIBLE_DEVICES=<prefill-gpu> \
python3 -m dynamo.vllm \
  --model "$MODEL" --enforce-eager \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
```

Keep decode NIXL-only:

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=20100 \
CUDA_VISIBLE_DEVICES=<decode-gpu> \
python3 -m dynamo.vllm \
  --model "$MODEL" --enforce-eager \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

Send the same long prompt twice through one frontend. Confirm all of the following before continuing:

1. worker logs show FlexKV initialized and its KV tensors registered;
2. the first request records a FlexKV PUT/offload of complete blocks;
3. the second request reports a nonzero FlexKV matched-token count / GET;
4. Dynamo frontend/indexer logs receive `Stored` events; and
5. output is correct and no request remains waiting for remote KV.

If GET tasks stall, rerun only this diagnostic with `FLEXKV_SYNC_GET=1`. This forces a synchronous connector wait and distinguishes transfer-completion wiring from cache matching. Do not use it for performance results.

## Stage 2 — local FlexKV with the existing multi-worker reproducer

Goal: compare the baseline current router with local FlexKV enabled, while keeping all remote/distributed FlexKV features off.

Copy `dynamo_repro_workers.slurm` to a run-specific file; do not overwrite the known baseline. In the **prefill loop only**:

1. export `DYNAMO_USE_FLEXKV=1` and `FLEXKV_CPU_CACHE_GB=<capacity>`;
2. replace the NIXL-only JSON with the `PdConnector` JSON from Stage 1;
3. keep each prefill worker's ZMQ KV-event publisher on its unique existing port;
4. leave decode workers NIXL-only;
5. preserve the existing NIXL/UCX environment settings for P/D transfer.

Keep all workload parameters identical to baseline: model, request trace, number of frontend replicas, number of prefill/decode workers, cache block size, GPU memory utilization, and warm-up length.

Collect per run:

- request TTFT and end-to-end latency, including P50/P95/P99;
- routing-decision logs: selected prefill worker, overlap, raw prefill blocks, decode blocks, score, and tie count;
- distribution of requests and prefill tokens by worker;
- FlexKV metrics: get requests, FlexKV-matched tokens, GPU-matched tokens, put requests, failed requests, and transfer latency if available;
- Dynamo KV event/index state, including the reported cache medium;
- worker GPU memory, CPU memory, and eviction/removal events.

Run at least: baseline NIXL-only, FlexKV-cold, and FlexKV-warm. The warm run should replay the same correlated-prefix workload only after cache population is confirmed.

Interpretation:

- If worker distribution improves while TTFT remains acceptable, FlexKV is reducing cache-owner affinity.
- If routing remains concentrated, check whether Dynamo's cache-event overlap credit is still dominating its load term.
- If TTFT rises, distinguish local CPU GET cost from router queueing; do not attribute both to FlexKV.

## Stage 3 — distributed FlexKV remote reuse

Goal: prove that a worker selected without a local prefix can fetch it from another node/worker through FlexKV.

This is a different infrastructure experiment, not a flag change. Before launching workers, provision and validate:

1. a FlexKV build with distributed/P2P support (`FLEXKV_ENABLE_P2P=1` in FlexKV's documented setup);
2. Mooncake Transfer Engine and usable RDMA devices;
3. Redis services for FlexKV global metadata and Mooncake metadata, with reachable addresses/authentication;
4. FlexKV distributed configuration: node identity, local IP, peer/metadata endpoints, and cache capacities;
5. compatible model, KV layout, block size, dtype, and cache namespace across prefill workers.

Use a two-node or otherwise independently addressable-worker setup. A single host can validate process wiring but is not evidence of network remote reuse.

Validation sequence:

1. Send a long shared prefix only to prefill worker A, then wait for its FlexKV PUT and distributed metadata publication.
2. Confirm worker B's FlexKV local snapshot can match that prefix before dispatching the verification request.
3. Send the same prefix to a request deliberately placed on B (use a temporary routing constraint or take A out of the candidate set; do not rely on random selection).
4. Confirm a remote GET/lease/Mooncake transfer in B's FlexKV logs, successful request output, and no recomputation of the transferred prefix.
5. Repeat once after A's GPU-native entry is evicted but its FlexKV backing entry remains, then once after the backing entry is evicted. The latter must cleanly fall back to recompute rather than return incorrect output.

## Decision table

| Result | Next action |
|---|---|
| Stage 1 fails | Fix image/package/connector setup; do not change router code. |
| Local FlexKV works, but distributed mode unavailable | Measure local offload only; do not claim remote-cache routing benefits. |
| Distributed GET works and load distribution improves | Use FlexKV as the cache substrate; evaluate tier-aware router cost next. |
| Distributed GET works but TTFT is too high | Compare direct cache-owner routing vs remote GET; investigate transfer bandwidth and cache tiers. |
| Requests still concentrate despite FlexKV | Inspect Dynamo's overlap credit and test cache-tier-aware scoring or a lower cache-credit weight. |

## Guardrails

- Never compare runs with different model image, vLLM version, block size, or cache namespace.
- Use only completed fixed-size blocks as reusable evidence; do not infer a hit from a mutable prompt tail.
- Keep `FLEXKV_SYNC_GET=0` for measured runs. It is a diagnosis knob, not a performance configuration.
- Treat Dynamo's radix tree as routing metadata only. FlexKV owns remote lookup, leasing, transfer, and data eviction.
- Do not add frontend-local/foreign-worker promotion code unless these experiments show that FlexKV plus tier-aware routing remains insufficient.
