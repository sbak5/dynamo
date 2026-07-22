# KV routing cost model and cache-distribution findings

Reviewed against the current Dynamo checkout on 2026-07-14:

- branch: `sbak/router_log_rev`
- commit: `b4d2769420`
- the relevant routing-cost behavior also matches local `main`.

## Current candidate decision

For each eligible worker, the default selector computes:

```
cost = prefill_load_scale
     * max(raw_prefill_blocks - overlap_credit_blocks, 0)
     + potential_decode_blocks
```

With the default `router_temperature = 0`, it selects the minimum-cost worker and randomly breaks exact ties. Source: `lib/kv-router/src/scheduling/selector.rs:217-315`.

For the usual device-cache-only/default configuration, a useful approximation is:

```
cost_w ~= max((active_prefill_tokens_w + ISL_tokens) / block_size
              - cached_prefix_blocks_w, 0)
          + active_decode_blocks_w
          + additional_active_blocks_w
```

- `active_prefill_tokens` is prompt-side work currently tracked on that worker.
- `cached_prefix_blocks` is the longest device-local matching prefix.
- `potential_decode_blocks = active_decode_blocks + additional_active_blocks`.
  The latter is the incoming request's new footprint relative to *active*
  sequences on that worker, not necessarily inactive cache misses.

The complete overlap credit can include device, host-pinned, disk, and shared-cache tiers. Defaults are device credit `1.0`, host `0.75`, disk `0.25`, shared-cache credit disabled, and overlap-credit decay disabled.

## What this estimates correctly

`ISL_blocks - cached_prefix_blocks` is a reasonable first-order proxy for the new K/V projection and MLP work. Cache hits avoid recomputing K/V for the cached prefix. Existing active-prefill and decode terms add local queue/load pressure, so the selector is not pure maximum-overlap routing.

## Attention work missing from the linear prefix-credit model

Let `R` be total prompt blocks, `H` cached prefix blocks, and `M = R - H` the uncached suffix. New suffix queries still attend over the cached K/V prefix:

```
remaining attention ~= M * H + M^2 / 2
                     = (R^2 - H^2) / 2
```

The current router represents request-side prefill work roughly as `M` blocks. If `h = H / R`:

```
linear remaining fraction     = 1 - h
attention remaining fraction  = 1 - h^2
ratio                         = 1 + h
```

At 90% hit rate, the linear metric says 10% of request work remains, while the attention component retains about 19% of full-prompt attention work. Thus the linear proxy can overstate the cache benefit by nearly 2x for the attention component at very high hit rates.

Whether that affects end-to-end prefill latency depends on context length, model width, batching, and kernel behavior: MLP/projection work scales as `O(M d^2)`; residual attention scales as `O(M R d)`.

A calibrated latency model could instead use:

```
prefill_work ~= a * (R - H) + b * (R^2 - H^2) / 2
```

with `a` and `b` fitted from measured prefill latency.

## Replica synchronization and stale reservations

Within one frontend, the scheduler:

1. Projects worker loads from its local active-sequence state.
2. Selects a worker.
3. Books the chosen request locally before returning the response.

Source: `lib/kv-router/src/scheduling/queue.rs:648-755`.

When `router_replica_sync` is enabled, local booking emits an `AddRequest` lifecycle event asynchronously:

```
add_request_local(...)
spawn_publish_event(...)
```

The publisher runs in a spawned task and does not await peer acknowledgement. Peer frontends apply the event later in a background NATS subscription.

Sources:

- `lib/kv-router/src/sequences/multi_worker.rs:367-383,501-543`
- `lib/kv-router/src/sequences/replica_sync.rs:53-145,148-212`
- `lib/llm/src/kv_router/sequence.rs:31-53`

Consequently, two replicas can make a decision using the same pre-reservation state:

```
F1 reads S -> selects A -> books A locally
F2 reads S -> selects A -> books A locally
F1/F2 events arrive at peers later
```

There is no distributed reservation, compare-and-swap, consensus, or acknowledged handoff in this path. This is eventual consistency, not coordinated load placement.

`router_replica_sync` defaults to false. With it disabled, peers do not share active-sequence state at all; with it enabled, they eventually converge but can still race at routing time.

## Persistent cache affinity and concentration

KV overlap is queried from the KV indexer before scheduling. Physical worker cache `Stored` events add blocks to its prefix index; `Removed` or `Cleared` events remove them. Cache overlap can therefore outlive the request's active prefill/decode load.

Sources:

- overlap lookup and scheduling handoff: `lib/llm/src/kv_router.rs:540-624`
- KV indexer event handling: `lib/kv-router/src/indexer/radix_tree.rs:180-224`

This creates a time-scale mismatch:

```
active prefill/decode load  -> temporary penalty
resident KV prefix overlap  -> persistent credit, until eviction/removal
```

After a hot worker's transient queue drains, its higher overlap still lowers future related requests' cost. Those requests are then less likely to seed the same prefix on other workers:

```
initial placement on A
-> A gains/retains prefix overlap
-> A gets lower cost for related requests
-> A receives more related traffic
-> cache and traffic concentrate on A
```

Cross-frontend stale reservations can amplify the initial placement burst. The effect is prefix-specific: unrelated prompts do not receive overlap credit from unrelated cached content.

## KV-index propagation and replicated prefixes

The KV index is not directly synchronized by frontend-to-frontend routing decisions. Instead, a prefill or decode worker publishes physical-cache `Stored`, `Removed`, and `Cleared` events. Each frontend subscribes to the worker KV-event subject and applies those events to its own local routing-index replica:

```
worker creates/evicts KV block
-> worker publishes RouterEvent
-> frontend A updates local indexer
-> frontend B updates local indexer
-> frontend C updates local indexer
```

Sources:

- worker-local apply followed by event publication: `lib/llm/src/kv_router/publisher/sinks.rs:33-48`
- frontend KV-event subscription: `lib/llm/src/kv_router/indexer/recovery/subscriber.rs:33-103`

Thus every frontend holds a separate, eventually convergent copy of *routing metadata* (prefix hashes to worker IDs/DP ranks), not KV tensors and not a single shared radix tree. `router_replica_sync` is separate: it carries active request lifecycle/load state, not these KV-cache events.

If the same prefix is physically cached on multiple workers, that is valid cache replication. The radix tree uses one shared token-hash path and records each worker/rank that covers it. A matching request therefore receives an overlap score for every replica of the prefix, and the scheduler can choose among them using the rest of the cost function.

Repeated delivery of the same `Stored` event to one local indexer is different: the tree recognizes the worker already covers the prefix, reuses the existing edge/coverage, and treats it as an idempotent duplicate store rather than creating another copy. See `lib/kv-router/src/indexer/radix_tree.rs:250-279,384-397`.

The optional `router_predicted_ttl_secs` feature is an exception to the worker-event-only rule: it creates a short-TTL, predict-on-route side indexer local to the frontend that made the routing decision. It can therefore create temporary frontend-to-frontend differences until the actual worker KV events arrive.

## Proposed: frontend-local cache territories with NUMA-like promotion

This section is a proposed design, not current Dynamo behavior.

The goal is to distribute cache ownership without requiring a prefix-aware global ingress controller. Each frontend is assigned a primary worker cohort, while other cohorts are a secondary cache tier:

```
frontend A -> primary worker cohort A
frontend B -> primary worker cohort B
frontend C -> primary worker cohort C
```

Every frontend may still observe global KV metadata, but routes to its own cohort by default. Foreign workers are used selectively for a cache hit, overload spillover, or cache promotion. This permits useful physical KV replication: a common prefix can have one copy in each frontend's worker territory, so normal traffic remains local while foreign copies remain available for fallback.

### Two-stage cohort routing

Let `local` be the receiving frontend's cohort and `foreign` all other cohorts:

```
T_local  = min predicted service time among local workers
T_remote = min predicted service time among foreign workers with a cache hit
```

Use local workers by default. Spill to a foreign cached replica only when:

1. no local worker can admit the request (capacity or overload failure);
2. local predicted TTFT breaches a configured target and the foreign candidate can meet it; or
3. the foreign advantage exceeds a locality penalty and hysteresis:

```
T_remote + locality_penalty + hysteresis < T_local
```

Foreign spillover also needs a per-target-cohort concurrency/token budget. Otherwise many frontends can spill simultaneously to the same attractive remote replica and recreate the present hotspot.

Candidate ordering should normally be:

```
1. local workers with a cache hit
2. local workers without a hit (maintain or seed local cache ownership)
3. foreign workers with an existing cache hit (controlled spillover)
4. foreign cold workers (normally ineligible)
```

### Foreign hit -> local cache promotion

Treat a foreign cache hit as an opportunity to populate the receiving frontend's local cache territory, analogous to bringing a read-only line closer in a NUMA hierarchy. Full KV prefix blocks are deterministic and immutable for a given model, adapter, and multimodal context, so this is simpler than CPU cache coherence: it needs replication and eviction bookkeeping, not write invalidation.

```
local frontend sees local miss + foreign prefix hit
-> reserve target KV capacity in local cohort
-> pin source blocks on foreign prefill worker
-> transfer full prefix blocks over NIXL to local prefill worker
-> register copied blocks under the same SequenceHash lineage
-> publish Stored event for local worker
-> future requests prefer the new local replica
```

The transfer-control state should be explicit:

```
Absent -> Copying(source, target, lease) -> Present
```

Only one local promotion should run for a `(prefix, target worker)` at once. Other requests should wait for it, use the foreign replica, or run locally without the copy. A copied block must not become visible as a local cache hit until transfer completion and registration succeed. Source blocks must remain pinned for the transfer duration.

Promote only completed fixed-size blocks, not an active sequence's mutable tail block. The existing chained `SequenceHash`/block identity is the appropriate transfer and registration key.

There are two promotion modes:

- **Synchronous promotion:** transfer, then run the uncached suffix locally. Use when `transfer time + local residual prefill time` beats remote service.
- **Asynchronous warming:** serve the current request from the foreign worker while transferring its reusable prefix into the local cohort for later requests.

Promotion should be demand- and cost-gated. A first-order rule is to promote only if expected future local reuse justifies transfer time and target KV capacity:

```
copy_cost(H) + local_prefill_cost(R - H) < expected future local savings
```

This lets cache replication grow where a frontend has sustained demand, rather than as an accidental side effect of globally greedy routing.

## Existing counterweights and what is absent

Existing mitigations:

- active prefill and potential decode load are in the score;
- exact score ties are randomized;
- nonzero `router_temperature` samples instead of taking the strict minimum;
- `overlap_score_credit_decay` can reduce cache credit on prefill-loaded workers, but defaults to `0`;
- opt-in overload thresholds can make a worker ineligible based on prefill load or KV utilization;
- cache eviction/removal removes index overlap.

The core score has no direct term for:

- per-worker resident KV occupancy;
- cache diversity across workers;
- the marginal value of seeding or replicating a popular prefix;
- a globally consistent reservation count.

`total_kv_blocks` is logged by the selector but is not part of its cost. KV utilization is used only by the separate, opt-in overload filter.

## Bottom line

The current router greedily optimizes immediate cache reuse plus locally observed active load. It does not optimize cluster-wide cache distribution. For correlated-prefix workloads, eventually-consistent replica load state plus persistent overlap credit can produce a preferential-attachment loop that concentrates requests and KV state on a subset of workers.

## Implementation effort for frontend-local cache territories and promotion

### What Dynamo already provides

The repository contains two useful, but incomplete, building blocks:

- The general Rust block manager can import a remote worker's NIXL metadata, construct `RemoteBlock` descriptors, and issue a NIXL block copy. This is low-level memory movement; it does not decide which cached prefix is safe to copy or make it a cache hit afterwards.
- KVBM has more suitable distributed primitives: an endpoint can expose a specified set of blocks for RDMA pull, and a coordinated worker can pull named blocks from a remote worker after remote metadata is imported. It also has logical allocation and registration by `SequenceHash`.

The presently wired runtime paths are narrower than the proposed feature:

- Dynamo's prefill router creates a fresh request-scoped `BootstrapInfo`/handoff for **prefill -> decode**. It does not name a source prefill cache prefix or a target prefill worker.
- The vLLM integration passes `kv_transfer_params` through vLLM's NIXL or Mooncake connector for that prefill-to-decode handoff. It does not expose arbitrary cached-prefix cloning to another prefill worker.
- The KVBM vLLM connector uses KVBM through the normal vLLM scheduler's external-cache match/load/save callbacks. Its lower-level remote-pull/session APIs are not surfaced as a background prefill-to-prefill promotion service.

Consequently, this is not a router-only change. NIXL data movement is reusable, but the cache ownership, registration, and promotion lifecycle must be added above it.

### Incremental delivery plan

| Phase | Scope | Relative effort | Main dependency |
|---|---|---:|---|
| 1 | Frontend-local worker cohorts, local-first candidate selection, foreign-hit spill policy, per-foreign-cohort budget, and metrics; no copying | Medium | KV router/frontend only |
| 2 | Promotion coordinator: deduplicate `(prefix, target)` jobs, admission/lease/timeout/retry state, and transfer-aware routing accounting | Medium-high | frontend plus worker control API |
| 3 | Prefill-worker cache promotion for one runtime (recommend KVBM/vLLM first): source pin, target allocation, RDMA pull, atomic cache registration, and `Stored` event after commit | High | KVBM/vLLM runtime changes |
| 4 | Full production hardening: capacity/eviction interaction, cancellation, worker loss, multi-DP behavior, transfer fairness, calibration, and end-to-end benchmark coverage | High | multi-worker GPU test environment |

Phase 1 is independently valuable and is the low-risk first implementation. It directly addresses cross-frontend cache herding by biasing each frontend to its own worker cohort. It should be feature-gated and start with static cohorts, rather than a global prefix controller or periodic migration.

### New interfaces required for actual promotion

The promotion path needs a small control protocol between the frontend and prefill workers. For a chosen source prefix and target worker, it must support:

1. **Describe and pin source blocks.** Source returns only completed, immutable block hashes and transfer descriptors; it pins them until commit/cancel/expiry.
2. **Reserve and allocate target blocks.** Target reserves capacity before transfer and allocates physical slots that are not visible to prefix matching yet.
3. **Transfer and verify.** Target pulls the missing lineage range over NIXL/KVBM and reports completion. The copy set is `local_hit + 1 .. final_complete_prefix`, not merely the portion newly computed on the foreign worker.
4. **Atomically register.** Target registers the copied slots under the canonical `SequenceHash` lineage, then emits its normal `Stored` events. Only this successful event makes the target eligible as a cache-hit replica.
5. **Abort safely.** On failure, target releases reserved slots and source releases its pin. A lease expiry handles frontend or worker failure.

This interface should live at the worker/cache-manager layer, not in the radix indexer. The indexer is a directory of cache facts; it cannot transfer tensors or guarantee that a physical cache block remains resident.

### Main engineering risks

- **Runtime portability.** The implementation differs by runtime and connector. KVBM already has the closest primitives; generic vLLM NIXL, SGLang, and TRT-LLM each need an equivalent capability or adapter. Do not promise one shared implementation before proving the first backend.
- **Cache-manager correctness.** A copied tensor is unusable until the target engine's own prefix-cache manager binds it to its physical block ID and hash lineage. Eviction, source pinning, and the mutable tail must be handled by that manager.
- **Control-plane races.** The promotion coordinator needs idempotency keyed by `(model/cache namespace, prefix lineage, target worker)`, leases, and a visible `Copying` state. Otherwise concurrent frontends recreate redundant transfers.
- **Load and fabric contention.** Copying can harm TTFT. It needs a separate transfer budget and a calibrated transfer-cost estimate; foreground serving should preempt or cap warming.
- **Correct cache namespace.** The key must include all cache-identity inputs already encoded by the runtime, including model/version, adapter, cache salt, multimodal context, and block layout. Token hashes alone are insufficient if those differ.

### Suggested staffing and timeline shape

For one well-supported backend (KVBM + vLLM), Phase 1 is roughly a focused router change; Phases 2 and 3 together are a multi-component feature requiring one router/control-plane owner and one runtime/KVBM owner. Treat it as several engineering weeks including a GPU integration environment, not a small patch. Supporting additional runtimes should be separate follow-on work, because their physical cache-manager APIs differ.

The highest-leverage sequence is: ship and measure Phase 1, add the promotion control protocol behind a feature flag, prove asynchronous warming for completed blocks on KVBM/vLLM, then decide whether synchronous promotion meets TTFT goals. This preserves the proposed architecture while avoiding a dependency on a new global ingress/prefix-partition controller.

## Revision: FlexKV already supplies most proposed cache mechanics

The earlier frontend-local promotion design assumed that Dynamo would need to build a prefill-worker-to-prefill-worker copy protocol. That assumption is no longer appropriate when FlexKV is enabled.

FlexKV already provides the machinery that proposal needed:

- an external cache hierarchy (GPU, CPU, SSD, and optional remote tier);
- a distributed prefix directory, represented locally as snapshots of global metadata;
- remote lookup, lease-protected source validity, allocation, and Mooncake RDMA transfer;
- local cache insertion/removal and tier-local eviction;
- vLLM connector support that loads a matching external prefix into the selected worker's pre-allocated GPU slots;
- Dynamo KV events when `DYNAMO_USE_FLEXKV=1`, so the normal Dynamo indexer can learn FlexKV cache insertions and removals.

The actual execution path is:

```
Dynamo router chooses worker W
-> vLLM on W checks its native GPU prefix state and FlexKV external cache
-> on external hit, vLLM allocates W's destination GPU blocks
-> FlexKV transfers matching blocks into those slots
-> W resumes the request from the loaded prefix
```

The remote GET is asynchronous in implementation but synchronous for the *current request*: that request cannot execute attention over the prefix until the transfer finishes. FlexKV PUT/offload after completion is asynchronous.

This means FlexKV removes the need for Dynamo to find a foreign prefill worker and copy blocks explicitly. The router may select a worker for load or locality; FlexKV can satisfy an external hit after dispatch. In a disaggregated vLLM deployment, FlexKV runs on the prefill side through `PdConnector([FlexKVConnectorV1, NixlConnector])`; NIXL remains responsible for the separate prefill-to-decode handoff.

### Two distinct directories

Do not conflate the data and metadata layers:

```
FlexKV index + storage pools
  -> owns actual KV blocks and their CPU/SSD/remote eviction

Dynamo frontend radix index
  -> routing metadata: prefix -> worker ID + reported medium
```

FlexKV's `BlockStored`/`BlockRemoved` events are collected only when `DYNAMO_USE_FLEXKV=1`. They let Dynamo learn local FlexKV cache facts, but Dynamo does not query FlexKV's global directory, source lease state, or remote transfer cost before routing.

### Revised problem statement

With FlexKV, the remaining incremental problem is a **tier-aware routing cost**, not cache-coherence implementation:

```
known local GPU hit
known local FlexKV CPU/SSD hit
unknown-to-Dynamo but possible FlexKV remote hit
true recompute miss
```

The current router cannot distinguish the last two before dispatch. It should therefore avoid forcing requests toward an existing cache owner merely because its own index has overlap. Initial routing can favor runtime load and known fast local tiers; FlexKV then resolves external hits. A later integration could expose an inexpensive FlexKV match/estimated-transfer-cost query to make the selector tier-aware.

The old frontend-local cohort, controlled foreign spillover, and asynchronous warming ideas remain optional *placement policies*, not prerequisites. They should be considered only if FlexKV experiments still show cache-driven concentration or unacceptable synchronous remote-GET TTFT.

### KVBM versus FlexKV: decision for this investigation

KVBM and FlexKV are overlapping hierarchical KV-cache systems, not additive requirements. KVBM is Dynamo-native and has low-level remote-block/session primitives, but the current vLLM `DynamoConnector` does not expose those primitives as the complete cross-worker remote-reuse workflow needed here. FlexKV already exposes that workflow through `FlexKVConnectorV1`: distributed metadata snapshots, remote lookup, leases, Mooncake transfer, target-slot loading, and cache events.

Therefore, **use FlexKV as the cache substrate for this KV-router investigation**. Do not implement a new KVBM-based promotion/coherence service before measuring FlexKV. KVBM remains a possible future backend or consolidation target, but it is out of scope for the first experiment series.
