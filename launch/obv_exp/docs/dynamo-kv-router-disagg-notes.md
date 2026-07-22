# Dynamo Call Stack, Disaggregated Serving, and KV-Aware Routing

Notes on how a request flows through the Dynamo frontend, how prefill/decode
disaggregation works, and how the KV router uses token-hash-based blocks to
make routing decisions. File:line references point at the repo state as of
2026-07-07 (`ai-dynamo/dynamo`, branch `main`).

## 1. Startup / init sequence (before any request)

- **HTTP server**: `lib/llm/src/entrypoint/input/http.rs:24` `run()` builds
  the service via `service_v2::HttpService::builder()...build()`. The actual
  axum/hyper server binds and serves in
  `lib/llm/src/http/service/service_v2.rs:625` `HttpService::run` →
  `run_with_listener` → `axum::serve(...)` (`:775`). OpenAI-compatible routes
  (incl. `chat_completions_router`, `openai.rs:2675`) are registered here.
- **Watcher**: `lib/llm/src/discovery/watcher.rs:131` `ModelWatcher`,
  constructed inside `run_watcher` (`http.rs:176/198`). Consumes a
  `DiscoveryStream` and reacts to model/worker `Added`/`Removed` events
  (`watcher.rs:282`).
- **Endpoint discovery task**: also inside `run_watcher` (`http.rs:176-237`).
  `discovery.list_and_watch(DiscoveryQuery::AllModels, ...)`
  (`http.rs:212-217`) opens the etcd-backed discovery stream of worker/model
  instances. A companion `_endpoint_enabler_task` (`http.rs:225-230`)
  consumes `ModelUpdate` events and calls `update_http_endpoints`, while
  `_watcher_task` feeds newly discovered worker endpoints into
  `ModelManager` — this becomes the candidate-worker pool that
  `KvPushRouter`/`KvScheduler` select from later.

## 2. Per-request call path

1. **HTTP entry** — `handler_chat_completions` (`http/service/openai.rs:1224`)
   → `chat_completions` (`:1747`) → `engine.generate(request)` (`:1857`).
2. **Pipeline wiring** — built in `entrypoint/input/common.rs:478-496`:
   `frontend → preprocessor_op → migration → token_backend → prefill_op → backend`
   (mirrored backward for streaming responses back).
3. **Preprocessing/tokenization** — `OpenAIPreprocessor`
   (`preprocessor.rs:243`) applies the chat template and tokenizes,
   producing a `PreprocessedRequest`.
4. **Disaggregation decision (prefill dispatch)** —
   `PrefillRouter::generate` (`kv_router/prefill_router/mod.rs:159`). If
   disagg is inactive, falls through to decode-only (aggregated) serving.
   Otherwise clones the request with `max_tokens=1` (`:197`) — prefill's real
   job is populating the KV cache, the 1 emitted token is a side effect — and
   calls `select_and_dispatch_prefill` (`:227`) with
   `prepare_prefill_dispatch` (`:328`) to run prefill on a chosen worker and
   arrange the KV transfer. On completion it sets
   `decode_req.routing_mut().prefill_worker_id` and calls `next.generate`
   (`:319`) to continue on the decode worker.
5. **KV-aware worker selection** — `KvPushRouter::generate`
   (`kv_router/push_router.rs:446`) → `select_with_affinity`/`select_worker`
   (`push_router/selection.rs:114`) → `KvScheduler::schedule*`
   (`scheduler.rs:162/207`), scoring candidate workers using KV-block overlap
   from `indexer/lookup.rs:321` `merge_overlap_scores`.
6. **Dispatch** — `dispatch_selection` (`push_router.rs:413`) sends the
   request to the chosen worker over the etcd-discovered NATS/TCP data
   plane.
7. **Response streaming** — flows back through the pipeline's backward edges
   (`common.rs:492-495`) and out as SSE via `chat_completions` in
   `openai.rs`.

```
etcd watcher/discovery
        │
        v
HTTP entry (openai.rs) ─▶ preprocessor (tokenize) ─▶ PrefillRouter
                                                          │
                                          disagg active?  │  inactive → decode-only
                                                 yes ▼
                                     prefill dispatch (max_tokens=1,
                                     KV transfer bootstrap) on prefill worker
                                                 │
                                                 v
                                     decode continues on decode worker
                                     (prefill_worker_id set on request)
                                                 │
                                                 v
                              KvPushRouter + KvScheduler (KV-overlap-aware pick)
                                                 │
                                                 v
                                  dispatch over NATS/etcd-discovered transport
                                                 │
                                                 v
                                    stream tokens back through backward
                                    edges → SSE response to client
```

## 3. Why disaggregate prefill and decode

An LLM request has two phases with very different compute profiles:

- **Prefill**: processes the whole input prompt in one pass — a single big
  batched matmul across all prompt positions computes Q/K/V and attention
  for every token at once (causal-masked), populating the KV cache.
  **Compute-bound**, batches efficiently, GPU-friendly.
- **Decode**: generates one token at a time. Each step embeds/projects only
  the *new* token, appends its K/V to the cache, and attends the new query
  over the full (old + new) cache. **Memory-bandwidth-bound** — tiny FLOPs
  per step, but the whole growing KV cache must be read from HBM every step.
  Cache size grows linearly with sequence length and is freed once a
  sequence finishes.

Running both phases on the same worker causes interference: a long prefill
for one request blocks/delays decode steps for others sharing that GPU,
hurting inter-token latency (ITL/TPOT), while decode's small, frequent steps
don't fully utilize compute that prefill wants to saturate. Splitting them:

- Prefill workers batch aggressively, scaled/sized for throughput.
- Decode workers stay latency-focused, avoiding prefill-induced stalls, and
  can use different parallelism configs tuned to memory bandwidth.

Cost of the split: the prefill worker's KV cache must be transferred (via
NIXL/RDMA) to the decode worker — this is what `prepare_prefill_dispatch`
and the bootstrap/KV-transfer info in `PrefillRouter` set up.

## 4. KV cache mechanics (what's actually being transferred/cached)

- Every token produces Query/Key/Value vectors via QKV projection.
- **Prefill**: batched matmul over all prompt tokens → K/V for every
  position, cached per layer, per head: shape
  `(num_heads, seq_len, head_dim)`.
- **Decode**: each step embeds only the new token, computes its Q/K/V,
  appends K/V to the cache (`concat`), and attends the new Q against the
  *entire* cache (old + new) — no recomputation of earlier tokens' K/V.
- The KV cache is what's transferred from prefill worker → decode worker in
  disaggregated serving, and what's reused across decode steps without
  redoing the expensive batched prefill matmul.

## 5. Worked example: prefill batching and KV reuse

Single sequence, tokens `x_1, x_2, x_3` (e.g. "The", "cat", "sat"), one
attention head. For token *i*: `q_i = x_i W_Q`, `k_i = x_i W_K`,
`v_i = x_i W_V`.

### Prefill — all terms of the sum computed together

Causal attention output for position *i* is a weighted sum over all
positions `j ≤ i`:

```
attn(i) = Σ_{j≤i} softmax_j(q_i · k_j / √d) · v_j
```

Prefill computes `q_i, k_i, v_i` **for every i = 1..3 in one shot**
(one matmul over the whole prompt, not three sequential ones):

```
(q_1, q_2, q_3) = (x_1, x_2, x_3) W_Q     [same for K, V]
```

then evaluates `attn(1), attn(2), attn(3)` together, each just masking out
`j > i` from the same shared `k_j, v_j` terms. After this pass, the model
**caches** every `k_j, v_j` it just computed:

```
cache = { k_1, v_1, k_2, v_2, k_3, v_3 }
```

`attn(3)` (the last real token, "sat") is what predicts the first generated
token — this is why prefill dispatch uses `max_tokens=1`
(`prefill_router/mod.rs:197`): the real deliverable is `cache`, the emitted
token is a side effect. `cache` is exactly what's shipped over NIXL/RDMA to
the decode worker.

### Decode — reusing the cached terms instead of recomputing them

Decode worker receives `cache = {k_1,v_1,k_2,v_2,k_3,v_3}` and generates
token 4 ("on"):

```
q_4 = x_4 W_Q,  k_4 = x_4 W_K,  v_4 = x_4 W_V     ← only 1 new term computed

attn(4) = softmax(q_4·k_1/√d)·v_1
        + softmax(q_4·k_2/√d)·v_2
        + softmax(q_4·k_3/√d)·v_3
        + softmax(q_4·k_4/√d)·v_4
          └─────────── reused from cache ───────────┘   └─ new ─┘
```

Three of the four terms in that sum are **read straight from `cache`** —
`k_1,v_1,k_2,v_2,k_3,v_3` are never recomputed. Only `q_4,k_4,v_4` are
freshly derived, then `k_4,v_4` get appended: `cache ← cache ∪ {k_4,v_4}`.
Generating token 5 repeats this with a 5-term sum, reusing 4 cached
terms and computing 1 new one — the cache (and the sum) grows by exactly
one term per generated token.

### Why this maps directly onto disaggregation

- Prefill computes the *entire* `{k_j, v_j}` set for the prompt in one
  parallel pass — dense, compute-bound, batches well across concurrent
  requests.
- Decode's `attn(i)` sum has `i` terms, but only the last one is ever new;
  the other `i-1` are cache reads. Compute per step is ~constant and tiny;
  what grows is the number of cache terms that must be read from memory —
  hence memory-bandwidth-bound, not compute-bound.
- These are different bottlenecks needing different scaling/hardware
  tuning, which is the rationale for separate prefill and decode worker
  pools (§3): prefill workers optimize for parallel throughput on the
  `{k_j,v_j}`-generation pass, decode workers optimize for fast repeated
  reads of an ever-growing `cache`.

## 6. KV-aware routing: blocks, hashes, and overlap score

### Blocks are fixed-size token chunks, not per-token KV values

```rust
// lib/llm/src/local_model.rs:34
const DEFAULT_KV_CACHE_BLOCK_SIZE: u32 = 16;
```

A prompt's tokens are sliced into consecutive chunks of `block_size` tokens
(configurable per model/engine). Chunking logic:
`compute_block_hash_for_seq(tokens, kv_block_size, options)`
(`lib/kv-router/src/protocols.rs:85-140`).

### Block identity = hash of token IDs (not KV tensor values)

```rust
// lib/kv-router/src/protocols.rs:565
pub struct LocalBlockHash(pub u64);
```

Computed via XXH3 over the token IDs in that chunk (`hash_block_no_mm`,
`protocols.rs:38-58`, `XXH3_SEED = 1337`), optionally mixing in multimodal
or LoRA metadata. Two requests sharing the same first N tokens produce the
same block hash — no tensor comparison needed.

**Prefix chaining**: block *i*'s hash folds in the previous block's hash
(`compute_next_seq_hash(parent_seq_hash, current_block_hash)`,
`protocols.rs:147-151`), producing a rolling `SequenceHash`
(`protocols.rs:572`) where block *i*'s hash encodes "blocks 0..i in this
exact order." Matching is therefore prefix-sensitive: a divergence at block
1 breaks matches for block 2 onward even if later raw tokens coincide.

### Overlap score = longest contiguous matching-block prefix, per worker

- `RadixTree` (`lib/kv-router/src/indexer/radix_tree.rs:49`), nodes keyed by
  `LocalBlockHash` edges.
- `RadixTree::find_matches(sequence, early_exit) -> OverlapScores`
  (`radix_tree.rs:200-202`) walks the tree matching the query's block-hash
  sequence edge-by-edge, tracking per-worker matched depth into
  `OverlapScores.scores: FxHashMap<WorkerWithDpRank, u32>`
  (`protocols.rs:978-980`).
- `KvIndexer` (`lib/kv-router/src/indexer/kv_indexer.rs:170`) drives this via
  `find_matches`/`find_matches_for_request`, feeding `KvScheduler` which
  picks the worker with the best overlap (more cached prefix = less
  prefill recompute if routed there).

This is purely **routing metadata** — hashes of token IDs used to find
cache-hit opportunities. It does not itself touch physical GPU memory.

### The same hash is also the physical KV cache storage key (when using KVBM)

When Dynamo's own KV Block Manager (KVBM) manages the cache, the identical
hash is reused as the literal lookup key for the physical stored block —
not just router metadata:

```rust
// lib/llm/src/block_manager/block/registry.rs:91
blocks: Arc<Mutex<HashMap<SequenceHash, Weak<BlockHandle>>>>
```
```rust
// lib/llm/src/block_manager/pool/managed/inactive.rs
lookup_map.insert(sequence_hash, block)
match_sequence_hash(sequence_hash)   // -> physical Block<S,L,M> (GPU/CPU handle)
```
```rust
// lib/kvbm-logical/src/blocks/pin.rs:17
// block identity = (manager_id, block_id, sequence_hash)
```

Both the router's `LocalBlockHash`/`SequenceHash` and kvbm's `SequenceHash`
(`PositionalLineageHash`) are built from the same underlying chain
(`dynamo_tokens::compute_hash_v2`/`compute_next_sequence_hash`,
`XXH3_SEED = 1337` — `lib/kv-router/src/protocols.rs:23,26-27`,
`lib/tokens/src/lib.rs:52,77`). So it's genuinely the same value doing
double duty, not two independently-computed hashes that happen to agree.

```
tokens -> block-hash chain (token IDs, chained with prefix) -> SequenceHash
              │                                                    │
              v                                                    v
     RadixTree: which worker           HashMap<SequenceHash, BlockHandle>:
     has this prefix cached?           does *this* worker actually have the
     (routing decision)                physical K/V tensors, and where?
                                        (cache-hit lookup, skips recompute)
```

**Caveat**: this direct hash-as-storage-key behavior applies when KVBM is
the active cache manager. If a backend uses its own native engine cache
(e.g. vanilla vLLM prefix caching without KVBM), the router hash still
drives routing, but the physical lookup uses that engine's own scheme. When
KVBM *is* the vLLM connector, Dynamo recomputes its own hash rather than
reusing vLLM's native block hash
(`lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py:93,109`),
so it's Dynamo's hash lineage end-to-end in that case.
