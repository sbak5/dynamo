# Dynamo KV Router: Cross-Frontend Routing Collisions — Session Summary

Follow-on to `dynamo-kv-router-disagg-notes.md`. That doc covers the call
path, disaggregated prefill/decode mechanics, and KV block hashing. This doc
covers a deeper investigation into two real repro runs' routing-decision
logs, a bug found and fixed in the analysis tooling, and a design
brainstorm around cross-frontend routing collisions. No dynamo source code
was changed in this session — only the standalone analysis scripts, at
`kv-router-routing-analysis/` alongside this file.

## 1. Log-based routing analysis tooling

Built three scripts (`kv-router-routing-analysis/plot_routing.py`,
`plot_stacked_logit.py`, `plot_cross_frontend.py`) that parse
`dynamo_kv_router::scheduling::selector` "Routing decision" log lines from
`logs/frontend-*.log` in a run directory (`RUN_DIR` env var) and plot:

- chosen routing logit over time, prefill vs. decode
- per-candidate stacked logit component breakdown (base cost / queued load
  from other in-flight requests / KV cache-hit credit), with the credit
  correctly rendered as a **subtraction** off the top (hatched region) once
  the real formula was verified against log data — an earlier version
  incorrectly added it as a positive stacked segment based on a wrong
  assumption about a config weight (see §3)
- cross-frontend collisions: cases where two independent frontend routers
  pick the same worker within a short time window

See the README in that directory for usage and the reverse-engineered
logit formula.

## 2. Logit formula (verified against log data)

```
logit = prefill_load_scale * max(raw_prefill_blocks - overlap_credit_blocks, 0) + decode_cost_blocks
```

- **`worker_type=prefill`**: `raw_prefill_blocks` = (this request's
  `isl_tokens` + that candidate's already-queued prefill tokens from other
  in-flight requests) / `block_size` (16). `overlap_credit_blocks` = that
  candidate's KV cache-hit blocks (`device_overlap_blocks`), confirmed to
  subtract from the **combined** (new + queued) total before the
  `max(...,0)` floor — verified against a log line with simultaneous
  nonzero overlap and nonzero queued tokens.
- **`worker_type=decode`**: `raw_prefill_blocks` is always 0 in every
  decode-type routing decision observed — no cache-affinity term applies.
  `logit` = this request's own estimated decode cost (constant per
  decision) + that candidate's already-queued decode blocks. Decode routing
  explicitly zeroes `overlap_score_credit` in source
  (`prefill_router/mod.rs:404`, `build_decode_router_override`) — confirmed
  deliberate, not an oversight, and repeated in multiple places
  (`lib/kv-router/src/scheduling/config.rs`, `lib/bindings/c/src/lib.rs`,
  even the mocker/replay tooling).

**Router does not simply maximize cache overlap** — it minimizes total
logit. Found 2 concrete cases (of 116 prefill decisions in one run) where
the chosen worker did *not* have the highest overlap among candidates,
because a competing worker's queued load pushed its total logit higher
despite a much bigger cache hit (75 vs. 13 overlap blocks).

## 3. Bug found and fixed: chosen-worker matching

Candidate worker IDs in the log are truncated (e.g. `346526`) while
`chosen_worker_id` is logged in full (e.g. `7587896060170346526`). An
initial version of the "chosen worker" highlighting used exact string
equality, which never matched — silently rendering no chosen-worker
markers at all across every plot. Fixed via suffix match
(`chosen_worker_id.endswith(candidate_worker_id)`).

## 4. Two signal sources behind the logit — different consistency guarantees

Traced where each input to the logit formula actually comes from:

- **Cache-affinity signal** (`device_overlap_blocks`, RadixTree state):
  sourced directly from **workers**, which broadcast their own
  `KvCacheEvent` (Stored/Removed/Cleared) on a shared NATS/ZMQ subject
  (`kv-events`). Every frontend subscribes to this *same worker-originated
  stream independently* — no frontend-to-frontend relay involved. This
  signal is naturally consistent across frontends, subject only to
  worker→frontend propagation lag.
- **Queued-load signal** (`queued_prefill`/`queued_decode`): genuinely
  frontend-local. Each frontend publishes its own dispatch actions
  (`ActiveSequenceEvent`: AddRequest/Free/MarkPrefillCompleted) to peer
  frontends via `lib/kv-router/src/sequences/replica_sync.rs`, over NATS,
  immediately per-event (not batched/interval-based; receiver-side batching
  is capped at 1ms / 256 events — negligible). This is documented as
  **"eventually consistent"** in `lib/kv-router/src/sequences/README.md`,
  with lag explicitly called out as an accepted design tradeoff, not a
  tunable SLA. No config knob controls this propagation speed — the two
  candidates that looked like they might (`DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS`,
  `queue_policy`/`quantum`/`prefill_busy_threshold_frac`) are both
  single-router-local settings, unrelated to cross-frontend sync.

## 5. Cross-frontend collision measurements (two runs)

Both frontends confirmed to share the exact same 8-worker pool (identical
`worker_id` sets in each frontend's startup "Adding worker" logs).

| Run | prefill decisions | prefill collisions (<0.3s) | decode decisions | decode collisions (<0.3s) |
|---|---|---|---|---|
| `20260708_213527` (random-length benchmark) | 116 | 34 (29%) | 116 | 44 (38%) |
| `20260709_230143` (agentic multi-turn) | 126 | 23 (18%) | 126 | 19 (15%) |

**Margin analysis** (how close the runner-up candidate's logit was to the
chosen one, at the moment of collision) split collisions into two very
different categories:

- **~40-45% "true ties"** (margin ≤ 5 blocks): the logit signal genuinely
  couldn't distinguish candidates — picking deterministically-lowest is
  arbitrary here. Randomized tie-breaking within a margin band would fix
  these at no cost, since alternatives are truly near-equivalent.
- **~55-60% "correctly convergent"** (large margin, up to 365-453 blocks):
  both frontends *independently and correctly* identified the same worker
  as meaningfully better (most likely real cache-hit affinity, which we
  confirmed is globally-consistent signal, not noise). Forcing
  randomization here would be actively harmful — it throws away real
  savings to dodge a collision that isn't actually about the ranking being
  wrong, but about neither frontend's *load* view accounting for the
  other's simultaneous pick.

**Fundamental point**: even with perfectly synchronized, zero-lag shared
state, N independent dispatchers computing the same deterministic argmin
over the same worker pool will still collide, because they all resolve to
the identical minimum simultaneously. Synchronization only shrinks the race
window; it doesn't remove the structural cause. This is the classic
"thundering herd on shortest-queue" problem.

## 6. Design brainstorm (not implemented)

Two regimes considered:

**A. Frontends ≤ workers (normal case)** — static local partition per
frontend (e.g. `worker_index % num_frontends == frontend_index`, derived
deterministically from existing etcd-backed discovery state, no new
coordination needed) + overflow to the shared pool only when local is
saturated. Proposed folding overflow into the existing logit formula as one
more weighted term, consistent with the codebase's existing pattern
(`prefill_load_scale`, `host_cache_hit_weight`, etc.):

```
effective_logit(candidate) = logit(candidate) + (0 if local else remote_partition_penalty_blocks)
```

This evaluates smoothly (no hard threshold/hysteresis to mis-tune), and
`remote_partition_penalty_blocks=0` degenerates back to today's
fully-shared-pool behavior — a natural rollback lever.

Refinements discussed:
- Worth partitioning **prefill only** — decode has no cache-affinity
  benefit (confirmed §2), so partitioning it trades away flexibility to
  solve a pure load-balance problem that power-of-two-choices (P2C) already
  solves well without exclusivity constraints.
- Partitioning prefill likely has a secondary benefit beyond collision
  avoidance: fewer distinct workers per frontend → higher repeat cache-hit
  rate within the partition (not yet measured).
- Combine with **margin-based randomized tie-breaking** (§5) for the
  small-margin collision subset — orthogonal improvement, doesn't conflict
  with partitioning.

**B. Frontends > workers (degenerate case)** — exclusive partitioning loses
its slack. Proposed two content/load-appropriate mechanisms instead of
extending partitioning:
- **Prefill**: rendezvous/consistent hashing keyed on the request's own
  `SequenceHash`, so independent frontends converge on the same preferred
  worker for the same content with zero coordination. **Caveat found**:
  pure single-target hashing hotspots on ties/low-signal cases (the exact
  cold-start scenario observed in the data, where all 4 candidates were
  identical) — trades probabilistic collision for guaranteed funneling,
  which is worse. Not adopted as-is; margin-randomization (§5) is the
  better-fitting tool for that specific case.
- **Decode**: power-of-two-choices (random small-subset sampling + pick
  best of subset) instead of full argmin — well-suited since decode routing
  is pure load-balancing with no affinity to preserve, and P2C is
  established to bound max load well under many independent, uncoordinated
  dispatchers.

## 7. Dead end investigated: selective/diffed KV transfer

Checked whether prefill→decode KV handoff does any "diff against what the
decode worker already has" to avoid redundant transfer (a plausible
optimization if decode workers sometimes already hold relevant blocks).
Confirmed **no such mechanism exists anywhere in the codebase** — searched
for config flags, env vars, and vendored backend-connector capabilities
(vLLM NIXL/Mooncake, SGLang prefill/decode handlers); all forward the full
computed block list unconditionally via an opaque `bootstrap_room` session
key. This is architecturally consistent with decode routing being
deliberately affinity-blind (§2) — there's rarely a reason to expect a
freshly-selected decode worker already holds relevant blocks, so a
diff/handshake wouldn't often pay off under the current design. Dynamo's
actual answer to "avoid redundant recompute/transfer" is a **shared
host/disk cache tier** (Flash Indexer, addressable by the same
`SequenceHash`), not per-transfer negotiation between two specific workers.

## 8. Dead end investigated: was the second run's heavy cache-hit pattern from cross-frontend sharing?

The `20260709_230143` run's agentic multi-turn dataset showed near-total
overlap-credit coverage in the stacked-logit plots. Initial check compared
the two frontends' synthetic dataset files
(`agentic-dataset/frontend-{0,1}/.../dataset.jsonl`, generated by NVIDIA's
external `aiperf` package, `aiperf.dataset.agentic_code_gen` — not vendored
in this repo) and found ~198 of ~201-219 `hash_id` values shared between
frontend-0 (seed1) and frontend-1 (seed2) — initially read as evidence of
heavy real cross-frontend content sharing, which would undercut the
partitioning argument in §6.

**Retracted after checking the generator source directly.** `hash_ids` are
assigned by `PrefixAllocator` purely by layer/position geometry (sequential
counters), independent of content. The actual token text behind a given
`hash_id` is sampled from a corpus RNG seeded off the top-level generation
seed (`rng.init(seed)`), advancing as each `hash_id` is first encountered
during that specific run. Since the two frontends used different seeds
(and near-certainly different hash_id-encounter orders), the same numeric
label maps to different actual token content in each dataset — the overlap
is a **bookkeeping artifact**, not real shared content. "Same config per
client" gives the same *statistical shape* of workload per client, not
shared content.

**Conclusion**: the partitioning argument in §6 stands for this dataset.
The heavy cache-hit-credit pattern observed is much more plausibly
explained by each session's own turns reusing their own prior turn's
cache — a purely single-frontend, single-session phenomenon unaffected by
partitioning.

**Left open**: whether the one fixed "L1" block (`hash_id=0`, ~500 tokens,
present in the initial turn of every session, described in config as a
global/shared tier) is generated identically across runs — a genuine
possible exception to the above conclusion. Not yet checked.

## Open threads for next session

- Verify the L1/`hash_id=0` block generation mechanism (§8).
- Decide whether to write this up as a DEP (a draft,
  `DEP-router-decision-logging-and-load-aware-routing-draft.md`, already
  exists in this checkout and may be relevant/overlapping).
- Prototype partition-assignment + overflow-penalty logit term against
  `lib/kv-router/src/scheduling/policy_config.rs` / `PrefillRouter`.
- Measure whether prefill-only partitioning actually raises intra-partition
  cache-hit rate as predicted in §6.
