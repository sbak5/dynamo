# Dynamo KV-Router Reproducer — Build & Experiment Guide

Slurm-based reproducer for disaggregated (prefill/decode) Dynamo serving with
the KV-aware router, used to build/test code changes (e.g. the routing-decision
logging work in `lib/kv-router/src/scheduling/selector.rs`) against real
traffic without needing a full cluster deployment.

All scripts referenced below live at the repo root (`~/scratch/dynamo/`).

## TL;DR

```bash
# 1. Build a wheel with your current code changes
sbatch --nodelist=<idle ipp2-* x86_64 H100 node> --gres=gpu:1 dynamo_repro_build_wheel.slurm

# 2. Pick a shared RUN_ID and submit both jobs (order doesn't matter, but see
#    "Coordinating job A and B" below for avoiding wasted walltime)
export RUN_ID=$(date +%Y%m%d_%H%M%S)
sbatch --nodelist=<idle 8-GPU node> --gpus-per-node=8 \
  --export=ALL,RUN_ID,NUM_GPUS_PER_NODE=8 dynamo_repro_workers.slurm
sbatch --nodelist=<idle ipp2-* node> --gres=gpu:1 \
  --export=ALL,RUN_ID dynamo_repro_infra_client.slurm

# 3. Logs land in $HOME/dynamo_repro/runs/$RUN_ID/logs/
```

---

## Directory layout

Two separate storage roots, for a reason (see "Why `$HOME`" below):

```
$HOME/dynamo_repro/                          # NFS-mounted on every partition
  wheelhouse-<name>/                         # built wheel FILES only (small)
  runs/<RUN_ID>/
    shared/infra_ip.txt                      # job A -> job B handoff
    logs/                                    # everything below
    agentic-dataset/                         # only for the agentic client variant

/home/scratch.sbak_coreai/dynamo_repro/      # only "ipp"-tagged nodes have this
  build-cache-<arch>/                        # persistent Cargo registry+target
  runs/<RUN_ID>/wheelhouse/                  # build job's own copy
  wheelhouse-latest -> runs/<RUN_ID>/wheelhouse
  archived-runs/                             # old $HOME runs, moved here at 80% quota
```

**Why `$HOME` for run artifacts, not scratch:** scratch
(`/home/scratch.sbak_coreai`) is only mounted on partitions whose nodes are
tagged `ipp*` in Slurm Features. `$HOME` is NFS-mounted everywhere, so using
it for wheels/logs/handoff lets job A and job B run on *any* partition,
including GPU-heavy 8-GPU pools that don't have scratch. The wheel **build**
job still needs scratch, for the multi-GB persistent Cargo cache (would blow
the 5GB `$HOME` quota) — so it must run on an `ipp*` node, but nothing
downstream does.

`$HOME` has only a 5GB quota. `dynamo_repro_infra_client*.slurm` auto-archives
all but the newest run directory to
`/home/scratch.sbak_coreai/dynamo_repro/archived-runs/` once `$HOME` crosses
80% used. Archived runs are then only browsable from scratch-having nodes.

---

## Cluster notes (read this before submitting anything)

**Two-job pattern.** Everything is split into job A (etcd + NATS + 2 frontend
replicas + client) and job B (decode/prefill workers), submitted separately.
This cluster's `topology.conf` rejects multi-node Slurm allocations outright
on the partitions we use, so a single multi-node job doesn't work — each role
gets its own single-node job, coordinating via `$HOME/dynamo_repro/runs/<RUN_ID>/shared/infra_ip.txt`.

**Check node availability before submitting — always.**
```bash
scontrol show node <name> | grep -E "State=|AllocTRES|CfgTRES"
```
`AllocTRES` empty (or `gres/gpu` less than `CfgTRES`'s) means free GPUs.
`MIXED`/`ALLOCATED`/`RESERVED` mean busy. This cluster runs a large shared CI
+ research workload — 8-GPU nodes are frequently all busy simultaneously;
check `squeue -w <node1>,<node2>,...` for `TIME_LEFT` on the jobs holding them
if you want to estimate a wait.

**Priority races.** A node can look free in `scontrol show node` and get
grabbed by someone else's job in the seconds before your `sbatch` lands.
`sbatch` will just queue in that case — check `sacct -j <jobid>` after
submitting and resubmit against a different node if needed.

**Known node families:**
| Family | Status |
|---|---|
| `ipp2-*` (single-GPU H100, x86_64) | Reliable. Used for wheel builds, job A, quick standalone tests. |
| `a4u8g-0145` (L40Sx8) | Reliable, our most-used 8-GPU node. |
| `viking-prod-*` (`dgxh100` partition, 8x H100) | Reliable but often fully busy. |
| `4u8g-gen-*` (`H100x8` partition) | Reliable but often busy/reserved. |
| `nvdl-a112-luna*`, `luna-prod-*`, `PDX*` (A100 "luna" family) | **Avoid.** Crashes instantly with `RaisedSignal:53 (Real-time signal 19)` before writing any output, on the `general` partition. Confirmed on multiple distinct nodes in this family. |
| Orin edge devices (e.g. `a1u2n2g-*`) | **Avoid.** Broken NVIDIA driver stack for this container image (`nvidia_uvm not loaded`). These can get matched by an overly broad `--constraint`, e.g. `ipp2-1` alone also matches some Orin nodes — always add `&H100` or `&x86_64` to constraints, or use `--nodelist` with a specific known-good node. |

Also seen once on a *known-good* node family (`4u8g-gen-0082`): the same
`RaisedSignal:53` crash, confirming it's a sporadic cluster-wide
container-launch glitch, not exclusively tied to the bad node families above.
If a job dies instantly with no output file and exit code `0:53`, just retry
on a different node — it's not your script.

**Login-node overload.** This is a heavily shared login node (can be 400+
concurrent users, load average 100+). `sinfo`, `git push`, and similar
commands can transiently fail with `pthread_create error: Resource
temporarily unavailable` or `fork/exec ...: resource temporarily
unavailable`. Just retry — these are not real errors.

---

## 1. Build the wheel — `dynamo_repro_build_wheel.slurm`

Builds `ai-dynamo` (pure Python) + `ai-dynamo-runtime` (Rust/PyO3, via
`maturin`) from `REPO_DIR` inside the same NGC image used at runtime, so the
compiled `.so`'s ABI matches. Incremental: Cargo's registry + target dir
persist across submissions under
`/home/scratch.sbak_coreai/dynamo_repro/build-cache-<arch>/` (keyed by
`uname -m` — aarch64 and x86_64 builds don't share a cache). Only
`dynamo`'s own crates recompile when iterating on code changes; the
third-party dependency tree compiles once.

**Must run on an `ipp*`-tagged node** (needs scratch for the build cache).
Match the target architecture: `x86_64` for most GPU partitions
(`dgxh100`, `H100x8`, `general`), `aarch64` only for `h100x4_aarch64`.

```bash
export RUN_ID=$(date +%Y%m%d_%H%M%S)
sbatch --partition=general --nodelist=ipp2-1589 --gres=gpu:1 \
  --export=ALL,RUN_ID dynamo_repro_build_wheel.slurm
```

Useful overrides (pass via `--export=ALL,VAR=value,...`):
- `REPO_DIR` — build from a different checkout, e.g. a `git worktree` at a
  specific commit, for A/B comparisons against a baseline. Default: this repo.
- `WHEELHOUSE_NAME` — lets two builds coexist under
  `$HOME/dynamo_repro/wheelhouse-<name>/` (e.g. `baseline` vs `latest`) so
  you can run experiments against both without rebuilding between them.
  Default: `latest`.

Wheel files are copied to `$HOME/dynamo_repro/wheelhouse-<WHEELHOUSE_NAME>/`
at the end — that's what job A / job B actually install from.

---

## 2. Run job A (infra + frontends + client)

Two interchangeable variants — same etcd/NATS/2-frontend setup, different
client workload:

### 2a. Flat synthetic multi-turn — `dynamo_repro_infra_client.slurm`

Uses `aiperf profile` with a synthetic multi-turn conversation generator: 10
sessions, ~5 turns each, shared 200-token system prompt, sticky-session
routing. Simple, fast, good for quick sanity checks of routing behavior.

```bash
sbatch --partition=general --nodelist=ipp2-1589 --gres=gpu:1 \
  --export=ALL,RUN_ID dynamo_repro_infra_client.slurm
```

### 2b. Agentic-code multi-turn — `dynamo_repro_infra_client_agentic.slurm`

Uses aiperf's **Agentic Code Dataset Generator** (`aiperf synthesize
agentic-code` → `aiperf profile --custom-dataset-type mooncake_trace`)
instead — models layered shared-prompt reuse (global tools/system prompt,
group-shared repo instructions, session-specific starting context,
incrementally-growing conversation history), large repo-context prompts,
and probabilistic session resets/retires — much closer to real coding-agent
traffic than the flat synthetic config.

**Important: uses a custom config, not aiperf's bundled `default`.** The
bundled default config is tuned for ~167K-context models
(`max_prompt_tokens=167000`, `cache.layer1_tokens=32000` alone). Overriding
just `--max-isl` (tested 4000–16000) doesn't help — `layer1_tokens=32000` is
fixed inside the config and always exceeds any of those caps, so every
session is born over-budget and force-retired on turn 1, 100% of the time,
no matter what `--max-isl` you pick. `dynamo_repro_agentic_config.json`
(copied to `$HOME/dynamo_repro/agentic-config.json` so this script stays
scratch-independent) shrinks `layer1_tokens` to 500 and `layer2` mean to
3000, with `max_prompt_tokens=12000` — sized for this reproducer's
32768-context model with real headroom for turn growth. Verified: **62
total turns across 10 sessions** (avg 6.2/session, range 2–9) instead of 10
turns total (1 per session, 100% immediate forced-retire) with the bundled
default.

```bash
sbatch --partition=general --nodelist=ipp2-1589 --gres=gpu:1 \
  --export=ALL,RUN_ID dynamo_repro_infra_client_agentic.slurm
```

Overrides: `NUM_SESSIONS` (default 10), `AGENTIC_CONFIG` (default
`$HOME/dynamo_repro/agentic-config.json` — copy
`dynamo_repro_agentic_config.json` there first, or point this at your own).

Both variants also:
- Install the wheel from `$HOME/dynamo_repro/wheelhouse-<WHEELHOUSE_NAME>/`
  (same `WHEELHOUSE_NAME` override as the build job — **must match job B's**).
- Accept `DYN_LOG` (default `info` — the routing-decision logging in
  `selector.rs` runs at INFO level unconditionally, no need for `debug`
  anymore) and `DYN_REQUEST_TRACE` (default `0` — off, since it does its own
  per-request file I/O that would confound CPU-time comparisons; set to `1`
  to also capture Stack D's per-request JSONL trace).
- Print CPU-time instrumentation for each frontend process
  (`/proc/<pid>/stat` snapshotted before/after the client run) — useful for
  measuring logging/code-change overhead in isolation from GPU-inference-
  dominated end-to-end latency. See `[repro-A] ... frontend-N CPU time
  during client run: X.XXXs` in the job's stdout.
- Explicitly kill etcd/NATS/frontends and exit once the client finishes,
  rather than hanging until the 1-hour walltime (those are long-lived
  servers that never exit on their own — a bare `wait` would just block
  until Slurm kills the job).

---

## 3. Run job B (workers) — `dynamo_repro_workers.slurm`

Launches N decode + N prefill vLLM workers (disaggregated, NIXL KV
transfer), one single-GPU process per worker — data-parallel, not tensor
parallel, which is what actually gives the KV router multiple distinct
`worker_id`s to choose between.

```bash
sbatch --partition=dgxh100 --nodelist=viking-prod-238 --gpus-per-node=8 \
  --export=ALL,RUN_ID,NUM_GPUS_PER_NODE=8,WHEELHOUSE_NAME=latest dynamo_repro_workers.slurm
```

- `NUM_GPUS_PER_NODE` (default 4) splits in half between decode and prefill
  — 8 GPUs → 4 decode + 4 prefill.
- `RUN_ID` is **required** (no default) — must match job A's.
- `WHEELHOUSE_NAME` must match job A's, so frontend and workers run the same
  build.
- Pre-downloads the model once, serially, to container-local `/tmp` before
  starting any worker (avoids `huggingface_hub` lock-contention errors from
  concurrent downloads — also avoids NFS lock issues by not using
  scratch/`$HOME` for `HF_HOME`).
- Sets `UCX_NET_DEVICES=all UCX_IB_GPU_DIRECT_RDMA=no` and
  `NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo` — see the script's inline
  comments if you hit GPUDirect RDMA (`ibv_reg_dmabuf_mr`) or NIC-name
  mismatch errors on a new node type.

---

## Coordinating job A and job B

Both need the **same `RUN_ID`**. Order doesn't strictly matter (each polls
for the other), but if the 8-GPU pool is busy, prefer:

1. Submit job B first (it'll queue).
2. Wait for job B to actually reach `RUNNING` (`squeue -j <jobid>`) before
   submitting job A — job A's own "wait for a worker to register the model"
   loop has a 1-hour walltime; if job B is still queued when job A's walltime
   runs out, job A dies and needs resubmitting for nothing.

```bash
until squeue -j <job_B_id> -h -o "%T" | grep -q RUNNING; do sleep 20; done
sbatch ... dynamo_repro_infra_client.slurm   # now safe to submit job A
```

---

## Reading the logs

All under `$HOME/dynamo_repro/runs/$RUN_ID/logs/`:

| File | Contents |
|---|---|
| `frontend-{0,1}.log` | Frontend + router logs. `Routing formula` (once, at selector startup) + `Routing decision` (once per request) — see below. |
| `client-frontend{0,1}.log` | `aiperf` client output — throughput/latency tables, request counts. |
| `decode-*.log`, `prefill-*.log` | Per-worker vLLM logs. |
| `etcd.log`, `nats.log` | Infra logs. |
| `synthesize-frontend{0,1}.log` | (agentic variant only) dataset-generation stats — `Session Endings` (forced retires / probabilistic resets / restart splits) tells you whether sessions actually grew multi-turn. |
| `request-trace-{0,1}.*.jsonl(.gz)` | (only if `DYN_REQUEST_TRACE=1`) Stack D's per-request block-hash trace. |

**`Routing decision` line format** (standard tracing `key=value`, space-
separated — matches the rest of the codebase's logfmt convention):
```
Routing decision request_id="..." router_mode="kv" worker_type=prefill isl_tokens=1213
num_candidates=4 chosen_worker_id=... chosen_dp_rank=0 chosen_logit=75.8125
margin=0.0 tie_count=4 raw_prefill_blocks=... overlap_credit_blocks=... decode_cost_blocks=...
shared_blocks_beyond=0 device_overlap_blocks=... active_prefill_tokens=... active_decode_blocks=...
effective_cached_blocks=... host_pinned_blocks=... disk_blocks=... total_kv_blocks=Some(...)
candidates=<short_worker_id>:<dp_rank>:<logit>(overlap=..,prefill_tok=..,decode_blk=..),...
```
`tie_count > 1` means the winner was picked by the router's reservoir-
sampling tie-break, not a clean win — pair with `margin=0.0` to spot exact
ties. `candidates=` lists every eligible candidate scored for that decision,
not just the winner. The one-time `Routing formula` line (per selector, at
startup) gives the fixed cost-function structure + weights those numbers
were computed from.

---

## Local CI checks before opening a PR — `dynamo_repro_test_ci.slurm`

Runs the same checks `pre-merge.yml` runs in CI, scoped to `dynamo-kv-router`
(the crate `selector.rs` touches) plus a workspace-wide `cargo fmt --check`:

```bash
sbatch --nodelist=ipp2-1589 dynamo_repro_test_ci.slurm
```

Runs, in order: `cargo fmt -- --check`, `cargo clippy -p dynamo-kv-router
--no-deps --all-targets -- -D warnings`, `cargo test -p dynamo-kv-router
--locked --all-targets`. Uses the same persistent build-cache as the wheel
build job, so it's fast after the first run.

Also run `pre-commit` locally (installs the same hooks CI's `pre-commit` job
uses):
```bash
python3 -m pip install --user --break-system-packages pre-commit
export PATH="$HOME/scratch/.local/bin:$PATH"   # wherever pip --user put it
pre-commit install                              # registers git hooks for future commits
pre-commit run --files <changed files>           # or --all-files (slow, whole repo)
```

---

## Quick troubleshooting reference

| Symptom | Cause / fix |
|---|---|
| Job dies instantly, no output file, exit `0:53` | Sporadic cluster container-launch glitch (`RaisedSignal:53`). Retry on a different node. |
| Job A hangs to its 1-hour walltime | Job B never started (still queued) — job A's model-registration wait loop timed out. Wait for job B to reach `RUNNING`, then resubmit job A. (Already fixed for the "client finished but job hangs anyway" case — job A now explicitly kills etcd/NATS/frontends and exits once the client is done.) |
| `pip install` / wheel mismatch errors | `WHEELHOUSE_NAME` mismatch between job A and job B, or wrong architecture wheel (aarch64 wheel on an x86_64 node or vice versa). |
| `sinfo`/`git push` fail with "resource temporarily unavailable" | Login node overload. Just retry. |
| Agentic client: 100% forced-retire, sessions never grow | Using aiperf's bundled `default` config instead of `dynamo_repro_agentic_config.json`. Pass `--config` explicitly (already the default in the agentic script — check `AGENTIC_CONFIG` wasn't overridden to something else). |
| `HF_HOME` "Lock acquisition failed" | `HF_HOME` pointed at NFS (scratch/`$HOME`) instead of container-local `/tmp`, or workers downloading concurrently instead of the serial pre-download step. |
