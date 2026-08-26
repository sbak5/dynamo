# Lifecycle Runtime Interactive Development

Use this guide to resume lifecycle-tracing work in a fresh session.

## Session contract

- Read `/home/scratch.sbak_coreai/dynamo-observe/research/model-based-diagnosis/PROPOSAL-final-candidate.md` first.
- Then read `~/scratch/dynamo/launch/kv-router-routing-analysis/RUNBOOK.md`.
- Use `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1` as the build and mock baseline.
- Preserve `~/scratch/dynamo` as the build checkout. Do not modify the real prefill/decode workload launchers.
- Use the dedicated frontend-only mock launcher in this worktree:
  `launch/kv-router-routing-analysis/launch/lifecycle-runtime-mock.slurm`.

## Isolate implementation work

On the remote machine, create or resume a separate worktree and use `git switch` there:

```bash
ssh dlc
cd ~/scratch/dynamo
git worktree add --detach ~/scratch/dynamo-lifecycle-dev v1.4.1
cd ~/scratch/dynamo-lifecycle-dev
git switch -c sbak/lifecycle-runtime-instrumentation
```

If the worktree already exists, inspect it and switch to the existing implementation
branch instead of recreating it. Keep the build checkout free of implementation edits.

## Build an override wheelhouse

The builder launcher is maintained in the build checkout. Point it at the isolated
worktree so the built wheels include the implementation branch:

```bash
BUILD_REPO=~/scratch/dynamo
DEV_REPO=~/scratch/dynamo-lifecycle-dev
WHEELHOUSE_NAME=lifecycle-runtime-v141-$(date +%Y%m%d-%H%M%S)

REPO_DIR="$DEV_REPO" \
IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1 \
WHEELHOUSE_NAME="$WHEELHOUSE_NAME" \
sbatch --nodelist=gh-nvl-203-compute02 \
  "$BUILD_REPO/launch/kv-router-routing-analysis/launch/build-qwen36-aarch64-wheel.slurm"
```

Wait for a successful Slurm status and confirm that
`~/dynamo_repro/wheelhouse-$WHEELHOUSE_NAME/` contains both the
`ai_dynamo` and `ai_dynamo_runtime` wheels.

## Run the iterative mock

The mock runs the frontend and one aggregated `dynamo.mocker` worker only. It does
not start prefill or decode roles and must not replace the real-workload scripts.

```bash
DEV_REPO=~/scratch/dynamo-lifecycle-dev
RUN_ID=lifecycle_mock_v141_$(date +%Y%m%d_%H%M%S)

cd "$DEV_REPO"
RUN_ID="$RUN_ID" \
IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1 \
WHEELHOUSE_NAME="$WHEELHOUSE_NAME" \
DYN_LIFECYCLE_TRACE_ENABLED=true \
sbatch --output="$HOME/dynamo_repro/${RUN_ID}.slurm.out" \
  --nodelist=gh-nvl-203-compute02 \
  launch/kv-router-routing-analysis/launch/lifecycle-runtime-mock.slurm
```

The launcher force-reinstalls the override wheels: matching version numbers alone
must not cause pip to retain the base-image packages.

## Logs and validation

The mock writes process logs in container-local temporary storage. Its cleanup trap
stages them back to the run-specific host directory, on success or failure:

```text
~/dynamo_repro/lifecycle_mock_runs/$RUN_ID/logs/
  etcd.log
  nats.log
  frontend.log
  mocker.log
  response.jsonl
```

Validate the job and response after it exits:

```bash
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed -X
sed -n '1,12p' "$HOME/dynamo_repro/lifecycle_mock_runs/$RUN_ID/logs/response.jsonl"
grep -Ei 'panic|error|failed|traceback' \
  "$HOME/dynamo_repro/lifecycle_mock_runs/$RUN_ID/logs/"{frontend,mocker}.log || true
```

A successful mock has exit code `0`, streaming `data:` records, and `[DONE]`.
ModelExpress fallback, text-only MM-routing fallback, and teardown lease/unregister
warnings may occur in this small mock and do not invalidate a completed response.

## Scope checkpoint

This mock validates the rebuilt runtime package and a frontend request path. It does
not prove lifecycle-span emission until a real pipeline boundary starts the registered
lifecycle spans. Keep timing-only spans free of detailed metrics until that next
milestone is explicitly implemented.

Before ending a session, run `git status --short --branch`, commit intentional
changes with DCO sign-off, and record the build/mock Slurm job IDs in the handoff.
