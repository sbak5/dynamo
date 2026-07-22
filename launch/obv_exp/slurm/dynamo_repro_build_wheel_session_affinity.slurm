#!/bin/bash
#SBATCH --job-name=dynamo-repro-build-wheel-sa
#SBATCH --partition=general
#SBATCH --nodelist=ipp2-1635
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#
# Reproducer, wheel-build step: builds `ai-dynamo` (pure-Python, includes the
# vLLM backend source) and `ai-dynamo-runtime` (Rust/PyO3 bindings) wheels
# from the checked-out source at REPO_DIR, so DEP-proposed code changes
# (router logging, FPM wiring, etc.) can be installed into the stock NGC
# container on frontend/client and worker nodes before they start dynamo
# processes.
#
# MUST run on a partition with /home/scratch.sbak_coreai mounted (only
# "ipp"-tagged partitions have this — see dynamo_repro_infra_client.slurm
# for how that was discovered; gh200/luna40g/luna80g/gb200nvl4 do NOT).
#
# Builds INSIDE the same NGC image used at runtime (not a separate
# manylinux builder image, unlike the official release pipeline in
# container/templates/wheel_builder.Dockerfile) so the compiled .so's ABI
# (glibc/CUDA/driver) matches the containers the wheels get installed into.
# Also matches CPU architecture (aarch64) to the worker nodes' image.
#
# Recipe follows README.md "Building from Source" +
# docs/getting-started/building-from-source.md, mirroring what
# container/templates/wheel_builder.Dockerfile's runtime_wheel_builder stage
# does, minus the manylinux/auditwheel portability machinery (not needed
# since we install straight back into the same image family we built in).
#
# Rebuilds are incremental: Cargo's registry/download cache and compiled
# target dir persist across job submissions (and across RUN_IDs) under
# ~/scratch/dynamo_repro/build-cache/ (not container-local /tmp, which is
# wiped every job). Only dynamo's own crates (and anything depending on
# them) recompile when iterating on code changes; the third-party
# dependency tree only compiles once.
#
# Directory layout under ~/scratch/dynamo_repro/ (BASE_DIR below):
#   build-cache/            persistent, shared across every RUN_ID (Cargo
#                            registry/target -- multi-GB, must stay on scratch,
#                            would blow the 5GB $HOME quota)
#   runs/<RUN_ID>/wheelhouse/  this run's built wheels
#   runs/<RUN_ID>/logs/        this run's build log
#   wheelhouse-latest -> runs/<RUN_ID>/wheelhouse   (stable scratch-side path)
#
# The built wheel FILES (small -- tens of MB, not the build cache) are also
# copied to $HOME/dynamo_repro/wheelhouse-latest/ at the end of this script.
# dynamo_repro_infra_client.slurm / dynamo_repro_workers.slurm install from
# THAT copy, not the scratch one -- $HOME is NFS-mounted on every partition
# (scratch is only mounted on "ipp"-tagged nodes), so this is what lets
# frontend/client/worker jobs run on ANY partition (e.g. 8-GPU nodes that
# don't have scratch), not just scratch-having ones. This build job itself
# must still run on a scratch-having partition (needs the persistent
# build-cache).
#
# Usage:
#   export RUN_ID=$(date +%Y%m%d_%H%M%S)   # or let this script pick one
#   sbatch dynamo_repro_build_wheel.slurm
# Then dynamo_repro_infra_client.slurm / dynamo_repro_workers.slurm pick up
# $HOME/dynamo_repro/wheelhouse-latest/ automatically -- no path to pass.

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1}"
export REPO_DIR="${REPO_DIR:-/home/scratch.sbak_coreai/dynamo}"

BASE_DIR="/home/scratch.sbak_coreai/dynamo_repro"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"

export WHEELHOUSE="${RUN_DIR}/wheelhouse"
mkdir -p "$WHEELHOUSE"

LOGDIR="${RUN_DIR}/logs"
mkdir -p "$LOGDIR"

GIT_SHA=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=""
git -C "$REPO_DIR" diff --quiet 2>/dev/null || GIT_DIRTY="-dirty"

# Persistent, reused-across-builds cache on scratch (not container-local
# /tmp, which is wiped every job since each srun gets a fresh container).
# CARGO_TARGET_DIR is what makes rebuilds incremental: cargo/rustc skip
# recompiling any crate whose source+deps didn't change since the last
# build here, which on this workspace is the dominant cost (the third-party
# dependency tree alone took ~5.5 min from cold on the first build; only
# dynamo's own crates need to rebuild when iterating on DEP code changes).
# Keyed by uname -m: aarch64 (h100x4_aarch64) and x86_64 (most other GPU
# partitions) builds produce incompatible object files, so they must not
# share a CARGO_TARGET_DIR.
export BUILD_CACHE="${BASE_DIR}/build-cache-$(uname -m)"
export RUSTUP_HOME="${BUILD_CACHE}/rustup"
export CARGO_HOME="${BUILD_CACHE}/cargo-home"
export CARGO_TARGET_DIR="${BUILD_CACHE}/cargo-target"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$CARGO_TARGET_DIR"

# WHEELHOUSE_NAME lets two builds coexist under $HOME (e.g. "baseline" vs
# "latest") for A/B comparisons -- see dynamo_repro_infra_client.slurm /
# dynamo_repro_workers.slurm, which install from the same named path.
export WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-latest}"
export HOME_WHEELHOUSE="${HOME}/dynamo_repro/wheelhouse-${WHEELHOUSE_NAME}"

echo "[repro-build] node=${SLURMD_NODENAME:-unknown} run_id=${RUN_ID}"
echo "[repro-build] repo=${REPO_DIR} git_sha=${GIT_SHA}${GIT_DIRTY}"
echo "[repro-build] wheelhouse=${WHEELHOUSE}"
echo "[repro-build] will also copy wheels to ${HOME_WHEELHOUSE}"

# --container-remap-root: apt-get inside the NGC image needs root.
srun --container-image="$IMAGE" \
     --container-mounts="/home/scratch.sbak_coreai:/home/scratch.sbak_coreai,${HOME}:${HOME}" \
     --container-remap-root \
     --export=ALL,REPO_DIR,WHEELHOUSE,RUSTUP_HOME,CARGO_HOME,CARGO_TARGET_DIR,HOME_WHEELHOUSE \
     bash -s <<'INNER' 2>&1 | tee "${LOGDIR}/build.log"
set -euo pipefail

echo "[repro-build] installing system build deps..."
apt-get update -qq
apt-get install -y -qq \
  build-essential libhwloc-dev libudev-dev pkg-config libclang-dev \
  protobuf-compiler python3-dev cmake curl

echo "[repro-build] installing Rust 1.93.1 (pinned in rust-toolchain.toml)..."
# RUSTUP_HOME/CARGO_HOME/CARGO_TARGET_DIR already exported from the outer
# script, pointing at persistent scratch paths -- rustup-init is idempotent
# and skips reinstalling an already-present toolchain.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain 1.93.1 --profile minimal
# shellcheck disable=SC1091
source "${CARGO_HOME}/env"

echo "[repro-build] setting up Python 3.12 build venv..."
pip install --quiet --upgrade uv
uv venv /tmp/build-venv --python 3.12
# shellcheck disable=SC1091
source /tmp/build-venv/bin/activate
uv pip install --quiet --upgrade meson pybind11 patchelf 'maturin[patchelf]' tomlkit pip

cd "$REPO_DIR"

echo "[repro-build] building ai-dynamo (pure-Python) wheel..."
uv build --wheel --out-dir "$WHEELHOUSE"

echo "[repro-build] building ai-dynamo-runtime (Rust/PyO3) wheel..."
cd "${REPO_DIR}/lib/bindings/python"
maturin build --release \
  --features "kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass" \
  --out "$WHEELHOUSE"

echo "[repro-build] done. Wheels:"
ls -la "$WHEELHOUSE"

# Copy (not symlink -- $HOME must be self-contained so nodes without the
# scratch mount can still install) the wheel FILES into $HOME. This is the
# path dynamo_repro_infra_client.slurm / dynamo_repro_workers.slurm actually
# install from.
rm -rf "$HOME_WHEELHOUSE"
mkdir -p "$HOME_WHEELHOUSE"
cp "${WHEELHOUSE}"/*.whl "$HOME_WHEELHOUSE"/
echo "[repro-build] copied wheels to ${HOME_WHEELHOUSE}:"
ls -la "$HOME_WHEELHOUSE"
INNER

# Stable symlink on the scratch side too (build-cache debugging convenience).
ln -sfn "$WHEELHOUSE" "${BASE_DIR}/wheelhouse-latest"

echo "[repro-build] scratch wheelhouse: ${WHEELHOUSE}"
echo "[repro-build] scratch symlink:    ${BASE_DIR}/wheelhouse-latest"
echo "[repro-build] HOME wheelhouse (installed from by other jobs): ${HOME_WHEELHOUSE}"
