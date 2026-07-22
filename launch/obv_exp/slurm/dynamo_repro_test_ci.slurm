#!/bin/bash
#SBATCH --job-name=dynamo-repro-test-ci
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#
# Scratch/throwaway script: run the same checks pre-merge.yml's rust-tests
# and rust-clippy jobs run, scoped to dynamo-kv-router (the only crate our
# selector.rs change touches) plus a workspace-wide `cargo fmt --check`
# (cheap, catches formatting regardless of scope). Uses the same persistent
# incremental build-cache as dynamo_repro_build_wheel.slurm. Not part of the
# reproducer pipeline; safe to delete.

set -euo pipefail
IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1}"
export REPO_DIR="${REPO_DIR:-/home/scratch.sbak_coreai/dynamo}"
BASE_DIR="/home/scratch.sbak_coreai/dynamo_repro"
export BUILD_CACHE="${BASE_DIR}/build-cache-$(uname -m)"
export RUSTUP_HOME="${BUILD_CACHE}/rustup"
export CARGO_HOME="${BUILD_CACHE}/cargo-home"
export CARGO_TARGET_DIR="${BUILD_CACHE}/cargo-target"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$CARGO_TARGET_DIR"

srun --container-image="$IMAGE" \
     --container-mounts="/home/scratch.sbak_coreai:/home/scratch.sbak_coreai" \
     --container-remap-root \
     --export=ALL,REPO_DIR,RUSTUP_HOME,CARGO_HOME,CARGO_TARGET_DIR \
     bash -s <<'INNER'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake curl unzip

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain 1.93.1 --profile minimal
source "${CARGO_HOME}/env"
rustup component add clippy rustfmt

cd "$REPO_DIR"

echo "=========================================="
echo "=== cargo fmt -- --check (workspace) ==="
echo "=========================================="
cargo fmt -- --check
echo "cargo fmt: PASS (no diff)"

echo "=========================================="
echo "=== cargo clippy -p dynamo-kv-router (matches CI: --no-deps --all-targets -- -D warnings) ==="
echo "=========================================="
cargo clippy -p dynamo-kv-router --no-deps --all-targets -- -D warnings

echo "=========================================="
echo "=== cargo test -p dynamo-kv-router --locked --all-targets ==="
echo "=========================================="
cargo test -p dynamo-kv-router --locked --all-targets

echo "=========================================="
echo "ALL CHECKS PASSED"
echo "=========================================="
INNER
