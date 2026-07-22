#!/bin/bash
#SBATCH --job-name=dynamo-repro-workers
#SBATCH --partition=h100x4_aarch64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#
# Reproducer, job B of 2: N decode workers + N prefill workers (vLLM backend,
# disaggregated, NIXL KV transfer), data-parallel — one single-GPU process per
# worker, splitting the node's GPUs in half between decode and prefill. This
# (not tensor parallelism) is what actually gives the KV router multiple
# distinct worker_ids to choose between, which is the point of this
# reproducer (observing routing decisions/contention across workers).
#
# Submitted as a SEPARATE job from dynamo_repro_infra_client.slurm (job A) —
# see that file for why (Slurm topology.conf on this partition rejects
# multi-node jobs outright).
#
# Learns where etcd/NATS/the frontends are by polling for the IP file job A
# writes to $SHARED. Must be submitted with the same RUN_ID as job A --
# everything for one run lives under $HOME/dynamo_repro/runs/<RUN_ID>/
# {shared,logs}/ (see dynamo_repro_infra_client.slurm for why $HOME rather
# than scratch -- it's what lets this job target GPU-heavy partitions that
# don't have the scratch mount, e.g. 8-GPU nodes).
#
# Usage (after job A is submitted and RUNNING):
#   export RUN_ID=<same-value-used-for-job-A>
#   sbatch dynamo_repro_workers.slurm

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
RUN_ID="${RUN_ID:?RUN_ID must be set to the same value used for dynamo_repro_infra_client.slurm}"

BASE_DIR="${HOME}/dynamo_repro"
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"
SHARED="${RUN_DIR}/shared"
export LOGDIR="${RUN_DIR}/logs"
mkdir -p "$LOGDIR"

# See dynamo_repro_infra_client.slurm for why this is exported explicitly
# rather than relied on as the container's own $HOME (the srun heredoc below
# is single-quoted, so it resolves via the container's environment unless
# passed through --export).
export HOST_HOME="$HOME"

# Matches dynamo_repro_build_wheel.slurm's WHEELHOUSE_NAME / job A's
# WHEELHOUSE_NAME -- must agree with job A so frontend and workers run the
# same dynamo build.
export WHEELHOUSE_NAME="${WHEELHOUSE_NAME:-latest}"

# HF_HOME is set to a container-LOCAL path (/tmp, inside the heredoc below),
# not scratch. huggingface_hub's download manager uses POSIX file locks on
# the blob cache, which NFS doesn't support properly -- concurrent workers
# sharing an NFS-mounted HF_HOME hit "Lock acquisition failed" errors. All 4
# workers below run inside this ONE container/srun invocation, so
# container-local /tmp is already shared between them; no bind-mount needed.

echo "[repro-B] waiting for ${SHARED}/infra_ip.txt from job A..."
until [ -s "${SHARED}/infra_ip.txt" ]; do sleep 2; done
export INFRA_IP=$(cat "${SHARED}/infra_ip.txt")
echo "[repro-B] node=${SLURMD_NODENAME} infra_ip=${INFRA_IP} run_id=${RUN_ID} num_gpus=${NUM_GPUS_PER_NODE}"
echo "[repro-B] logs in ${LOGDIR}, HF cache will be container-local /tmp"

export ETCD_ENDPOINTS="${INFRA_IP}:2379"
export NATS_SERVER="nats://${INFRA_IP}:4222"

# Inner script is piped via stdin (heredoc) rather than passed as a `bash -c`
# string, so none of it needs escaping against the outer script — all values
# from the outer script cross over as plain exported env vars instead.
srun --container-image="$IMAGE" \
     --container-mounts="${HOST_HOME}:${HOST_HOME}" \
     --export=ALL,ETCD_ENDPOINTS,NATS_SERVER,MODEL,NUM_GPUS_PER_NODE,INFRA_IP,LOGDIR,HOST_HOME,WHEELHOUSE_NAME \
     bash -s <<'INNER'
set -euo pipefail

# Install the custom-built wheel (dynamo_repro_build_wheel.slurm output),
# if present, before starting anything -- see dynamo_repro_infra_client.slurm
# for the matching step there (both must run the same dynamo build).
WHEELHOUSE="${HOST_HOME}/dynamo_repro/wheelhouse-${WHEELHOUSE_NAME}"
if [ -d "$WHEELHOUSE" ]; then
  echo "[repro-B] installing custom dynamo wheel from ${WHEELHOUSE}"
  pip install --no-deps "${WHEELHOUSE}"/ai_dynamo_runtime*.whl "${WHEELHOUSE}"/ai_dynamo-*any.whl
  pip show ai-dynamo ai-dynamo-runtime 2>/dev/null | grep -E "^Name|^Version"
else
  echo "[repro-B] no wheelhouse found at ${WHEELHOUSE}, using stock image dynamo install"
fi

export HF_HOME=/tmp/hf_cache
mkdir -p "$HF_HOME"
until curl -sf "http://${INFRA_IP}:8000/health" >/dev/null 2>&1; do sleep 2; done

# Pre-download the model ONCE, serially, before starting any workers. All
# workers below share this one container-local HF_HOME; if they instead
# raced to download the same blobs concurrently, huggingface_hub's file lock
# fails outright (doesn't block-and-wait) with "Lock acquisition failed".
echo "[repro-B] pre-downloading ${MODEL} to avoid concurrent-download lock contention..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')"

NUM_GPUS="${NUM_GPUS_PER_NODE}"
NUM_DECODE=$(( NUM_GPUS / 2 ))
NUM_PREFILL=$(( NUM_GPUS - NUM_DECODE ))
echo "[repro-B] launching ${NUM_DECODE} decode + ${NUM_PREFILL} prefill workers across ${NUM_GPUS} GPUs"

# The container image bakes in NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME=
# enp226s0 (tuned for some other reference cluster), which doesn't match
# this node's actual NIC name -> gloo/nccl fail to bind; force loopback for
# both (workers are single-GPU, on one node -- no real multi-node NCCL
# needed). For UCX (NIXL's transport): restricting UCX_NET_DEVICES to
# loopback broke even self/intra-agent transport selection ("no route to
# <multicast-addr>"), so keep UCX_NET_DEVICES=all (it picks this node's real
# InfiniBand device, mlx5_0, which IS needed for UCX's own backend init to
# succeed) but disable GPUDirect RDMA specifically via
# UCX_IB_GPU_DIRECT_RDMA=no, since THAT's the actual failing step
# ("ibv_reg_dmabuf_mr ... Unknown error 524", "md supports: host" -- no
# GPUDirect RDMA support in this environment). This forces GPU-memory
# transfers to stage through host memory instead of direct dma-buf.
gpu=0
for i in $(seq 0 $((NUM_DECODE - 1))); do
  port=$((20100 + i))
  DYN_LOG=info NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo UCX_NET_DEVICES=all UCX_IB_GPU_DIRECT_RDMA=no \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$port" CUDA_VISIBLE_DEVICES="$gpu" python3 -m dynamo.vllm \
      --model "$MODEL" --enforce-eager \
      --disaggregation-mode decode \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
      > "${LOGDIR}/decode-${i}.log" 2>&1 &
  gpu=$((gpu + 1))
done

for i in $(seq 0 $((NUM_PREFILL - 1))); do
  port=$((20200 + i))
  events_port=$((20081 + i))
  DYN_LOG=info NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo UCX_NET_DEVICES=all UCX_IB_GPU_DIRECT_RDMA=no \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$port" CUDA_VISIBLE_DEVICES="$gpu" python3 -m dynamo.vllm \
      --model "$MODEL" --enforce-eager \
      --disaggregation-mode prefill \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
      --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${events_port}\",\"enable_kv_cache_events\":true}" \
      > "${LOGDIR}/prefill-${i}.log" 2>&1 &
  gpu=$((gpu + 1))
done

wait
INNER

echo "[repro-B] workers exited. Logs in ${LOGDIR}"
