#!/bin/bash
#SBATCH --job-name=dynamo-aiperf-version
#SBATCH --partition=general
#SBATCH --nodelist=ipp2-1635
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/sbak/dynamo_repro/aiperf-version-%j.log

set -euo pipefail
srun --container-image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1 \
  bash -lc 'which aiperf; aiperf --version; pip index versions aiperf'
