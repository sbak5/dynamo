#!/bin/bash
#SBATCH --job-name=dynamo-aiperf-help
#SBATCH --partition=general
#SBATCH --nodelist=ipp2-1635
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/sbak/dynamo_repro/aiperf-help-%j.log

set -euo pipefail
srun --container-image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1 \
  aiperf profile --help
