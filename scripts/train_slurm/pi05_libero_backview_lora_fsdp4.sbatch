#!/usr/bin/env bash
#SBATCH --job-name=pi05-bv-lora
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --exclude=gpu01
#SBATCH --output=/opt/liutong/openpi_slurm/logs/%x_%j.out

set -euo pipefail

repo_root=/home/liutong/Projects/openpi-5090
env_root=/var/tmp/liutong/conda-envs/openpi
env_archive=/opt/liutong/openpi_slurm/openpi-env.tar.zst
log_root=/opt/liutong/openpi_slurm/logs
checkpoint_root=/opt/liutong/openpi-5090-checkpoints
gpu_log=${log_root}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv

mkdir -p /var/tmp/liutong/conda-envs "${log_root}" "${checkpoint_root}"
if [[ ! -x "${env_root}/bin/python" ]]; then
  tar -C /var/tmp/liutong/conda-envs \
    -I /home/liutong/miniconda3/bin/zstd \
    -xf "${env_archive}"
fi

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "${gpu_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

export HF_LEROBOT_HOME=/opt/liutong
export OPENPI_DATA_HOME=/home/liutong/.cache/openpi
export PATH="${env_root}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${repo_root}"
"${env_root}/bin/python" scripts/train.py pi05_libero_backview_lora \
  --exp-name=pi05_libero_backview_lora_fsdp4_bs32_30k \
  --checkpoint-base-dir="${checkpoint_root}" \
  --data.assets.assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --batch-size=32 \
  --fsdp-devices=4 \
  --num-workers=8 \
  --num-train-steps=30000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --no-wandb-enabled
