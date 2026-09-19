#!/usr/bin/env bash
#SBATCH --job-name=pi05-bv-lora2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/liutong/Projects/openpi-5090/slurm-log/%x_%j.out

set -euo pipefail

export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi

mkdir -p /opt/liutong/openpi-5090-checkpoints

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "/home/liutong/Projects/openpi-5090/slurm-log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" &
trap 'kill $! 2>/dev/null || true' EXIT

export HF_LEROBOT_HOME=/opt/liutong
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/home/liutong/.cache/openpi
export PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /home/liutong/Projects/openpi-5090
python scripts/train.py pi05_libero_backview_lora \
  --exp-name=pi05_libero_backview_lora_fsdp2_bs16_30k \
  --checkpoint-base-dir=/opt/liutong/openpi-5090-checkpoints \
  --data.assets.assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --weight-loader.params-path=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --batch-size=16 \
  --fsdp-devices=2 \
  --num-workers=4 \
  --num-train-steps=30000 \
  --save-interval=5000 \
  --keep-period=5000 \
  --no-wandb-enabled
