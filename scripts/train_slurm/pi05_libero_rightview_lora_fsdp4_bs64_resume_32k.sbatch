#!/usr/bin/env bash
#SBATCH --job-name=pi05-rv-sft-bs64-r32k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/sft-multiview-bs64-resume-32k/slurm-log/%x_%j.out

set -euo pipefail

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy http_proxy https_proxy no_proxy

export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi

export HF_HOME=/opt/liutong/huggingface
export HF_DATASETS_CACHE=/opt/liutong/huggingface/datasets
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
export TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD=1
export TMPDIR="/opt/liutong/openpi-slurm-tmp/${SLURM_JOB_ID}"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${TMPDIR}"

cd /home/liutong/Projects/openpi-5090
python scripts/train.py pi05_libero_rightview_lora \
  --exp-name=pi05_libero_rightview_lora_fsdp4_bs64_30k \
  --checkpoint-base-dir=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/rightview_bs64_30k \
  --data.assets.assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --weight-loader.params-path=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --seed=42 \
  --batch-size=64 \
  --fsdp-devices=4 \
  --num-workers=8 \
  --num-train-steps=32000 \
  --lr-schedule.warmup-steps=1000 \
  --lr-schedule.peak-lr=2.5e-5 \
  --lr-schedule.decay-steps=30000 \
  --lr-schedule.decay-lr=2.5e-6 \
  --save-interval=1000 \
  --keep-period=1 \
  --resume \
  --resume-data-loader \
  --no-wandb-enabled
