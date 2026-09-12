#!/usr/bin/env bash
#SBATCH --job-name=pi05-bv-acl-2k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/liutong/Projects/openpi-5090/slurm-log/%x_%j.out

set -euo pipefail

export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi

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

mkdir -p /home/liutong/Projects/openpi-5090/slurm-log
mkdir -p /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/components_2k
mkdir -p "${TMPDIR}"

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "/home/liutong/Projects/openpi-5090/slurm-log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" &
trap 'kill $! 2>/dev/null || true' EXIT

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.train_distill \
  --name=pi05_libero_backview_acpd_acl_only \
  --exp-name=pi05_libero_backview_acpd_acl_only_lora_fsdp2_bs32_2k \
  --student-config-name=pi05_libero_backview_lora \
  --student-init-params=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --student-view-key=backview_image \
  --teacher-view-key=agentview_image \
  --teacher-use-wrist-image \
  --wrist-view-key=wrist_image \
  --supervised-loss-weight=1.0 \
  --acpd-loss-weight=0.0 \
  --acpd-variance-loss-weight=0.1 \
  --action-corr-loss-weight=0.5 \
  --align-layers 6 12 \
  --acpd-memory-dim=1024 \
  --acpd-projector-hidden-dim=2048 \
  --acpd-detach-query \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --asset-id=libero_multiview \
  --checkpoint-base-dir=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/components_2k \
  --seed=42 \
  --batch-size=32 \
  --gradient-accumulation-steps=1 \
  --num-workers=8 \
  --num-train-steps=2000 \
  --warmup-steps=1000 \
  --peak-lr=2.5e-5 \
  --decay-steps=30000 \
  --decay-lr=2.5e-6 \
  --log-interval=100 \
  --save-interval=100000 \
  --keep-period=100000 \
  --no-save-final-checkpoint \
  --fsdp-devices=2 \
  --no-wandb-enabled
