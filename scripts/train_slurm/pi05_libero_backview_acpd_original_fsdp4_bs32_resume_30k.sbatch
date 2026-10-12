#!/usr/bin/env bash
# Original paper ACPD: resume the complete 5K layers-6+12 run to 30K.
# Save every 5K through 25K, then every 1K; preserve all checkpoints.
#SBATCH --job-name=pi05-bv-acpd-original-r30k
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=8-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-30k/slurm-log/%x_%j.out

set -euo pipefail

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy http_proxy https_proxy no_proxy

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

mkdir -p /opt/liutong/openpi-5090-research/acpd-original-backview-bs32-30k/slurm-log
mkdir -p /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k
mkdir -p "${TMPDIR}"

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.train_distill \
  --name=pi05_libero_backview_acpd_original_layers6_12 \
  --exp-name=pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k \
  --student-config-name=pi05_libero_backview_lora \
  --student-init-params=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --student-view-key=backview_image \
  --teacher-view-key=agentview_image \
  --teacher-use-wrist-image \
  --wrist-view-key=wrist_image \
  --supervised-loss-weight=1.0 \
  --acpd-loss-weight=0.2 \
  --acpd-variance-loss-weight=0.1 \
  --action-corr-loss-weight=0.5 \
  --align-layers 6 12 \
  --acpd-memory-dim=1024 \
  --acpd-projector-hidden-dim=2048 \
  --acpd-detach-query \
  --no-exact-contribution-fusion \
  --no-contribution-feature-fusion \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --asset-id=libero_multiview \
  --checkpoint-base-dir=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k \
  --seed=42 \
  --batch-size=32 \
  --gradient-accumulation-steps=1 \
  --num-workers=8 \
  --num-train-steps=30000 \
  --warmup-steps=1000 \
  --peak-lr=2.5e-5 \
  --decay-steps=30000 \
  --decay-lr=2.5e-6 \
  --log-interval=100 \
  --save-interval=5000 \
  --save-interval-after 25000 1000 \
  --keep-period=1 \
  --resume \
  --resume-from=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/layers_5k/pi05_libero_backview_acpd_lora_layers6_12/pi05_libero_backview_acpd_lora_fsdp2_layers6_12_bs32_5k/4999 \
  --resume-data-loader \
  --fsdp-devices=4 \
  --no-wandb-enabled
