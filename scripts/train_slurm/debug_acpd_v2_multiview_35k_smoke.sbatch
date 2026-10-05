#!/usr/bin/env bash
# One-GPU smoke test for the three ACPD-v2 multiview configurations.
#SBATCH --job-name=debug-acpdv2-multiview-smoke
#SBATCH --partition=debug01
#SBATCH --nodelist=debug01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --time=01:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-v2-multiview-35k/slurm-log/%x_%A_%a.out

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

views=(leftview rightview topview)
view="${views[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p /opt/liutong/openpi-5090-research/acpd-v2-multiview-35k/slurm-log
mkdir -p /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k_smoke/${view}
mkdir -p "${TMPDIR}"

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.train_distill \
  --name=pi05_libero_${view}_acpd_v2_layer10_smoke \
  --exp-name=pi05_libero_${view}_acpd_v2_layer10_smoke \
  --student-config-name=pi05_libero_${view}_lora \
  --student-init-params=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --student-view-key=${view}_image \
  --teacher-view-key=agentview_image \
  --teacher-use-wrist-image \
  --wrist-view-key=wrist_image \
  --supervised-loss-weight=1.0 \
  --acpd-loss-weight=0.2 \
  --action-corr-loss-weight=0.5 \
  --align-layers 10 \
  --exact-contribution-fusion \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --asset-id=libero_multiview \
  --checkpoint-base-dir=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k_smoke/${view} \
  --seed=42 \
  --batch-size=1 \
  --gradient-accumulation-steps=1 \
  --num-workers=0 \
  --num-train-steps=2 \
  --warmup-steps=1 \
  --peak-lr=2.5e-5 \
  --decay-steps=2 \
  --decay-lr=2.5e-6 \
  --log-interval=1 \
  --save-interval=100000 \
  --keep-period=100000 \
  --no-save-final-checkpoint \
  --fsdp-devices=1 \
  --no-wandb-enabled
