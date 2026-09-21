#!/usr/bin/env bash
#SBATCH --job-name=probe-bv-acpdv2-grad-fast
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
mkdir -p /opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict
mkdir -p "${TMPDIR}"

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.gradient_conflict_probe \
  --checkpoint-label=h9-5k-bs1-n100 \
  --student-params=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_recovery_20k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_20k_keep_all/4999/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --output-dir=/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-5k-bs1-n100 \
  --batch-size=1 \
  --num-batches=100 \
  --num-workers=0 \
  --fsdp-devices=1

python -m scripts.distillation_acpd.gradient_conflict_probe \
  --checkpoint-label=h9-30k-bs1-n100 \
  --student-params=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --output-dir=/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-30k-bs1-n100 \
  --batch-size=1 \
  --num-batches=100 \
  --num-workers=0 \
  --fsdp-devices=1
