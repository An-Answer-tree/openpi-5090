#!/usr/bin/env bash
# Reconstruct ACPD-v2-Fixed checkpoints 31K--34K from the complete 30K state.
# Read the source checkpoint in place; write only to the separate trajectory.
#SBATCH --job-name=pi05-lv-acpdv2-bs64-r34k
#SBATCH --partition=batch
#SBATCH --nodelist=gpu04
#SBATCH --reservation=res_gpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-v2-trajectory-31k-34k/slurm-log/%x_%j.out

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
python -m scripts.distillation_acpd.train_distill \
  --name=pi05_libero_leftview_acpd_v2_layer10 \
  --exp-name=pi05_libero_leftview_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k \
  --student-config-name=pi05_libero_leftview_lora \
  --student-init-params=/home/liutong/.cache/openpi-5090/pi05_base/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --student-view-key=leftview_image \
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
  --checkpoint-base-dir=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/leftview \
  --seed=42 \
  --batch-size=64 \
  --gradient-accumulation-steps=1 \
  --num-workers=0 \
  --num-train-steps=34000 \
  --warmup-steps=1000 \
  --peak-lr=2.5e-5 \
  --decay-steps=30000 \
  --decay-lr=2.5e-6 \
  --log-interval=100 \
  --save-interval=1000 \
  --keep-period=1 \
  --resume \
  --resume-from=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/leftview/pi05_libero_leftview_acpd_v2_layer10/pi05_libero_leftview_acpd_v2_layer10_lora_fsdp4_bs64_35k/29999 \
  --resume-data-loader \
  --fsdp-devices=4 \
  --no-wandb-enabled
