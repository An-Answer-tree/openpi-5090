#!/usr/bin/env bash
#SBATCH --job-name=pi05-bv-exact-attn-scan
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/slurm-log/%x_%j.out

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

mkdir -p "${TMPDIR}"
mkdir -p /opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results

nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/slurm-log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" &
gpu_monitor_pid=$!
trap 'kill "${gpu_monitor_pid}" 2>/dev/null || true' EXIT

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.exact_attention_probe \
  --student-init-params=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/task_success_5k/pi05_libero_backview_flow_only_5k/pi05_libero_backview_lora_fsdp2_bs32_5k/4999/params \
  --teacher-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --result-dir=/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --student-view-key=backview_image \
  --teacher-view-key=agentview_image \
  --wrist-view-key=wrist_image \
  --layers 7 8 10 11 \
  --probe-seeds 11 29 47 \
  --batch-size=8 \
  --gradient-accumulation-steps=4 \
  --num-train-steps=500 \
  --num-validation-batches=32 \
  --validation-fraction=0.1 \
  --learning-rate=1e-3 \
  --log-interval=50 \
  --bootstrap-samples=2000 \
  --seed=42 \
  --fsdp-devices=2
