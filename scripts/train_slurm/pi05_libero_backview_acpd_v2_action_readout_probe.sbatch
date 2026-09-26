#!/usr/bin/env bash
#SBATCH --job-name=probe-bv-acpdv2-action-readout
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/slurm-log/%x_%j.out

set -euo pipefail

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy http_proxy https_proxy no_proxy
export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi

export HF_LEROBOT_HOME=/opt/liutong
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=/home/liutong/.cache/openpi
export PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/opt/liutong/openpi-slurm-tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.action_readout_probe \
  --student-params=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999/params \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --output-dir=/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results \
  --dataset-repo-id=libero_multiview_tuned_6view_lerobot \
  --batch-size=8 \
  --num-train-steps=500 \
  --num-validation-batches=128 \
  --validation-fraction=0.1 \
  --hidden-width=128 \
  --learning-rate=1e-3 \
  --seed=42 "$@"
