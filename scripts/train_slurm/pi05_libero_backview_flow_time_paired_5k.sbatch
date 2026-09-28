#!/usr/bin/env bash
#SBATCH --job-name=probe-bv-flow-time-paired-5k
#SBATCH --partition=debug01
#SBATCH --nodelist=debug01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/liutong/Projects/openpi-5090/slurm-log/%x_%j.out

set -euo pipefail

source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi

export HF_LEROBOT_HOME=/opt/liutong
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.flow_time_probe \
  --sft-params=/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/4999/params \
  --acpd-params=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_recovery_20k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_20k_keep_all/4999/params \
  --assets-dir=/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist \
  --output-dir=/home/liutong/Projects/openpi-5090/experiments/mechanism/acpd-v2-flow-time-5k/results \
  --batch-size=8 \
  --num-batches=16 \
  --seed=43 \
  --all-times-per-sample
