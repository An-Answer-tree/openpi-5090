#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-smoke
#SBATCH --partition=debug01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/liutong/Projects/openpi-5090/slurm-log/%x_%j.out

set -euo pipefail

cd /home/liutong/Projects/openpi-5090
bash examples/libero/eval_slurm/run_fixed_backview_suite.bash \
  /opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp2_bs16_30k/29999 \
  /opt/liutong/openpi-5090-evals/smoke/pi05_libero_backview_lora_fsdp2_bs16_30k/29999/spatial \
  "${CUDA_VISIBLE_DEVICES}" \
  "$((20000 + SLURM_JOB_ID % 10000))" \
  libero_spatial \
  1
