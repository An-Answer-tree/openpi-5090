#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-acpdv2-5k
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

cd /home/liutong/Projects/openpi-5090
bash examples/libero/eval_slurm/run_fixed_backview_benchmark.bash \
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_5k/pi05_libero_backview_acpd_v2_layer9/pi05_libero_backview_acpd_v2_lora_fsdp4_mbs8_acc4_bs32_5k/4999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-task-success-5k/layer9-exact-contribution/4999 \
  pi05_libero_backview_acpd_v2_lora
