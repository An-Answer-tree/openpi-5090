#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-acl5k
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
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/task_success_5k/pi05_libero_backview_acl_only_5k/pi05_libero_backview_acpd_lora_fsdp2_bs32_5k/4999 \
  /opt/liutong/openpi-5090-evals/acpd-task-success-5k/acl-only/4999
