#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-sft-bs64-30k
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

cd /home/liutong/Projects/openpi-5090
bash examples/libero/eval_slurm/run_fixed_backview_benchmark_2gpu.bash \
  /opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/29999 \
  /opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999 \
  pi05_libero_backview_lora \
  backview_image
