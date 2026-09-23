#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-sft-bs64-35k-60k
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
  /opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/34999 \
  /opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/34999 \
  pi05_libero_backview_lora \
  backview_image
bash examples/libero/eval_slurm/run_fixed_backview_benchmark_2gpu.bash \
  /opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/39999 \
  /opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/39999 \
  pi05_libero_backview_lora \
  backview_image
bash examples/libero/eval_slurm/run_fixed_backview_benchmark_2gpu.bash \
  /opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/49999 \
  /opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/49999 \
  pi05_libero_backview_lora \
  backview_image
bash examples/libero/eval_slurm/run_fixed_backview_benchmark_2gpu.bash \
  /opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/59999 \
  /opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/59999 \
  pi05_libero_backview_lora \
  backview_image
