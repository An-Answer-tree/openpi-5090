#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-lora4-cos50
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
  /opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/50000 \
  /opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/50000
