#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-h11-aligned-bs64-5k
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
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/injection_location_30k/pi05_libero_backview_acpd_v2_h11_layer10_aligned/pi05_libero_backview_acpd_v2_h11_layer10_aligned_lora_fsdp4_bs64_30k/4999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-injection-location-5k/aligned-attention/4999 \
  pi05_libero_backview_acpd_v2_layer10_aligned_lora
