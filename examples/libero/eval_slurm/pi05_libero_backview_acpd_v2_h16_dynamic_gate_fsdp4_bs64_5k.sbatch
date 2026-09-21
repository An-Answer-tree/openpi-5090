#!/usr/bin/env bash
#SBATCH --job-name=eval-bv-h16-dynamic-gate-bs64-5k
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
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/dynamic_view_gate_5k/pi05_libero_backview_acpd_v2_h16_dynamic_gate/pi05_libero_backview_acpd_v2_h16_dynamic_gate_lora_fsdp4_bs64_5k/4999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-h16-dynamic-view-gate-5k/4999 \
  pi05_libero_backview_acpd_v2_layer10_dynamic_gate_lora
