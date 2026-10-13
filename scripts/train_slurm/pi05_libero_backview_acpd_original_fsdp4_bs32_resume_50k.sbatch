#!/usr/bin/env bash
# Original paper ACPD: continue the existing 30K run to 50K without LR reset.
# Reuse its model, optimizer, deterministic loader, and checkpoint retention.
#SBATCH --job-name=pi05-bv-acpd-original-r50k
#SBATCH --partition=batch
#SBATCH --nodelist=gpu04
#SBATCH --reservation=res_gpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-50k/slurm-log/%x_%j.out

set -euo pipefail

exec bash /home/liutong/Projects/openpi-5090/scripts/train_slurm/pi05_libero_backview_acpd_original_fsdp4_bs32_resume_30k.sbatch \
  --num-train-steps=50000 \
  --resume-from=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k/29999
