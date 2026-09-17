#!/usr/bin/env bash
#SBATCH --job-name=finish-bv-acpdv2-l9-5k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/liutong/Projects/openpi-5090/slurm-log/%x_%j.out

set -euo pipefail

training_job_id="${1:?Usage: sbatch $0 TRAINING_JOB_ID}"

mkdir -p /home/liutong/Projects/openpi-5090/slurm-log

until [[ -f "/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_30k/pi05_libero_backview_acpd_v2_layer9/pi05_libero_backview_acpd_v2_lora_fsdp4_mbs8_acc4_bs32_30k/4999/_CHECKPOINT_METADATA" && \
         -d "/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_30k/pi05_libero_backview_acpd_v2_layer9/pi05_libero_backview_acpd_v2_lora_fsdp4_mbs8_acc4_bs32_30k/4999/params" && \
         -d "/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_30k/pi05_libero_backview_acpd_v2_layer9/pi05_libero_backview_acpd_v2_lora_fsdp4_mbs8_acc4_bs32_30k/4999/assets" ]]; do
  sleep 60
done

evaluation_job_id=$(sbatch --parsable /home/liutong/Projects/openpi-5090/examples/libero/eval_slurm/pi05_libero_backview_acpd_v2_layer9_fsdp4_bs32_5k.sbatch)
echo "Submitted layer-9 evaluation job ${evaluation_job_id}; stopping training job ${training_job_id}."
scancel "${training_job_id}"
