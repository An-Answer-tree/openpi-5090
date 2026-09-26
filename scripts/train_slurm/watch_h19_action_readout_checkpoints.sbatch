#!/usr/bin/env bash
# Polls the two planned checkpoints and submits their evaluation arrays.
#SBATCH --job-name=watch-h19-action-readout
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/slurm-log/%x_%j.out

set -euo pipefail

checkpoint_root=/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/action_readout_30k/pi05_libero_backview_acpd_v2_action_readout/pi05_libero_backview_acpd_v2_action_readout_lora_fsdp4_bs64_30k
eval_root=/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/eval
eval_script=/home/liutong/Projects/openpi-5090/examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch
policy_config=pi05_libero_backview_acpd_v2_layer10_action_readout_lora

mkdir -p "${eval_root}"

submit_when_ready() {
  local step=$1
  local label=$2
  local checkpoint_dir="${checkpoint_root}/${step}"
  local eval_dir="${eval_root}/${label}"
  local submitted_marker="${eval_dir}/.submitted"

  while [[ ! -f "${checkpoint_dir}/_CHECKPOINT_METADATA" ]]; do
    sleep 60
  done
  if [[ -f "${submitted_marker}" ]]; then
    return
  fi

  mkdir -p "${eval_dir}"
  job_id=$(sbatch --parsable \
    --job-name="eval-h19-ar-${label}" \
    --mem=24G \
    --cpus-per-task=8 \
    --time=2-00:00:00 \
    "${eval_script}" \
    "${checkpoint_dir}" \
    "${eval_dir}" \
    "${policy_config}" \
    backview_image \
    250)
  printf '%s\n' "${job_id}" >"${submitted_marker}"
  printf 'Submitted H19 %s evaluation as %s\n' "${label}" "${job_id}"
}

submit_when_ready 4999 5k
submit_when_ready 9999 10k
