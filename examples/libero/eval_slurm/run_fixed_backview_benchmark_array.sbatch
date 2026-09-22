#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --exclude=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%A_%a.out

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR EVAL_DIR [POLICY_CONFIG] [BASE_IMAGE_KEY]" >&2
  exit 2
fi

checkpoint_dir=$1
eval_dir=$2
policy_config=${3:-pi05_libero_backview_lora}
base_image_key=${4:-backview_image}
suite_names=(libero_spatial libero_object libero_goal libero_10)
output_names=(spatial object goal libero_10)
task_id=${SLURM_ARRAY_TASK_ID}

test -f "${checkpoint_dir}/_CHECKPOINT_METADATA"
mkdir -p "${eval_dir}/logs"

gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
port=$((20000 + (SLURM_ARRAY_JOB_ID % 10000) * 4 + task_id))

cd /home/liutong/Projects/openpi-5090
bash examples/libero/eval_slurm/run_fixed_backview_suite.bash \
  "${checkpoint_dir}" \
  "${eval_dir}/${output_names[task_id]}" \
  "${gpu_id}" \
  "${port}" \
  "${suite_names[task_id]}" \
  50 \
  "${policy_config}" \
  "${base_image_key}" >"${eval_dir}/logs/${output_names[task_id]}.log" 2>&1
