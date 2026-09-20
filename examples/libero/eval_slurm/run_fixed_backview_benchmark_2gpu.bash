#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR EVAL_DIR [POLICY_CONFIG] [BASE_IMAGE_KEY]" >&2
  exit 2
fi

checkpoint_dir=$1
eval_dir=$2
policy_config=${3:-pi05_libero_backview_lora}
base_image_key=${4:-backview_image}

test -f "${checkpoint_dir}/_CHECKPOINT_METADATA"
mkdir -p "${eval_dir}/logs"

port_base=$((20000 + (SLURM_JOB_ID % 10000) * 4))
IFS=',' read -ra allocated_gpus <<<"${CUDA_VISIBLE_DEVICES}"
pair_pids=()

cleanup() {
  if ((${#pair_pids[@]})); then
    kill "${pair_pids[@]}" 2>/dev/null || true
    wait "${pair_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_suite() {
  local gpu_id=$1
  local port=$2
  local suite_name=$3
  local output_name=$4

  bash examples/libero/eval_slurm/run_fixed_backview_suite.bash \
    "${checkpoint_dir}" \
    "${eval_dir}/${output_name}" \
    "${gpu_id}" \
    "${port}" \
    "${suite_name}" \
    50 \
    "${policy_config}" \
    "${base_image_key}" >"${eval_dir}/logs/${output_name}.log" 2>&1
}

run_pair() {
  local gpu_id=$1
  local first_port=$2
  local first_suite=$3
  local first_output=$4
  local second_port=$5
  local second_suite=$6
  local second_output=$7

  run_suite "${gpu_id}" "${first_port}" "${first_suite}" "${first_output}"
  run_suite "${gpu_id}" "${second_port}" "${second_suite}" "${second_output}"
}

cd /home/liutong/Projects/openpi-5090
run_pair "${allocated_gpus[0]}" "${port_base}" libero_spatial spatial \
  "$((port_base + 2))" libero_goal goal &
pair_pids+=("$!")
run_pair "${allocated_gpus[1]}" "$((port_base + 1))" libero_object object \
  "$((port_base + 3))" libero_10 libero_10 &
pair_pids+=("$!")

for pair_pid in "${pair_pids[@]}"; do
  wait "${pair_pid}"
done

grep -H "Final success rate" "${eval_dir}"/logs/*.log | tee "${eval_dir}/summary.txt"
