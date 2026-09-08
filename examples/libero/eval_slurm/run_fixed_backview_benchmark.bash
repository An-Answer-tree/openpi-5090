#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR EVAL_DIR" >&2
  exit 2
fi

checkpoint_dir=$1
eval_dir=$2

test -f "${checkpoint_dir}/_CHECKPOINT_METADATA"
mkdir -p "${eval_dir}/logs"

port_base=$((20000 + (SLURM_JOB_ID % 10000) * 4))
IFS=',' read -ra allocated_gpus <<<"${CUDA_VISIBLE_DEVICES}"
suite_pids=()

cleanup() {
  if ((${#suite_pids[@]})); then
    kill "${suite_pids[@]}" 2>/dev/null || true
    wait "${suite_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_suite() {
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
    50 >"${eval_dir}/logs/${output_name}.log" 2>&1 &
  suite_pids+=("$!")
}

cd /home/liutong/Projects/openpi-5090
start_suite "${allocated_gpus[0]}" "${port_base}" libero_spatial spatial
start_suite "${allocated_gpus[1]}" "$((port_base + 1))" libero_object object
start_suite "${allocated_gpus[2]}" "$((port_base + 2))" libero_goal goal
start_suite "${allocated_gpus[3]}" "$((port_base + 3))" libero_10 libero_10

for suite_pid in "${suite_pids[@]}"; do
  wait "${suite_pid}"
done

grep -H "Final success rate" "${eval_dir}"/logs/*.log | tee "${eval_dir}/summary.txt"
