#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR EVAL_DIR GPU_ID PORT SUITE NUM_TRIALS" >&2
  exit 2
fi

checkpoint_dir=$1
eval_dir=$2
gpu_id=$3
port=$4
suite_name=$5
num_trials=$6

test -f "${checkpoint_dir}/_CHECKPOINT_METADATA"
mkdir -p "${eval_dir}"
export TMPDIR="/dev/shm/openpi-slurm-tmp/${SLURM_JOB_ID}/${suite_name}"
mkdir -p "${TMPDIR}"

source /home/liutong/miniconda3/etc/profile.d/conda.sh

export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export LIBERO_CONFIG_PATH=/home/liutong/.cache/openpi/libero
export MUJOCO_GL=egl
export OPENPI_DATA_HOME=/home/liutong/.cache/openpi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/liutong/Projects/openpi-5090/third_party/libero:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

set +u
conda activate openpi
set -u
CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/serve_policy.py \
  --env LIBERO \
  --port "${port}" \
  --no-record \
  policy:checkpoint \
  --policy.config pi05_libero_backview_lora \
  --policy.dir "${checkpoint_dir}" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 360); do
  if curl --silent --fail --noproxy '*' "http://127.0.0.1:${port}/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server for ${suite_name} exited during startup." >&2
    wait "${server_pid}"
  fi
  sleep 5
done
curl --silent --fail --noproxy '*' "http://127.0.0.1:${port}/healthz" >/dev/null

set +u
conda activate openpi-libero
set -u
CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
  python -m examples.libero.multiview_eval.main \
  --host 127.0.0.1 \
  --port "${port}" \
  --resize-size 224 \
  --replan-steps 5 \
  --base-image-key backview_image \
  --task-suite-name "${suite_name}" \
  --num-steps-wait 10 \
  --num-trials-per-task "${num_trials}" \
  --video-out-path "${eval_dir}/videos" \
  --save-videos \
  --seed 7
