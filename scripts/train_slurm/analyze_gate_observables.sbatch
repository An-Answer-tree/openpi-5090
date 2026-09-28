#!/usr/bin/env bash
# Summarizes the two observable gate probe array tasks after they finish.
# Pass the parent array id with --export=GATE_PROBE_JOB_ID=<id>.
#SBATCH --job-name=analyze-gate-observables
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/slurm-log/%x_%j.out

set -euo pipefail

export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /home/liutong/Projects/openpi-5090
results_root=/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results
output_path="${results_root}/${GATE_PROBE_JOB_ID}/observable-reliability.json"
declare -A inputs=()
declare -A seen_summaries=()

record_summary() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] || return
  [[ -n "${seen_summaries[${summary_path}]:-}" ]] && return
  seen_summaries["${summary_path}"]=1
  local checkpoint_path
  checkpoint_path=$(jq -r '.config.checkpoint_params' "${summary_path}")
  case "${checkpoint_path}" in
    */9999/params) inputs[10k]="${summary_path%/*}/per_sample.npz" ;;
    */29999/params) inputs[30k]="${summary_path%/*}/per_sample.npz" ;;
  esac
}

while read -r child_id; do
  record_summary "${results_root}/${child_id}/summary.json"
done < <(sacct -n -X -P -j "${GATE_PROBE_JOB_ID}" -o JobIDRaw | sed '/^[[:space:]]*$/d')

# sacct may be temporarily unavailable; the array log names are deterministic.
for task_id in 0 1; do
  log_path="/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/slurm-log/probe-bv-acpdv2-gate-effect_${GATE_PROBE_JOB_ID}_${task_id}.out"
  output_dir=$(sed -n 's/.*Output directory: //p' "${log_path}" 2>/dev/null | tail -n 1)
  [[ -n "${output_dir}" ]] && record_summary "${output_dir}/summary.json"
done

[[ -f "${inputs[10k]:-}" ]] || { echo "Missing 10K probe output" >&2; exit 1; }
[[ -f "${inputs[30k]:-}" ]] || { echo "Missing 30K probe output" >&2; exit 1; }

mkdir -p "${results_root}/${GATE_PROBE_JOB_ID}"
conda run -n openpi python -m scripts.distillation_acpd.analyze_gate_observables \
  --input="${inputs[10k]}" \
  --output="${results_root}/${GATE_PROBE_JOB_ID}/observable-reliability-10k.json"
conda run -n openpi python -m scripts.distillation_acpd.analyze_gate_observables \
  --input="${inputs[30k]}" \
  --output="${results_root}/${GATE_PROBE_JOB_ID}/observable-reliability-30k.json"
jq -n \
  --slurpfile tenk "${results_root}/${GATE_PROBE_JOB_ID}/observable-reliability-10k.json" \
  --slurpfile thirtyk "${results_root}/${GATE_PROBE_JOB_ID}/observable-reliability-30k.json" \
  '{"10k": $tenk[0], "30k": $thirtyk[0]}' > "${output_path}"
