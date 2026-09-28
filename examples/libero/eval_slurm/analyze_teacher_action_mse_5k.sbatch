#!/usr/bin/env bash
# Compares the teacher-action MSE ablation with matched H9 and SFT at 5K.
# Submit with --dependency=afterok:<evaluation-array-job>.
#SBATCH --job-name=analyze-teacher-action-mse-5k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy http_proxy https_proxy no_proxy
export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

baseline_h9=/opt/liutong/openpi-5090-evals/acpd-v2-injection-location-5k/final-hidden/4999
baseline_sft=/opt/liutong/openpi-5090-evals/sft-backview-bs64-5k/4999
treatment=/opt/liutong/openpi-5090-evals/acpd-v2-teacher-action-mse-5k/4999
output_dir=/opt/liutong/openpi-5090-evals/acpd-v2-teacher-action-mse-5k/4999

test -d "${baseline_h9}/logs"
test -d "${baseline_sft}/logs"
test -d "${treatment}/logs"
mkdir -p "${output_dir}"

cd /home/liutong/Projects/openpi-5090
python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir="${baseline_h9}" \
  --treatment-dir="${treatment}" \
  --output="${output_dir}/h9_minus_teacher_action_mse.json"
python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir="${baseline_sft}" \
  --treatment-dir="${treatment}" \
  --output="${output_dir}/teacher_action_mse_minus_sft.json"
