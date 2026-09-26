#!/usr/bin/env bash
#SBATCH --job-name=analyze-h17-decay-15k
#SBATCH --partition=batch
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

cd /home/liutong/Projects/openpi-5090
treatment=/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/14999-corrected
output_dir=/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/14999-corrected

python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir=/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/14999 \
  --treatment-dir="${treatment}" \
  --output="${output_dir}/h17_vs_h9.json" \
  | tee "${output_dir}/h17_vs_h9.md"

python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir=/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/sft/15000 \
  --treatment-dir="${treatment}" \
  --output="${output_dir}/h17_vs_sft.json" \
  | tee "${output_dir}/h17_vs_sft.md"
