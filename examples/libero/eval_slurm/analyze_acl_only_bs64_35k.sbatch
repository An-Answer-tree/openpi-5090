#!/usr/bin/env bash
#SBATCH --job-name=analyze-bv-acl-only-bs64
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

if [[ $# -ne 1 || ( $1 != 29999 && $1 != 34999 ) ]]; then
  echo "Usage: $0 {29999|34999}" >&2
  exit 2
fi

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy http_proxy https_proxy no_proxy
export CONDA_ENVS_PATH=/home/liutong/miniconda3/envs
source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

if [[ $1 == 29999 ]]; then
  sft_dir=/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999
else
  sft_dir=/opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/35000
fi

acl_dir=/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/$1
h9_dir=/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/$1

cd /home/liutong/Projects/openpi-5090
grep -H 'Final success rate' "${acl_dir}"/logs/*.log > "${acl_dir}/summary.txt"
python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir="${acl_dir}" \
  --treatment-dir="${h9_dir}" \
  --output="${acl_dir}/h9_minus_acl.json"
python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir="${sft_dir}" \
  --treatment-dir="${acl_dir}" \
  --output="${acl_dir}/acl_minus_sft.json"
