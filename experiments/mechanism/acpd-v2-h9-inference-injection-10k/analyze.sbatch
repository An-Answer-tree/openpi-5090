#!/usr/bin/env bash
#SBATCH --job-name=analyze-h9-injection-10k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

source /home/liutong/miniconda3/etc/profile.d/conda.sh
conda activate openpi
cd /home/liutong/Projects/openpi-5090

python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir=/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/9999 \
  --treatment-dir=/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/9999 \
  --output=experiments/mechanism/acpd-v2-h9-inference-injection-10k/results/paired_analysis.json
