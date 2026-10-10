#!/usr/bin/env bash
#SBATCH --job-name=summary-h9-injection-off-35k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

cd /home/liutong/Projects/openpi-5090
bash examples/libero/eval_slurm/summarize_fixed_backview_benchmark.sbatch \
  /opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/34999

source /home/liutong/miniconda3/etc/profile.d/conda.sh
set +u
conda activate openpi
set -u

python -m scripts.distillation_acpd.analyze_paired_eval \
  --baseline-dir=/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/34999 \
  --treatment-dir=/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999 \
  --bootstrap-samples=10000 \
  --seed=42 \
  --expected-trials-per-task=50 \
  --output=experiments/mechanism/acpd-v2-h9-inference-injection-35k/results/paired_analysis.json
