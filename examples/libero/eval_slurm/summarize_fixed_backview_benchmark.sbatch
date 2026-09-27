#!/usr/bin/env bash
#SBATCH --job-name=summary-bv-eval
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

eval_dir=$1
summary="${eval_dir}/summary.txt"

for suite in spatial object goal libero_10; do
  grep -H 'Final success rate' "${eval_dir}/logs/${suite}.log"
done > "${summary}"

pooled=$(awk -F '[()]' '
  { split($2, counts, "/"); success += counts[1]; episodes += counts[2] }
  END {
    if (NR != 4 || episodes != 2000) exit 1
    printf "Pooled success rate: %.4f (%d/%d)", success / episodes, success, episodes
  }
' "${summary}")
printf '%s\n' "${pooled}" >> "${summary}"
