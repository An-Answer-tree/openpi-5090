#!/usr/bin/env bash
#SBATCH --job-name=summary-h17-decay-15k
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/opt/liutong/openpi-5090-evals/slurm-log/%x_%j.out

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EVAL_DIR" >&2
  exit 2
fi

eval_dir=$1
test -d "${eval_dir}/logs"

python - "${eval_dir}" <<'PY'
import json
import pathlib
import re
import sys

eval_dir = pathlib.Path(sys.argv[1])
suites = ("spatial", "object", "goal", "libero_10")
pattern = re.compile(r"Final success rate:\s*([0-9.]+)\s*\((\d+)\/(\d+)\)")
rows = []

for suite in suites:
  log_path = eval_dir / "logs" / f"{suite}.log"
  text = log_path.read_text(encoding="utf-8")
  matches = pattern.findall(text)
  if not matches:
    raise RuntimeError(f"No final success rate in {log_path}")
  rate, success, episodes = matches[-1]
  rows.append({"suite": suite, "success_rate": float(rate),
               "success": int(success), "episodes": int(episodes)})

success = sum(row["success"] for row in rows)
episodes = sum(row["episodes"] for row in rows)
pooled = success / episodes
result = {"checkpoint_step": 14999, "suites": rows, "pooled": pooled,
          "success": success, "episodes": episodes}
(eval_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                        encoding="utf-8")
(eval_dir / "summary.md").write_text(
    "# H17-Decay 15K\n\n"
    "| Suite | Success | Episodes | Rate |\n|---|---:|---:|---:|\n"
    + "".join(f"| {row['suite']} | {row['success']} | {row['episodes']} | "
              f"{row['success_rate']:.4f} |\n" for row in rows)
    + f"| **Pooled** | **{success}** | **{episodes}** | **{pooled:.4f}** |\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2))
PY
