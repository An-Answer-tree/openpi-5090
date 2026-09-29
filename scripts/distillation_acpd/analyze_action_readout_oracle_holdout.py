"""Cross-checks H18 oracle readout selection on held-out samples."""

import argparse
import json
from pathlib import Path

import numpy as np


ARM_NAMES = ("frozen_h9", "hidden_only", "hidden_layer10", "hidden_contribution")
BASELINE_ARM = 2


def _bootstrap(delta: np.ndarray, *, samples: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    values = delta[indices].mean(axis=1)
    return float(values.mean()), np.percentile(values, [2.5, 97.5]).tolist()


def analyze(path: Path, *, bootstrap_samples: int, seed: int) -> dict:
    """Selects arms on alternating samples and evaluates the other half."""
    with np.load(path, allow_pickle=False) as data:
        errors = np.asarray(data["errors"], dtype=np.float64)
        episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    episode_ids = np.unique(episode_index)
    episode_deltas = []
    selected_counts = np.zeros(len(ARM_NAMES), dtype=np.int64)
    eligible_episodes = 0
    for episode_id in episode_ids:
        rows = np.flatnonzero(episode_index == episode_id)
        if rows.size < 2:
            continue
        eligible_episodes += 1
        direction_deltas = []
        for parity in (0, 1):
            calibration = rows[parity::2]
            evaluation = rows[1 - parity :: 2]
            if calibration.size == 0 or evaluation.size == 0:
                continue
            selected_arm = int(np.argmin(errors[calibration].mean(axis=0)))
            selected_counts[selected_arm] += 1
            direction_deltas.append(
                errors[evaluation, selected_arm].mean() - errors[evaluation, BASELINE_ARM].mean()
            )
        if direction_deltas:
            episode_deltas.append(np.mean(direction_deltas))
    deltas = np.asarray(episode_deltas, dtype=np.float64)
    mean_delta, ci = _bootstrap(deltas, samples=bootstrap_samples, seed=seed)
    baseline_mse = float(errors[:, BASELINE_ARM].mean())
    return {
        "source": str(path),
        "samples": int(errors.shape[0]),
        "episodes": int(episode_ids.size),
        "eligible_episodes": int(eligible_episodes),
        "arm_names": list(ARM_NAMES),
        "baseline_arm": ARM_NAMES[BASELINE_ARM],
        "holdout_oracle_minus_baseline": mean_delta,
        "holdout_oracle_minus_baseline_ci95": ci,
        "holdout_oracle_relative_improvement_percent": float(-mean_delta / baseline_mse * 100.0),
        "selection_counts": {
            name: int(count) for name, count in zip(ARM_NAMES, selected_counts)
        },
        "passes_holdout_oracle_screen": bool(-mean_delta / baseline_mse * 100.0 >= 1.0 and ci[1] < 0.0),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.input, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
