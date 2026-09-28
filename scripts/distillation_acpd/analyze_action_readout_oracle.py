"""Analyzes the conditional routing upper bound of H18 action readouts."""

import argparse
import json
from pathlib import Path

import numpy as np


ARM_NAMES = ("frozen_h9", "hidden_only", "hidden_layer10", "hidden_contribution")


def _episode_means(errors: np.ndarray, episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns episode IDs and mean error for each arm."""
    episode_ids = np.unique(episode_index)
    means = np.empty((episode_ids.size, errors.shape[1]), dtype=np.float64)
    for row, episode_id in enumerate(episode_ids):
        means[row] = errors[episode_index == episode_id].mean(axis=0)
    return episode_ids, means


def _bootstrap_delta(
    episode_means: np.ndarray,
    baseline: np.ndarray,
    oracle: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Bootstraps oracle minus fixed-baseline episode MSE."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, episode_means.shape[0], size=(samples, episode_means.shape[0]))
    deltas = (episode_means[indices, oracle] - episode_means[indices, baseline]).mean(axis=1)
    return float(deltas.mean()), np.percentile(deltas, [2.5, 97.5]).tolist()


def analyze(path: Path, *, bootstrap_samples: int, seed: int) -> dict:
    """Computes fixed, oracle, and winner statistics."""
    with np.load(path, allow_pickle=False) as data:
        errors = np.asarray(data["errors"], dtype=np.float64)
        episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    if errors.ndim != 2 or errors.shape[1] != len(ARM_NAMES):
        raise ValueError(f"Expected errors with shape [N, {len(ARM_NAMES)}], got {errors.shape}.")
    episode_ids, episode_means = _episode_means(errors, episode_index)
    fixed_means = episode_means.mean(axis=0)
    best_fixed = int(np.argmin(fixed_means))
    episode_oracle = np.argmin(episode_means, axis=1)
    episode_oracle_means = episode_means[np.arange(episode_means.shape[0]), episode_oracle]
    sample_oracle = np.argmin(errors, axis=1)
    sample_oracle_mean = float(errors[np.arange(errors.shape[0]), sample_oracle].mean())
    bootstrap_mean, bootstrap_ci = _bootstrap_delta(
        episode_means,
        np.full(episode_means.shape[0], best_fixed),
        episode_oracle,
        samples=bootstrap_samples,
        seed=seed,
    )
    episode_winner_counts = np.bincount(episode_oracle, minlength=len(ARM_NAMES))
    sample_winner_counts = np.bincount(sample_oracle, minlength=len(ARM_NAMES))
    relative_improvement = -bootstrap_mean / fixed_means[best_fixed] * 100.0
    return {
        "source": str(path),
        "samples": int(errors.shape[0]),
        "episodes": int(episode_ids.size),
        "arm_names": list(ARM_NAMES),
        "fixed_episode_mean_mse": {name: float(value) for name, value in zip(ARM_NAMES, fixed_means)},
        "best_fixed_arm": ARM_NAMES[best_fixed],
        "episode_oracle_mean_mse": float(episode_oracle_means.mean()),
        "sample_oracle_mean_mse": sample_oracle_mean,
        "episode_oracle_minus_best_fixed": bootstrap_mean,
        "episode_oracle_minus_best_fixed_ci95": bootstrap_ci,
        "episode_oracle_relative_improvement_percent": float(relative_improvement),
        "episode_winner_fraction": {
            name: float(count / episode_ids.size) for name, count in zip(ARM_NAMES, episode_winner_counts)
        },
        "sample_winner_fraction": {
            name: float(count / errors.shape[0]) for name, count in zip(ARM_NAMES, sample_winner_counts)
        },
        "passes_actionable_oracle_screen": bool(relative_improvement >= 1.0 and bootstrap_ci[1] < 0.0),
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
