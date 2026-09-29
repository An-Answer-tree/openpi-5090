"""Analyzes fixed ensembles of the frozen H18 action readout heads."""

import argparse
import json
from pathlib import Path

import numpy as np


ARM_NAMES = ("hidden_only", "hidden_layer10", "hidden_contribution")
BASELINE_INDEX = 1


def _readout_outputs(inputs: np.ndarray, velocity: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Returns corrected velocity for each H18 readout head."""
    preactivation = np.einsum("rbhd,rdk->rbhk", inputs, params["input_kernel"])
    preactivation += params["input_bias"][:, None, None]
    hidden = preactivation / (1.0 + np.exp(-preactivation))
    corrections = np.einsum("rbhk,rka->rbha", hidden, params["output_kernel"])
    corrections += params["output_bias"][:, None, None]
    return velocity[None] + corrections


def _episode_mse(outputs: np.ndarray, target: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    """Returns [episodes, outputs] mean flow MSE."""
    errors = np.mean(np.square(outputs - target[None]), axis=(2, 3)).T
    episodes = np.unique(episode_index)
    return np.stack([errors[episode_index == episode].mean(axis=0) for episode in episodes])


def _bootstrap_delta(delta: np.ndarray, *, samples: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    values = delta[indices].mean(axis=1)
    return float(values.mean()), np.percentile(values, [2.5, 97.5]).tolist()


def analyze(
    features_path: Path,
    readout_params_path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Evaluates fixed and two-fold selected readout ensembles."""
    with np.load(features_path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float64)
        velocity = np.asarray(data["velocity"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    with np.load(readout_params_path, allow_pickle=False) as data:
        params = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
    outputs = _readout_outputs(inputs, velocity, params)
    per_episode = _episode_mse(outputs, target, episode_index)
    fixed_half = 0.5 * per_episode[:, 0] + 0.5 * per_episode[:, 2]
    baseline = per_episode[:, BASELINE_INDEX]
    fixed_delta, fixed_ci = _bootstrap_delta(
        fixed_half - baseline,
        samples=bootstrap_samples,
        seed=seed,
    )

    episodes = np.unique(episode_index)
    selected_alpha = np.empty(episodes.size, dtype=np.float64)
    selected_error = np.empty(episodes.size, dtype=np.float64)
    alpha_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    for parity in (0, 1):
        test_mask = episodes % 2 == parity
        train_mask = ~test_mask
        train_episode_indices = np.flatnonzero(train_mask)
        test_episode_indices = np.flatnonzero(test_mask)
        train_scores = np.stack(
            [alpha * per_episode[train_episode_indices, 1] + (1 - alpha) * per_episode[train_episode_indices, 2] for alpha in alpha_grid],
            axis=1,
        )
        best_alpha = alpha_grid[np.argmin(train_scores.mean(axis=0))]
        selected_alpha[test_episode_indices] = best_alpha
        selected_error[test_episode_indices] = (
            best_alpha * per_episode[test_episode_indices, 1]
            + (1 - best_alpha) * per_episode[test_episode_indices, 2]
        )
    selected_delta, selected_ci = _bootstrap_delta(
        selected_error - baseline,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "features": str(features_path),
        "readout_params": str(readout_params_path),
        "samples": int(inputs.shape[1]),
        "episodes": int(episodes.size),
        "fixed_baseline": "hidden_layer10",
        "fixed_half_mix": {"hidden_only": 0.5, "hidden_contribution": 0.5},
        "fixed_half_mix_minus_baseline": fixed_delta,
        "fixed_half_mix_minus_baseline_ci95": fixed_ci,
        "fixed_half_mix_relative_change_percent": float(fixed_delta / baseline.mean() * 100.0),
        "cross_validated_alpha_grid": alpha_grid.tolist(),
        "cross_validated_alpha_counts": {
            str(alpha): int(np.sum(selected_alpha == alpha)) for alpha in alpha_grid
        },
        "cross_validated_mix_minus_baseline": selected_delta,
        "cross_validated_mix_minus_baseline_ci95": selected_ci,
        "cross_validated_mix_relative_change_percent": float(selected_delta / baseline.mean() * 100.0),
        "passes_fixed_mix_screen": bool(-fixed_delta / baseline.mean() * 100.0 >= 1.0 and fixed_ci[1] < 0.0),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--readout-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(
        args.features,
        args.readout_params,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
