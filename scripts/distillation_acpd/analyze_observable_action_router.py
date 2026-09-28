"""Evaluates a target-free router over frozen H18 action readouts."""

import argparse
import json
from pathlib import Path

import numpy as np


ARM_NAMES = ("frozen_h9", "hidden_only", "hidden_layer10", "hidden_contribution")
BASELINE_ARM = 2


def _summarize_inputs(inputs: np.ndarray) -> np.ndarray:
    """Pools visible action features into a compact deterministic representation."""
    if inputs.ndim != 4 or inputs.shape[0] != 3 or inputs.shape[-1] % 96:
        raise ValueError(f"Expected inputs [3, N, H, D] with D divisible by 96, got {inputs.shape}.")
    pooled = inputs.mean(axis=2).reshape(3, inputs.shape[1], 96, inputs.shape[-1] // 96).mean(axis=-1)
    return np.transpose(pooled, (1, 0, 2)).reshape(inputs.shape[1], -1)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    """Fits a small multi-output ridge model and predicts test arm errors."""
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_x = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    test_x = np.concatenate([test_x, np.ones((test_x.shape[0], 1))], axis=1)
    gram = train_x.T @ train_x
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.solve(gram, train_x.T @ train_y)
    return test_x @ weights


def _episode_bootstrap(delta: np.ndarray, *, samples: int, seed: int) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    values = delta[indices].mean(axis=1)
    return float(values.mean()), np.percentile(values, [2.5, 97.5]).tolist()


def analyze(features_path: Path, errors_path: Path, *, ridge: float, bootstrap_samples: int, seed: int) -> dict:
    """Runs two-fold episode-held-out target-free routing."""
    with np.load(features_path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float64)
        episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    with np.load(errors_path, allow_pickle=False) as data:
        errors = np.asarray(data["errors"], dtype=np.float64)
    if inputs.shape[1] != errors.shape[0] or episode_index.size != errors.shape[0]:
        raise ValueError("Feature and error files do not contain the same samples.")
    features = _summarize_inputs(inputs)
    selected = np.empty(errors.shape[0], dtype=np.int64)
    fold_rows = []
    for held_out_parity in (0, 1):
        test_mask = episode_index % 2 == held_out_parity
        train_mask = ~test_mask
        predictions = _fit_ridge(features[train_mask], errors[train_mask], features[test_mask], ridge)
        selected[test_mask] = np.argmin(predictions, axis=1)
        train_best = int(np.argmin(errors[train_mask].mean(axis=0)))
        fold_rows.append(
            {
                "held_out_parity": held_out_parity,
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "train_best_fixed_arm": ARM_NAMES[train_best],
            }
        )
    routed_error = errors[np.arange(errors.shape[0]), selected]
    baseline_error = errors[:, BASELINE_ARM]
    episode_ids = np.unique(episode_index)
    episode_delta = np.asarray(
        [
            routed_error[episode_index == episode_id].mean()
            - baseline_error[episode_index == episode_id].mean()
            for episode_id in episode_ids
        ]
    )
    delta, ci = _episode_bootstrap(episode_delta, samples=bootstrap_samples, seed=seed)
    selection_counts = np.bincount(selected, minlength=len(ARM_NAMES))
    return {
        "features": str(features_path),
        "errors": str(errors_path),
        "samples": int(errors.shape[0]),
        "episodes": int(episode_ids.size),
        "arm_names": list(ARM_NAMES),
        "baseline_arm": ARM_NAMES[BASELINE_ARM],
        "routed_mean_mse": float(routed_error.mean()),
        "baseline_mean_mse": float(baseline_error.mean()),
        "routed_minus_baseline": delta,
        "routed_minus_baseline_ci95": ci,
        "relative_improvement_percent": float(-delta / baseline_error.mean() * 100.0),
        "selection_fraction": {
            name: float(count / errors.shape[0]) for name, count in zip(ARM_NAMES, selection_counts)
        },
        "folds": fold_rows,
        "ridge": ridge,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "passes_actionable_screen": bool(-delta / baseline_error.mean() * 100.0 >= 1.0 and ci[1] < 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(
        args.features,
        args.errors,
        ridge=args.ridge,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
