"""Compares the two predicted-view injection branches in a frozen H9 probe."""

import argparse
import json
import pathlib

import numpy as np


def analyze_view_branches(
    errors: np.ndarray,
    episode_index: np.ndarray,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Returns episode-paired MSE contrasts for off, agent, wrist, and on."""
    episode_errors = np.stack([errors[episode_index == episode].mean(axis=0) for episode in np.unique(episode_index)])
    contrasts = {
        "wrist_minus_agent": episode_errors[:, 2] - episode_errors[:, 1],
        "on_minus_agent": episode_errors[:, 3] - episode_errors[:, 1],
        "interaction": episode_errors[:, 3] - episode_errors[:, 1] - episode_errors[:, 2] + episode_errors[:, 0],
    }
    sample_indices = np.random.default_rng(seed).integers(
        len(episode_errors), size=(bootstrap_samples, len(episode_errors))
    )
    return {
        "samples": len(errors),
        "episodes": len(episode_errors),
        "contrasts": {
            name: {
                "mean": float(values.mean()),
                "paired_ci95": np.percentile(values[sample_indices].mean(axis=1), [2.5, 97.5]).tolist(),
            }
            for name, values in contrasts.items()
        },
    }


def main() -> None:
    """Reads existing probe arrays and writes a small JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-10k", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-30k", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    results = {}
    for label, path in (("10k", args.checkpoint_10k), ("30k", args.checkpoint_30k)):
        with np.load(path) as data:
            results[label] = analyze_view_branches(data["errors"], data["episode_index"])
        results[label]["source"] = str(path)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
