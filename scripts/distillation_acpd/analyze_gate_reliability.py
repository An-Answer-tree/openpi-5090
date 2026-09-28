"""Analyzes whether an oracle correction direction predicts gate usefulness."""

import argparse
import json
import pathlib

import numpy as np


def _episode_means(values: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    """Averages values within each episode in ascending episode order."""
    return np.stack([values[episode_index == episode].mean(axis=0) for episode in np.unique(episode_index)])


def _group_summary(delta: np.ndarray, cosine: np.ndarray, aligned: np.ndarray) -> dict[str, float | int]:
    """Summarizes gate benefit for one alignment group."""
    group_delta = delta[aligned]
    return {
        "episodes": int(aligned.sum()),
        "fraction": float(aligned.mean()),
        "mean_on_minus_off_mse": float(group_delta.mean()),
        "benefit_fraction": float((group_delta < 0).mean()),
        "mean_oracle_cosine": float(cosine[aligned].mean()),
    }


def analyze_reliability(errors: np.ndarray, metrics: np.ndarray, episode_index: np.ndarray) -> dict:
    """Returns sample and episode correlations for the oracle correction cosine."""
    delta = errors[:, 3] - errors[:, 0]
    magnitude_ratio = metrics[:, 0]
    oracle_cosine = metrics[:, 1]
    episode_delta = _episode_means(delta, episode_index)
    episode_magnitude = _episode_means(magnitude_ratio, episode_index)
    episode_cosine = _episode_means(oracle_cosine, episode_index)
    return {
        "samples": len(delta),
        "episodes": len(episode_delta),
        "sample_cosine_delta_correlation": float(np.corrcoef(oracle_cosine, delta)[0, 1]),
        "episode_cosine_delta_correlation": float(np.corrcoef(episode_cosine, episode_delta)[0, 1]),
        "sample_magnitude_delta_correlation": float(np.corrcoef(magnitude_ratio, delta)[0, 1]),
        "episode_magnitude_delta_correlation": float(np.corrcoef(episode_magnitude, episode_delta)[0, 1]),
        "episode_groups": {
            "oracle_aligned_cosine_ge_0": _group_summary(episode_delta, episode_cosine, episode_cosine >= 0),
            "oracle_misaligned_cosine_lt_0": _group_summary(episode_delta, episode_cosine, episode_cosine < 0),
        },
        "scope": "Oracle algebraic sanity check: cosine uses the true flow target and is unavailable at inference.",
    }


def main() -> None:
    """Reads existing gate probe arrays and writes a small JSON analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-10k", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-30k", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    results = {}
    for label, path in (("10k", args.checkpoint_10k), ("30k", args.checkpoint_30k)):
        with np.load(path) as data:
            results[label] = analyze_reliability(data["errors"], data["metrics"], data["episode_index"])
        results[label]["source"] = str(path)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
