"""Tests whether view-only gate features predict local injection benefit."""

import argparse
import json
import pathlib

import numpy as np

_OBSERVABLE_NAMES = (
    "view_delta_cosine",
    "view_delta_disagreement",
    "view_norm_imbalance",
)


def _episode_means(values: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    """Averages values within each episode in ascending episode order."""
    return np.stack([values[episode_index == episode].mean(axis=0) for episode in np.unique(episode_index)])


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Computes the probability that a positive score exceeds a negative score."""
    positive = scores[labels]
    negative = scores[~labels]
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _cross_validated_auc(feature: np.ndarray, labels: np.ndarray) -> list[float]:
    """Chooses score orientation on one episode fold and evaluates the other."""
    scores = []
    for held_out_parity in (0, 1):
        train = np.arange(len(feature)) % 2 != held_out_parity
        test = ~train
        train_difference = feature[train][labels[train]].mean() - feature[train][~labels[train]].mean()
        orientation = 1.0 if train_difference >= 0 else -1.0
        scores.append(_auc(orientation * feature[test], labels[test]))
    return scores


def _cross_validated_policy(
    feature: np.ndarray,
    episode_errors: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float]]:
    """Evaluates a two-fold threshold policy that selects injection or off."""
    results = []
    for held_out_parity in (0, 1):
        train = np.arange(len(feature)) % 2 != held_out_parity
        test = ~train
        train_feature = feature[train]
        train_labels = labels[train]
        orientation = 1.0 if train_feature[train_labels].mean() >= train_feature[~train_labels].mean() else -1.0
        train_score = orientation * train_feature
        threshold = 0.5 * (np.median(train_score[train_labels]) + np.median(train_score[~train_labels]))
        use_on = orientation * feature[test] >= threshold
        test_delta = episode_errors[test, 3] - episode_errors[test, 0]
        results.append(
            {
                "selected_on_fraction": float(use_on.mean()),
                "selected_minus_off_mse": float(np.mean(np.where(use_on, test_delta, 0.0))),
                "selected_minus_on_mse": float(np.mean(np.where(use_on, 0.0, -test_delta))),
            }
        )
    return results


def analyze_observables(errors: np.ndarray, metrics: np.ndarray, episode_index: np.ndarray) -> dict:
    """Reports held-out episode predictability of view-only observables."""
    episode_errors = _episode_means(errors, episode_index)
    episode_delta = episode_errors[:, 3] - episode_errors[:, 0]
    episode_metrics = _episode_means(metrics[:, 2:5], episode_index)
    labels = episode_delta < 0
    oracle_error = np.minimum(episode_errors[:, 0], episode_errors[:, 3])
    oracle_policy = {
        "selected_on_fraction": float((episode_errors[:, 3] < episode_errors[:, 0]).mean()),
        "mean_mse": float(oracle_error.mean()),
        "mean_mse_minus_off": float(oracle_error.mean() - episode_errors[:, 0].mean()),
        "mean_mse_minus_on": float(oracle_error.mean() - episode_errors[:, 3].mean()),
    }
    results = {}
    for index, name in enumerate(_OBSERVABLE_NAMES):
        feature = episode_metrics[:, index]
        results[name] = {
            "episode_pearson_correlation_with_on_minus_off_mse": float(np.corrcoef(feature, episode_delta)[0, 1]),
            "mean_observable_for_benefit_episodes": float(feature[labels].mean()),
            "mean_observable_for_harm_episodes": float(feature[~labels].mean()),
            "held_out_auc_per_fold": _cross_validated_auc(feature, labels),
            "held_out_policy_per_fold": _cross_validated_policy(feature, episode_errors, labels),
        }
    return {
        "episodes": len(episode_delta),
        "benefit_episodes": int(labels.sum()),
        "harm_episodes": int((~labels).sum()),
        "oracle_policy": oracle_policy,
        "observable_results": results,
        "scope": "Features use only predicted view branches; labels use offline true flow targets.",
    }


def main() -> None:
    """Reads one gate probe array and writes observable predictability results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with np.load(args.input) as data:
        result = analyze_observables(data["errors"], data["metrics"], data["episode_index"])
    result["source"] = str(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
