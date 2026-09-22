"""Compares paired gradient-conflict probe outputs."""

import dataclasses
import json
import pathlib

import numpy as np
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    """Command-line arguments."""

    early_rows: pathlib.Path
    late_rows: pathlib.Path
    output: pathlib.Path
    bootstrap_samples: int = 10_000
    seed: int = 42


def _load_rows(path: pathlib.Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _paired_interval(
    early: np.ndarray,
    late: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    indices = rng.integers(0, len(early), size=(samples, len(early)))
    early_samples = early[indices]
    late_samples = late[indices]
    if statistic == "mean":
        differences = late_samples.mean(axis=1) - early_samples.mean(axis=1)
    else:
        differences = np.median(late_samples, axis=1) - np.median(early_samples, axis=1)
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def analyze(args: Args) -> dict[str, object]:
    """Returns paired 5K-to-30K gradient changes."""
    early_rows = _load_rows(args.early_rows)
    late_rows = _load_rows(args.late_rows)
    early_batches = [row["batch"] for row in early_rows]
    late_batches = [row["batch"] for row in late_rows]
    if early_batches != late_batches:
        raise ValueError("Early and late probe rows must contain the same ordered batches.")

    rng = np.random.default_rng(args.seed)
    metrics = {}
    for component in ("contribution", "action_corr", "privileged"):
        conflict_early = np.asarray([row[f"{component}_conflict"] for row in early_rows])
        conflict_late = np.asarray([row[f"{component}_conflict"] for row in late_rows])
        cosine_early = np.asarray([row[f"{component}_cosine"] for row in early_rows])
        cosine_late = np.asarray([row[f"{component}_cosine"] for row in late_rows])
        norm_ratio_early = np.asarray([row[f"{component}_norm_ratio"] for row in early_rows])
        norm_ratio_late = np.asarray([row[f"{component}_norm_ratio"] for row in late_rows])
        metrics[component] = {
            "conflict_rate_early": float(conflict_early.mean()),
            "conflict_rate_late": float(conflict_late.mean()),
            "conflict_rate_delta": float(conflict_late.mean() - conflict_early.mean()),
            "conflict_rate_delta_ci95": _paired_interval(
                conflict_early, conflict_late, "mean", rng, args.bootstrap_samples
            ),
            "cosine_median_early": float(np.median(cosine_early)),
            "cosine_median_late": float(np.median(cosine_late)),
            "cosine_median_delta": float(np.median(cosine_late) - np.median(cosine_early)),
            "cosine_median_delta_ci95": _paired_interval(
                cosine_early, cosine_late, "median", rng, args.bootstrap_samples
            ),
            "norm_ratio_median_early": float(np.median(norm_ratio_early)),
            "norm_ratio_median_late": float(np.median(norm_ratio_late)),
        }

    privileged = metrics["privileged"]
    supports_conflict = privileged["conflict_rate_delta"] >= 0.10 and privileged["cosine_median_late"] < 0.0
    return {
        "early_rows": str(args.early_rows),
        "late_rows": str(args.late_rows),
        "num_paired_batches": len(early_rows),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "metrics": metrics,
        "supports_late_gradient_conflict": supports_conflict,
    }


def main(args: Args) -> None:
    """Writes the paired analysis as JSON."""
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
