import json

from scripts.distillation_acpd.analyze_gradient_conflict import analyze
from scripts.distillation_acpd.analyze_gradient_conflict import Args


def test_analyze_selects_conflict_aware_acpd(tmp_path):
    early_path = tmp_path / "early.jsonl"
    late_path = tmp_path / "late.jsonl"
    early_rows = []
    late_rows = []
    for batch in range(10):
        early_row = {"batch": batch}
        late_row = {"batch": batch}
        for component in ("contribution", "action_corr", "privileged"):
            early_row.update(
                {
                    f"{component}_conflict": 0.0,
                    f"{component}_cosine": 0.2,
                    f"{component}_norm_ratio": 1.0,
                }
            )
            late_row.update(
                {
                    f"{component}_conflict": 1.0,
                    f"{component}_cosine": -0.2,
                    f"{component}_norm_ratio": 1.5,
                }
            )
        early_rows.append(early_row)
        late_rows.append(late_row)
    early_path.write_text("".join(json.dumps(row) + "\n" for row in early_rows))
    late_path.write_text("".join(json.dumps(row) + "\n" for row in late_rows))

    result = analyze(
        Args(
            early_rows=early_path,
            late_rows=late_path,
            output=tmp_path / "analysis.json",
            bootstrap_samples=100,
        )
    )

    assert result["supports_late_gradient_conflict"] is True
    assert result["next_experiment"] == "conflict_aware_acpd"
    assert result["metrics"]["privileged"]["conflict_rate_delta"] == 1.0
