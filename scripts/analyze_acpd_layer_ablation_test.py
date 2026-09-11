import numpy as np

from scripts.analyze_acpd_layer_ablation import read_metrics
from scripts.analyze_acpd_layer_ablation import summarize


def test_read_metrics_and_summarize(tmp_path):
    log_path = tmp_path / "train.out"
    log_path.write_text(
        "Step 100: supervised_loss=0.0800, loss=0.3000, action_corr_loss=0.0200, "
        "acpd_prediction_loss=0.9000, acpd_variance_loss=0.8000\n"
        "Step 4000: supervised_loss=0.0400, loss=0.2000, action_corr_loss=0.0100, "
        "acpd_prediction_loss=0.5000, acpd_variance_loss=0.4000\n"
        "Step 4900: supervised_loss=0.0200, loss=0.1000, action_corr_loss=0.0050, "
        "acpd_prediction_loss=0.3000, acpd_variance_loss=0.2000\n"
    )

    metrics = read_metrics(log_path)
    summary = summarize("layers6_12", metrics)

    assert len(metrics) == 3
    assert summary["last_step"] == 4900
    np.testing.assert_allclose(summary["supervised_loss_mean_100_4900"], 0.0466666667)
    np.testing.assert_allclose(summary["supervised_loss_mean_4000_4900"], 0.03)
