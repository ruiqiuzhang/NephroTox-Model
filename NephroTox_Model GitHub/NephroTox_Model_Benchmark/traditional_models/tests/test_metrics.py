import numpy as np

from nephrotox.metrics import calculate_metrics, expected_calibration_error


def test_perfect_predictions_have_expected_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.01, 0.10, 0.90, 0.99])
    metrics = calculate_metrics(y, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["mcc"] == 1.0


def test_calibration_error_is_bounded() -> None:
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.2, 0.8, 0.4, 0.6])
    value = expected_calibration_error(y, probability)
    assert 0.0 <= value <= 1.0

