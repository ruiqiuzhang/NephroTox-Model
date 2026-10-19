from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probability, edges[1:-1]), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        observed = float(np.mean(y[mask]))
        predicted = float(np.mean(probability[mask]))
        error += float(np.mean(mask)) * abs(observed - predicted)
    return float(error)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if y.shape != probability.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if not np.all(np.isfinite(probability)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "brier": float(brier_score_loss(y, probability)),
        "ece_10bin": expected_calibration_error(y, probability, n_bins=10),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def calibration_table(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.minimum(np.digitize(probability, edges[1:-1]), n_bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_id in range(n_bins):
        mask = ids == bin_id
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "n": int(mask.sum()),
                "mean_predicted_probability": (
                    float(probability[mask].mean()) if mask.any() else np.nan
                ),
                "observed_positive_fraction": (
                    float(y[mask].mean()) if mask.any() else np.nan
                ),
            }
        )
    return rows

