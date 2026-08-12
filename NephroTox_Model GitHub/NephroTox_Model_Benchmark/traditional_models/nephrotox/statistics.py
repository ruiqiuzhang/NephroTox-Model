from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import calculate_metrics


CI_METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "brier",
    "ece_10bin",
]


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile 95% confidence intervals using class-stratified resampling."""
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for stratified bootstrap CIs.")
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in CI_METRICS}
    for _ in range(n_replicates):
        sampled = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        metrics = calculate_metrics(y[sampled], probability[sampled], threshold)
        for metric in CI_METRICS:
            samples[metric].append(float(metrics[metric]))
    point = calculate_metrics(y, probability, threshold)
    return {
        metric: {
            "estimate": float(point[metric]),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in samples.items()
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    metric: str = "roc_auc",
    threshold: float = 0.5,
    n_replicates: int = 2000,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Paired bootstrap comparison (A minus B) on identical test observations."""
    if metric not in CI_METRICS:
        raise ValueError(f"Unsupported comparison metric: {metric}")
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(probabilities_a, dtype=float)
    b = np.asarray(probabilities_b, dtype=float)
    if not (y.shape == a.shape == b.shape):
        raise ValueError("All paired arrays must have the same shape.")
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_replicates, dtype=float)
    for replicate in range(n_replicates):
        sampled = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        metric_a = calculate_metrics(y[sampled], a[sampled], threshold)[metric]
        metric_b = calculate_metrics(y[sampled], b[sampled], threshold)[metric]
        differences[replicate] = float(metric_a) - float(metric_b)
    observed = float(
        calculate_metrics(y, a, threshold)[metric]
        - calculate_metrics(y, b, threshold)[metric]
    )
    lower_tail = (np.count_nonzero(differences <= 0.0) + 1) / (n_replicates + 1)
    upper_tail = (np.count_nonzero(differences >= 0.0) + 1) / (n_replicates + 1)
    two_sided_p = float(2 * min(lower_tail, upper_tail))
    return {
        "metric": metric,
        "difference_a_minus_b": observed,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "p_value_two_sided": min(two_sided_p, 1.0),
        "n_replicates": int(n_replicates),
    }
