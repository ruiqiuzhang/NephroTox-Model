"""Locked-model evaluation from current DATA and saved, aligned predictions.

Test labels are used only after the complete, hash-verified global model lock.
The primary threshold is 0.5. Reference-negative is a dataset label, not proof
of clinical safety. All intervals are conditional on the fitted predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data import ROOT, rows_fingerprint

METRICS = ("ACC", "SE", "SP", "Precision", "F1", "MCC", "BA", "ROC_AUC", "AP", "Brier", "LogLoss")
COUNTS = ("TN", "FP", "FN", "TP")
HEATMAP_METRICS = ("ROC_AUC", "ACC", "Precision", "SE", "SP", "F1", "MCC", "BA")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path, records):
    records = list(records)
    if not records:
        raise ValueError(f"Cannot write an empty report: {Path(path).name}")
    columns = list(dict.fromkeys(k for record in records for k in record))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _validated(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be binary 0/1")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return y.astype(np.int8), probability


def _single_metrics(y, p, threshold):
    positive, predicted = y == 1, p >= threshold
    tp = int(np.count_nonzero(positive & predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    tn = int(np.count_nonzero(~positive & ~predicted))
    safe = lambda a, b: float(a / b) if b else 0.0
    sensitivity, specificity = safe(tp, tp + fn), safe(tn, tn + fp)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    auc = float((rankdata(p)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    # AP integrates recall increments at distinct score thresholds. Grouping
    # tied scores matches average_precision_score (not trapezoidal PR AUC).
    order = np.argsort(-p, kind="stable")
    distinct = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    ctp = np.cumsum(y[order], dtype=float)[distinct]
    precision = ctp / (distinct + 1)
    ap = float(np.sum(np.diff(np.r_[0.0, ctp]) * precision) / npos) if npos else 0.0
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "ACC": safe(tp + tn, len(y)), "SE": sensitivity, "SP": specificity,
        "Precision": safe(tp, tp + fp), "F1": safe(2 * tp, 2 * tp + fp + fn),
        "MCC": safe(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
        "BA": (sensitivity + specificity) / 2, "ROC_AUC": auc, "AP": ap,
        "Brier": float(np.mean((p - y) ** 2)),
        "LogLoss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def metric_values(y, probability, threshold=0.5):
    """Compute binary metrics from the same saved probabilities and threshold.

    Undefined single-class ROC-AUC is None; zero-denominator classification
    metrics use 0.0. This makes bootstrap/CSV behavior explicit and reproducible.
    """
    y, probability = _validated(y, probability)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must lie in [0, 1]")
    return _single_metrics(y, probability, threshold)


def _stratified_indices(y, n_bootstrap, seed):
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("At least two bootstrap replicates are required")
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(positive) or not len(negative):
        raise ValueError("Stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.choice(negative, (n_bootstrap, len(negative)), replace=True),
                           rng.choice(positive, (n_bootstrap, len(positive)), replace=True)), axis=1)


def _bootstrap_auc(y, p, indices):
    sampled_y, sampled_p = y[indices], p[indices]
    npos = sampled_y.sum(axis=1)
    nneg = sampled_y.shape[1] - npos
    ranks = rankdata(sampled_p, axis=1)
    return ((ranks * sampled_y).sum(axis=1) - npos * (npos + 1) / 2) / (npos * nneg)


def _bootstrap_distributions(y, p, indices, threshold):
    """Vectorized metrics in bounded batches, sharing each score sort for AUC/AP."""
    result = {name: np.empty(len(indices), dtype=float) for name in METRICS}
    for start in range(0, len(indices), 128):
        index = indices[start:start + 128]
        sy, sp = y[index], p[index]
        positive, predicted = sy.astype(bool), sp >= threshold
        tp = np.sum(positive & predicted, axis=1).astype(float)
        fp = np.sum(~positive & predicted, axis=1).astype(float)
        fn = np.sum(positive & ~predicted, axis=1).astype(float)
        tn = np.sum(~positive & ~predicted, axis=1).astype(float)
        divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        se, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
        order = np.argsort(-sp, axis=1, kind="stable")
        sorted_y, sorted_p = np.take_along_axis(sy, order, 1), np.take_along_axis(sp, order, 1)
        cumulative_tp = np.cumsum(sorted_y, axis=1, dtype=float)
        positions = np.arange(1, sy.shape[1] + 1)
        cumulative_fp = positions - cumulative_tp
        boundary = np.concatenate((np.diff(sorted_p, axis=1) != 0, np.ones((len(index), 1), dtype=bool)), axis=1)
        zeros = np.zeros((len(index), 1))
        previous_tp = np.concatenate((zeros, np.maximum.accumulate(np.where(boundary, cumulative_tp, 0), axis=1)[:, :-1]), axis=1)
        previous_fp = np.concatenate((zeros, np.maximum.accumulate(np.where(boundary, cumulative_fp, 0), axis=1)[:, :-1]), axis=1)
        npos, nneg = sy.sum(axis=1), sy.shape[1] - sy.sum(axis=1)
        auc = np.sum(np.where(boundary, (cumulative_fp - previous_fp) * (cumulative_tp + previous_tp), 0), axis=1) / (2 * npos * nneg)
        ap = np.sum(np.where(boundary, (cumulative_tp - previous_tp) * cumulative_tp / positions, 0), axis=1) / npos
        clipped = np.clip(sp, np.finfo(float).eps, 1 - np.finfo(float).eps)
        values = {"ACC": (tp + tn) / sy.shape[1], "SE": se, "SP": specificity,
                  "Precision": divide(tp, tp + fp), "F1": divide(2 * tp, 2 * tp + fp + fn),
                  "MCC": divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
                  "BA": (se + specificity) / 2, "ROC_AUC": auc, "AP": ap,
                  "Brier": np.mean((sp - sy) ** 2, axis=1),
                  "LogLoss": -np.mean(sy * np.log(clipped) + (1 - sy) * np.log1p(-clipped), axis=1)}
        for name in METRICS:
            result[name][start:start + len(index)] = values[name]
    return result


def _ci_from_indices(y, p, indices, seed, threshold=0.5, confidence=0.95, metric_keys=METRICS):
    point = metric_values(y, p, threshold)
    samples = _bootstrap_distributions(y, p, indices, threshold)
    alpha = (1 - confidence) / 2
    return {name: {"estimate": point[name],
                    "lower": float(np.quantile(samples[name], alpha)),
                    "upper": float(np.quantile(samples[name], 1 - alpha)),
                    "n_bootstrap": len(indices), "seed": int(seed),
                    "method": "class-stratified percentile bootstrap", "confidence": confidence}
            for name in metric_keys}


def stratified_bootstrap_ci(y, probability, n_bootstrap=2000, seed=20260810,
                            threshold=0.5, confidence=0.95, metric_keys=METRICS):
    """Percentile CI, sampling with replacement separately within label strata.

    CIs describe observation resampling conditional on current trained models;
    they do not include retraining or group-cluster uncertainty.
    """
    y, p = _validated(y, probability)
    if not 0 < confidence < 1 or not set(metric_keys).issubset(METRICS):
        raise ValueError("Invalid confidence level or metric names")
    indices = _stratified_indices(y, n_bootstrap, seed)
    return _ci_from_indices(y, p, indices, seed, threshold, confidence, metric_keys)


def _group_bootstrap_weights(groups, n_bootstrap, seed):
    """Each sampled group contributes every original member with multiplicity."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or any(v is None or str(v).strip() in ("", "nan", "None") for v in groups):
        raise ValueError("Structure groups must be a nonempty one-dimensional array without missing IDs")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("At least two group bootstrap replicates are required")
    names, inverse = np.unique(groups.astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(n_bootstrap, len(names)))
    multiplicity = np.asarray([np.bincount(draw, minlength=len(names)) for draw in draws], dtype=np.int32)
    return multiplicity[:, inverse], names, multiplicity


def _weighted_bootstrap_distributions(y, p, weights, threshold=0.5):
    """Exact duplicate-row bootstrap metrics through integer sample weights.

    Score sorting is shared by all replicates, including weighted ties. This
    avoids expanding large resampled structure groups into long row arrays.
    """
    result = {name: np.empty(len(weights), dtype=float) for name in METRICS}
    order = np.argsort(-p, kind="stable")
    boundary = np.r_[np.diff(p[order]) != 0, True]
    positive, predicted = y == 1, p >= threshold
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    sample_loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    for start in range(0, len(weights), 128):
        weight = weights[start:start + 128].astype(float)
        n = weight.sum(axis=1)
        tp, fp = weight[:, positive & predicted].sum(axis=1), weight[:, ~positive & predicted].sum(axis=1)
        fn, tn = weight[:, positive & ~predicted].sum(axis=1), weight[:, ~positive & ~predicted].sum(axis=1)
        divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        se, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
        sorted_weight = weight[:, order]
        ctp = np.cumsum(sorted_weight * y[order], axis=1)
        cfp = np.cumsum(sorted_weight * (1 - y[order]), axis=1)
        zeros = np.zeros((len(weight), 1))
        previous_tp = np.concatenate((zeros, np.maximum.accumulate(np.where(boundary, ctp, 0), axis=1)[:, :-1]), axis=1)
        previous_fp = np.concatenate((zeros, np.maximum.accumulate(np.where(boundary, cfp, 0), axis=1)[:, :-1]), axis=1)
        total_positive, total_negative = tp + fn, tn + fp
        auc = np.sum(np.where(boundary, (cfp - previous_fp) * (ctp + previous_tp), 0), axis=1) / (2 * total_positive * total_negative)
        cumulative_precision = np.divide(ctp, ctp + cfp, out=np.zeros_like(ctp), where=(ctp + cfp) != 0)
        ap = np.sum(np.where(boundary, (ctp - previous_tp) * cumulative_precision, 0), axis=1) / total_positive
        values = {"ACC": (tp + tn) / n, "SE": se, "SP": specificity,
                  "Precision": divide(tp, tp + fp), "F1": divide(2 * tp, 2 * tp + fp + fn),
                  "MCC": divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
                  "BA": (se + specificity) / 2, "ROC_AUC": auc, "AP": ap,
                  "Brier": weight @ ((p - y) ** 2) / n, "LogLoss": weight @ sample_loss / n}
        for name in METRICS:
            result[name][start:start + len(weight)] = values[name]
    return result


def _group_ci_from_weights(y, p, weights, groups, seed, threshold=0.5, confidence=0.95, metric_keys=METRICS):
    point = metric_values(y, p, threshold)
    valid = ((weights[:, y == 1].sum(axis=1) > 0) & (weights[:, y == 0].sum(axis=1) > 0))
    if int(valid.sum()) < 2:
        raise ValueError("Fewer than two group bootstrap draws contain both classes")
    samples = _weighted_bootstrap_distributions(y, p, weights[valid], threshold)
    alpha = (1 - confidence) / 2
    return {name: {"estimate": point[name], "lower": float(np.quantile(samples[name], alpha)),
                    "upper": float(np.quantile(samples[name], 1 - alpha)), "n_bootstrap_requested": len(weights),
                    "n_bootstrap_valid": int(valid.sum()), "n_bootstrap_single_class_discarded": int((~valid).sum()),
                    "n_structure_groups": len(set(groups)), "seed": int(seed), "confidence": confidence,
                    "method": "structure-group percentile bootstrap; each sampled group contributes all members; single-class draws discarded; no replacement draws"}
            for name in metric_keys}


def structure_group_bootstrap_ci(y, probability, groups, n_bootstrap=2000, seed=20260810,
                                  threshold=0.5, confidence=0.95, metric_keys=METRICS):
    """Cluster-bootstrap sensitivity CI conditional on saved fitted predictions."""
    y, p = _validated(y, probability)
    groups = np.asarray(groups)
    if groups.shape != y.shape or not 0 < confidence < 1 or not set(metric_keys).issubset(METRICS):
        raise ValueError("Invalid structure groups, confidence level or metric names")
    weights, _, _ = _group_bootstrap_weights(groups, n_bootstrap, seed)
    return _group_ci_from_weights(y, p, weights, groups, seed, threshold, confidence, metric_keys)


def holm_adjust(p_values):
    """Familywise Holm step-down adjustment, preserving the input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P values must lie in [0, 1]")
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(order), dtype=float)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(order) - np.arange(len(order))) * p_values[order]))
    return adjusted


def paired_auc_comparison(y, probabilities, pairs=None, n_bootstrap=2000, seed=20260810):
    """Paired AUC differences with shared stratified resamples and Holm tests.

    The two-sided bootstrap test uses the centered null distribution and the
    plus-one correction. Pair selection must be fixed without test outcomes.
    """
    if len(probabilities) < 2:
        return []
    names = list(probabilities)
    y, first = _validated(y, probabilities[names[0]])
    matrix = {names[0]: first}
    for name in names[1:]:
        _, matrix[name] = _validated(y, probabilities[name])
    pairs = list(combinations(names, 2)) if pairs is None else list(pairs)
    if len(set(pairs)) != len(pairs) or any(a == b or a not in matrix or b not in matrix for a, b in pairs):
        raise ValueError("Paired AUC comparisons contain invalid/duplicate model pairs")
    indices = _stratified_indices(y, n_bootstrap, seed)
    needed = set(n for pair in pairs for n in pair)
    distributions = {name: _bootstrap_auc(y, matrix[name], indices) for name in needed}
    points = {name: metric_values(y, matrix[name])["ROC_AUC"] for name in needed}
    records = []
    for first_name, second_name in pairs:
        point = points[first_name] - points[second_name]
        delta = distributions[first_name] - distributions[second_name]
        p_value = (1 + int(np.count_nonzero(np.abs(delta - point) >= abs(point) - 1e-15))) / (n_bootstrap + 1)
        records.append({"model_1": first_name, "model_2": second_name, "metric": "ROC_AUC",
                        "difference": float(point), "lower": float(np.quantile(delta, 0.025)),
                        "upper": float(np.quantile(delta, 0.975)), "p_bootstrap": p_value,
                        "n_bootstrap": n_bootstrap, "seed": seed,
                        "method": "paired class-stratified centered bootstrap test; percentile 95% CI"})
    for record, p_value in zip(records, holm_adjust([r["p_bootstrap"] for r in records])):
        record.update(p_holm=float(p_value), significant_holm_005=bool(p_value < 0.05),
                      comparison_family_size=len(records))
    return records


def calibration_bins(y, probability, n_bins=10):
    """Equal-width probability bins; empty bins remain visible with count zero."""
    y, p = _validated(y, probability)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("At least two calibration bins are required")
    assignments = np.minimum((p * n_bins).astype(int), n_bins - 1)
    result = []
    for i in range(n_bins):
        mask = assignments == i
        result.append({"bin": i + 1, "lower": i / n_bins, "upper": (i + 1) / n_bins,
                       "n": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else None,
                       "positive_fraction": float(y[mask].mean()) if mask.any() else None})
    return result


def verify_model_lock(results_dir, rows, expected_model_ids=None, lock_path=None):
    """Verify the global lock before any test-label extraction or statistics."""
    folder = Path(results_dir)
    lock_path = Path(lock_path) if lock_path else folder / "model_lock.json"
    if not lock_path.is_file():
        raise ValueError("Test evaluation is forbidden before the complete global model lock exists")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("test_evaluation_authorized") is not True:
        raise ValueError("Global lock does not authorize test evaluation")
    if not lock.get("locked_at_utc") or not lock.get("protocol_sha256"):
        raise ValueError("Global model lock is incomplete")
    if not (folder / "protocol.json").is_file() or _digest(folder / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Frozen modeling protocol changed or is missing after global locking")
    if lock.get("source_rows_fingerprint") != rows_fingerprint(rows):
        raise ValueError("DATA fingerprint differs from the global model lock")
    source_manifest = json.loads((ROOT / "data/source_manifest.json").read_text(encoding="utf-8"))
    if lock.get("source_excel_sha256") != source_manifest["sha256"]:
        raise ValueError("DATA workbook SHA differs from the global model lock")
    model_ids = lock.get("expected_model_ids")
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("Global model list must contain unique IDs")
    if expected_model_ids is not None and set(model_ids) != set(expected_model_ids):
        raise ValueError("Global lock does not contain the prespecified model set")
    entries = lock.get("models", [])
    if len(entries) != len(model_ids) or {m.get("model_id") for m in entries} != set(model_ids):
        raise ValueError("Not every prespecified model has been locked")
    metadata = {}
    for entry in entries:
        model_id = entry["model_id"]
        if Path(model_id).name != model_id or model_id in (".", ".."):
            raise ValueError("Invalid model ID in global lock")
        for filename, key in (("metadata.json", "metadata_sha256"), ("predictions.npz", "predictions_sha256")):
            if not (folder / model_id / filename).is_file() or _digest(folder / model_id / filename) != entry.get(key):
                raise ValueError(f"Locked {filename} changed or is missing for {model_id}")
        if not entry.get("model_sha256"):
            raise ValueError(f"Trained-model digest is missing for {model_id}")
        meta = json.loads((folder / model_id / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("complete") is not True or meta.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError(f"Incomplete or inconsistent locked metadata for {model_id}")
        model_file = meta.get("model_file")
        if not isinstance(model_file, str) or Path(model_file).name != model_file:
            raise ValueError(f"Missing/invalid trained-model artifact name for {model_id}")
        if not (folder / model_id / model_file).is_file() or _digest(folder / model_id / model_file) != entry["model_sha256"]:
            raise ValueError(f"Locked trained-model artifact changed or is missing for {model_id}")
        if meta.get("model_sha256") != entry["model_sha256"]:
            raise ValueError(f"Trained-model metadata digest is inconsistent for {model_id}")
        metadata[model_id] = meta
    return lock, metadata


def load_locked_predictions(results_dir, rows, assignments, expected_model_ids=None, lock_path=None):
    """Validate the frozen IDs, split indices, probabilities, and model hashes."""
    lock, metadata = verify_model_lock(results_dir, rows, expected_model_ids, lock_path)
    if isinstance(assignments, dict):
        assignments = assignments["assignments"]
    if len(assignments) != len(rows) or [r["compound_id"] for r in assignments] != [r["compound_id"] for r in rows]:
        raise ValueError("Split assignments do not match current DATA row order")
    ids = np.asarray([r["compound_id"] for r in rows], dtype=str)
    dev = np.asarray([i for i, r in enumerate(assignments) if r["dataset_partition"] == "development"], dtype=int)
    test = np.asarray([i for i, r in enumerate(assignments) if r["dataset_partition"] == "test"], dtype=int)
    if len(dev) + len(test) != len(rows) or set(dev) & set(test):
        raise ValueError("Assignments do not form disjoint development and test sets")
    folds = np.asarray([int(assignments[i]["cv_validation_fold"]) for i in dev])
    if set(folds) != set(range(1, 11)):
        raise ValueError("Expected the fixed development 10-fold assignment")
    groups = np.asarray([r["structure_group_id"] for r in assignments], dtype=str)
    if set(groups[dev]) & set(groups[test]):
        raise ValueError("Structure-group leakage between development and test")
    for fold in range(1, 11):
        if set(groups[dev[folds == fold]]) & set(groups[dev[folds != fold]]):
            raise ValueError("Structure-group leakage in the development folds")
    predictions = {}
    for model_id in lock["expected_model_ids"]:
        with np.load(Path(results_dir) / model_id / "predictions.npz", allow_pickle=False) as payload:
            required = {"compound_id", "development_index", "test_index", "oof_probability", "test_probability"}
            if not required.issubset(payload.files):
                raise ValueError(f"Incomplete prediction payload for {model_id}")
            if not np.array_equal(payload["compound_id"].astype(str), ids):
                raise ValueError(f"Prediction compound IDs differ from DATA for {model_id}")
            if not np.array_equal(payload["development_index"], dev) or not np.array_equal(payload["test_index"], test):
                raise ValueError(f"Prediction split indices differ from fixed assignments for {model_id}")
            oof, holdout = payload["oof_probability"].astype(float), payload["test_probability"].astype(float)
        # Shape/range validation does not require any test labels.
        for p, index in ((oof, dev), (holdout, test)):
            if p.shape != (len(index),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f"Invalid aligned probabilities for {model_id}")
        predictions[model_id] = {"development": oof, "test": holdout}
    return {"lock": lock, "metadata": metadata, "predictions": predictions,
            "development_index": dev, "test_index": test, "folds": folds, "groups": groups,
            "assignments": assignments}


def report_predictions(results_dir, rows, assignments, *, n_bootstrap=2000, seed=20260810,
                       expected_model_ids=None, lock_path=None):
    """Create complete prediction/metric/source tables from locked current models.

    The conventional primary is selected by development fold-mean ROC-AUC,
    with model ID as the deterministic tie-breaker. The prespecified AUC test
    family compares this primary against other main models and compares all
    three graph pairs, deduplicating identical pairs. Count models are a
    sensitivity analysis and cannot become the primary retrospectively.
    """
    bundle = load_locked_predictions(results_dir, rows, assignments, expected_model_ids, lock_path)
    folder = Path(results_dir) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([r["label"] for r in rows], dtype=int)  # only after lock validation
    metadata, predictions = bundle["metadata"], bundle["predictions"]
    dev, test, folds = bundle["development_index"], bundle["test_index"], bundle["folds"]
    summaries, per_fold, calibration, runtime, prediction_records = [], [], [], [], {"development": [], "test": []}
    bootstrap_index = {name: _stratified_indices(labels[index], n_bootstrap, seed)
                       for name, index in (("development", dev), ("test", test))}
    for model_id in bundle["lock"]["expected_model_ids"]:
        meta = metadata[model_id]
        runtime.append({"model_id": model_id, "algorithm": meta["algorithm"], "representation": meta["representation"],
                        "role": meta["role"], "model_bytes": meta.get("model_bytes"),
                        "final_fit_seconds": meta.get("final_fit_seconds"),
                        "single_test_inference_seconds": meta.get("test_inference_seconds"),
                        "single_test_inference_ms_per_compound": 1000 * meta["test_inference_seconds"] / len(test) if "test_inference_seconds" in meta else None,
                        "current_process_elapsed_seconds": meta.get("total_elapsed_seconds"),
                        "training_threads": meta.get("training_threads"),
                        "timing_scope": "actual measured wall times; a single held-out inference pass; current-process elapsed may exclude resumed work"})
        for split_name, index in (("development", dev), ("test", test)):
            probability = predictions[model_id][split_name]
            ci = _ci_from_indices(labels[index], probability, bootstrap_index[split_name], seed)
            values = metric_values(labels[index], probability)
            record = {"model_id": model_id, "algorithm": meta["algorithm"], "representation": meta["representation"],
                      "role": meta["role"], "partition": "development_oof" if split_name == "development" else "test",
                      "n": len(index), "positive": int(labels[index].sum()), "reference_negative": int(len(index) - labels[index].sum()),
                      "threshold": 0.5, **values}
            for name, interval in ci.items():
                record[name + "_lower"] = interval["lower"]
                record[name + "_upper"] = interval["upper"]
            summaries.append(record)
            for global_index, p in zip(index, probability):
                row, split = rows[global_index], bundle["assignments"][global_index]
                prediction_records[split_name].append({"model_id": model_id, "algorithm": meta["algorithm"],
                    "representation": meta["representation"], "role": meta["role"], "compound_id": row["compound_id"],
                    "name": row["name"], "PubChem_CID": row["PubChem_CID"], "label": row["label"],
                    "probability": float(p), "predicted_label": int(p >= 0.5), "threshold": 0.5,
                    "structure_group_id": split["structure_group_id"], "cv_validation_fold": split["cv_validation_fold"]})
            for item in calibration_bins(labels[index], probability):
                calibration.append({"model_id": model_id, "partition": record["partition"], **item})
        for fold in range(1, 11):
            mask = folds == fold
            per_fold.append({"model_id": model_id, "algorithm": meta["algorithm"], "representation": meta["representation"],
                             "role": meta["role"], "fold": fold, "n": int(mask.sum()),
                             "n_structure_groups": len(set(bundle["groups"][dev[mask]])),
                             **metric_values(labels[dev[mask]], predictions[model_id]["development"][mask])})
    mean_auc = {name: float(np.mean([r["ROC_AUC"] for r in per_fold if r["model_id"] == name])) for name in metadata}
    conventional = [name for name, m in metadata.items() if m["role"] == "main" and m["representation"] != "graph"]
    if not conventional:
        raise ValueError("No main conventional model is available for prespecified primary selection")
    primary = min(conventional, key=lambda name: (-mean_auc[name], name))
    main_models = [name for name, m in metadata.items() if m["role"] == "main"]
    graph_models = [name for name in main_models if metadata[name]["representation"] == "graph"]
    group_ci = []
    group_draws = {}
    for partition, index in (("development_oof", dev), ("test", test)):
        groups = bundle["groups"][index]
        weights, group_names, multiplicity = _group_bootstrap_weights(groups, n_bootstrap, seed)
        group_draws[partition + "_group_id"] = group_names
        group_draws[partition + "_multiplicity"] = multiplicity
        for model_id in [primary] + graph_models:
            probabilities = predictions[model_id]["development" if partition == "development_oof" else "test"]
            ci = _group_ci_from_weights(labels[index], probabilities, weights, groups, seed)
            for name, interval in ci.items():
                group_ci.append({"model_id": model_id, "partition": partition, "metric": name,
                                 "primary_selected_by_development_only": model_id == primary, **interval})
    pairs = [(primary, name) for name in main_models if name != primary]
    pairs += list(combinations(graph_models, 2))
    paired = paired_auc_comparison(labels[test], {name: predictions[name]["test"] for name in main_models},
                                   pairs=pairs, n_bootstrap=n_bootstrap, seed=seed)
    for record in summaries:
        record["development_fold_mean_ROC_AUC"] = mean_auc[record["model_id"]]
        record["development_fold_sd_ROC_AUC"] = float(np.std([r["ROC_AUC"] for r in per_fold if r["model_id"] == record["model_id"]], ddof=1))
        record["primary_selected_by_development_only"] = record["model_id"] == primary
    _write_csv(folder / "oof_predictions.csv", prediction_records["development"])
    _write_csv(folder / "test_predictions.csv", prediction_records["test"])
    _write_csv(folder / "perfold_metrics.csv", per_fold)
    _write_csv(folder / "model_summary.csv", summaries)
    _write_csv(folder / "calibration_bins.csv", calibration)
    _write_csv(folder / "runtime_summary.csv", runtime)
    _write_csv(folder / "group_bootstrap_ci.csv", group_ci)
    np.savez_compressed(folder / "group_bootstrap_draws.npz", **group_draws)
    if paired:
        _write_csv(folder / "paired_auc.csv", paired)
    # Model selection audit uses only the development results above.
    _write_json(folder / "primary_selection.json", {"primary_model_id": primary, "selection_population": "development only",
        "selection_metric": "mean of the 10 outer-validation ROC-AUC values", "tie_breaker": "model_id ascending",
        "candidate_model_ids": conventional, "mean_outer_auc": {n: mean_auc[n] for n in conventional},
        "test_used_for_selection": False})
    manifest = {"schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows_fingerprint": bundle["lock"]["source_rows_fingerprint"],
        "source_excel_sha256": bundle["lock"]["source_excel_sha256"], "protocol_sha256": bundle["lock"]["protocol_sha256"],
        "global_lock_sha256": _digest(Path(lock_path) if lock_path else Path(results_dir) / "model_lock.json"),
        "model_ids": list(metadata), "primary_model_id": primary, "threshold": 0.5,
        "n_bootstrap": n_bootstrap, "bootstrap_seed": seed, "confidence": 0.95,
        "ci_method": "class-stratified percentile bootstrap conditional on saved predictions; not retraining uncertainty",
        "paired_auc_method": "centered paired class-stratified bootstrap test, plus-one correction, Holm across the prespecified AUC family",
        "paired_auc_family_size": len(paired), "heldout_description": "internal structure-group-held-out test; not an external cohort",
        "label_0_description": "reference-negative; absence of documented nephrotoxicity is not clinical safety evidence",
        "group_bootstrap_included": True, "group_bootstrap_model_ids": [primary] + graph_models,
        "group_bootstrap_method": "whole structure groups resampled with replacement; single-class draws discarded and effective sample counts reported; no retraining uncertainty",
        "files": {p.name: _digest(p) for p in folder.iterdir() if p.suffix in (".csv", ".json", ".npz") and p.name != "report_manifest.json"}}
    _write_json(folder / "report_manifest.json", manifest)
    return manifest
