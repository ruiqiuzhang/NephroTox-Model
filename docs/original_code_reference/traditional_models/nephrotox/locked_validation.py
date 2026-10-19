from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import project_path, write_json
from .data import sha256_file
from .locked_pipeline import MODELS, REPRESENTATIONS, _verify_lock
from .metrics import calculate_metrics


def _assert_close_metrics(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    for metric in [
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
    ]:
        if not np.isclose(float(expected[metric]), float(observed[metric]), atol=1e-12):
            raise RuntimeError(
                f"Metric mismatch for {metric}: expected {expected[metric]}, "
                f"observed {observed[metric]}"
            )


def validate_locked_outputs(config: dict[str, Any]) -> dict[str, Any]:
    lock, lock_hash = _verify_lock(config)
    results_root = project_path(config, config["run"]["results_root"])
    evaluation_manifest = results_root / "evaluation/test_evaluation_manifest.json"
    if not evaluation_manifest.exists():
        raise RuntimeError("The locked test evaluation is incomplete.")
    evaluation = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    if evaluation["lock_manifest_sha256"] != lock_hash:
        raise RuntimeError("Evaluation and candidate-lock hashes differ.")

    development = pd.read_csv(
        project_path(config, config["run"]["development_file"])
    )
    test_features = pd.read_csv(
        project_path(config, config["run"]["test_features_file"])
    ).sort_values("row_index")
    expected_test_rows = test_features["row_index"].to_numpy(dtype=int)
    threshold = float(config["validation"]["classification_threshold"])
    checks: list[dict[str, Any]] = []

    for representation in REPRESENTATIONS:
        for model_name in MODELS:
            tuning_dir = results_root / "tuning" / representation / model_name
            evaluation_dir = results_root / "evaluation" / representation / model_name
            tuning = pd.read_csv(tuning_dir / "tuning_results.csv")
            candidate_predictions = pd.read_csv(
                tuning_dir / "candidate_oof_predictions.csv"
            )
            oof = pd.read_csv(tuning_dir / "oof_predictions.csv")
            expected_candidates = len(config["models"][model_name])
            expected_tuning_rows = expected_candidates * int(
                config["validation"]["cv_folds"]
            )
            if len(tuning) != expected_tuning_rows:
                raise RuntimeError(f"Incomplete tuning rows in {tuning_dir}")
            if len(candidate_predictions) != expected_candidates * len(development):
                raise RuntimeError(f"Incomplete candidate predictions in {tuning_dir}")
            if len(oof) != len(development) or oof["row_index"].nunique() != len(
                development
            ):
                raise RuntimeError(f"Incomplete selected OOF predictions in {tuning_dir}")

            predictions = pd.read_csv(
                evaluation_dir / "test_predictions.csv"
            ).sort_values("row_index")
            if not np.array_equal(
                predictions["row_index"].to_numpy(dtype=int), expected_test_rows
            ):
                raise RuntimeError(f"Test row mismatch in {evaluation_dir}")
            probability = predictions["probability"].to_numpy(dtype=float)
            if not np.all(np.isfinite(probability)) or np.any(
                (probability < 0) | (probability > 1)
            ):
                raise RuntimeError(f"Invalid probabilities in {evaluation_dir}")
            expected_prediction = (probability >= threshold).astype(int)
            if not np.array_equal(
                predictions["prediction_0.5"].to_numpy(dtype=int), expected_prediction
            ):
                raise RuntimeError(f"Threshold predictions differ in {evaluation_dir}")
            metrics = json.loads(
                (evaluation_dir / "metrics.json").read_text(encoding="utf-8")
            )
            recomputed = calculate_metrics(
                predictions["label"].to_numpy(dtype=int), probability, threshold
            )
            _assert_close_metrics(metrics["locked_test_metrics"], recomputed)
            if metrics.get("test_set_used_for_tuning") is not False:
                raise RuntimeError(f"Invalid test-access flag in {evaluation_dir}")
            for artifact in metrics["model_artifacts"]:
                path = results_root / artifact["path"]
                if not path.exists() or path.stat().st_size != artifact["bytes"]:
                    raise RuntimeError(f"Missing model artifact: {path}")
                if sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"Model artifact hash mismatch: {path}")
            checks.append(
                {
                    "representation": representation,
                    "model": model_name,
                    "status": "PASS",
                    "n_development_oof": len(oof),
                    "n_test": len(predictions),
                    "n_model_artifacts": len(metrics["model_artifacts"]),
                }
            )

    summary_path = results_root / "summary/all_metrics_with_ci.csv"
    figure_manifest_path = results_root / "figures/figure_manifest.json"
    if not summary_path.exists() or len(pd.read_csv(summary_path)) != 25:
        raise RuntimeError("The 25-row summary with confidence intervals is missing.")
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    for figure_name in ["Figure4", "Figure5", "FigureS_AD"]:
        for raw_path in figure_manifest[figure_name]["outputs"]:
            path = Path(raw_path)
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing figure output: {path}")
            if path.suffix.lower() == ".svg" and "<text" not in path.read_text(
                encoding="utf-8"
            ):
                raise RuntimeError(f"SVG text is not editable: {path}")
    ad_source = results_root / "figures/source_data/FigureS_AD"
    ad_test = pd.read_csv(ad_source / "test_applicability_domain.csv")
    if len(ad_test) != 371:
        raise RuntimeError("The test applicability-domain table must contain 371 rows.")
    required_ad_columns = {
        "primary_model",
        "primary_model_role",
        "primary_normalized_threshold_distance",
        "descriptive_model",
        "descriptive_model_role",
        "descriptive_normalized_threshold_distance",
    }
    if not required_ad_columns.issubset(ad_test.columns):
        raise RuntimeError("Primary and descriptive AD model columns are incomplete.")
    if len(pd.read_csv(ad_source / "development_leave_one_out_similarity.csv")) != 1482:
        raise RuntimeError("The development LOO similarity table must contain 1,482 rows.")
    if len(pd.read_csv(ad_source / "ad_performance_all25.csv")) != 150:
        raise RuntimeError("Expected six AD strata for each of the 25 models.")
    if len(pd.read_csv(project_path(config, "data/splits/dataset_split.csv"))) != 1853:
        raise RuntimeError("dataset_split.csv must contain 1,853 rows.")
    if len(pd.read_csv(project_path(config, "data/splits/cv_folds.csv"))) != 1482:
        raise RuntimeError("cv_folds.csv must contain 1,482 rows.")
    supplementary_dir = results_root / "supplementary_tables"
    if len(pd.read_csv(supplementary_dir / "Supplementary_Table_S1.csv")) != 25:
        raise RuntimeError("Supplementary Table S1 must contain 25 model combinations.")
    if len(pd.read_csv(supplementary_dir / "Supplementary_Table_S2A_search_spaces.csv")) != 15:
        raise RuntimeError("Supplementary Table S2A must contain 15 candidate configurations.")
    if len(pd.read_csv(supplementary_dir / "Supplementary_Table_S2B_optimized.csv")) != 25:
        raise RuntimeError("Supplementary Table S2B must contain 25 optimized configurations.")

    payload = {
        "status": "PASS",
        "run_id": config["run"]["id"],
        "lock_manifest_sha256": lock_hash,
        "n_experiments": len(checks),
        "protocol_checks": {
            "test_evaluated_once_after_lock": evaluation[
                "test_evaluated_once_after_lock"
            ],
            "n_locked_experiments": lock["n_experiments"],
            "development_size": int(len(development)),
            "test_size": int(len(test_features)),
        },
        "experiments": checks,
    }
    write_json(payload, results_root / "output_validation.json")
    return payload
