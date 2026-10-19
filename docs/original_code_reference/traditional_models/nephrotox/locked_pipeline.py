from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from .config import project_path, write_json
from .data import sha256_file
from .dnn import (
    FeedForwardDNN,
    fit_fixed_epochs,
    fit_with_early_stopping,
    predict_probabilities,
)
from .features import load_feature_matrix
from .metrics import calculate_metrics, calibration_table
from .training import (
    MODELS,
    REPRESENTATIONS,
    _classical_pipeline,
    _dnn_transform,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protocol_name(config: dict[str, Any]) -> str:
    return str(config["run"].get("protocol", config["run"]["id"]))


def _results_root(config: dict[str, Any]) -> Path:
    return project_path(config, config["run"]["results_root"])


def _tuning_root(config: dict[str, Any], quick: bool = False) -> Path:
    suffix = "smoke/tuning" if quick else "tuning"
    return _results_root(config) / suffix


def _experiment_dir(
    config: dict[str, Any], representation: str, model_name: str, quick: bool = False
) -> Path:
    return _tuning_root(config, quick) / representation / model_name


def _load_development(config: dict[str, Any]) -> pd.DataFrame:
    path = project_path(config, config["run"]["development_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"Development protocol file is missing: {path}. Run 00_prepare_data.py first."
        )
    frame = pd.read_csv(path).sort_values("row_index").reset_index(drop=True)
    required = {"row_index", "compound_id", "label", "fold", "scaffold_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Development file lacks required columns: {missing}")
    expected_folds = set(range(int(config["validation"]["cv_folds"])))
    observed_folds = set(frame["fold"].astype(int).unique())
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected development folds {sorted(expected_folds)}, observed {sorted(observed_folds)}."
        )
    return frame


def _selected_oof_frame(
    development: pd.DataFrame,
    probabilities: np.ndarray,
    representation: str,
    model_name: str,
    threshold: float,
    protocol: str,
) -> pd.DataFrame:
    output = development[
        [
            "row_index",
            "compound_id",
            "name",
            "label",
            "standardized_smiles",
            "connectivity_key",
            "murcko_scaffold",
            "scaffold_group",
            "fold",
        ]
    ].copy()
    output["probability"] = np.asarray(probabilities, dtype=float)
    output["prediction_0.5"] = (output["probability"] >= threshold).astype(int)
    output["protocol"] = protocol
    output["representation"] = representation
    output["model"] = model_name
    output["partition"] = "development_oof"
    return output


def _select_candidate(summary: pd.DataFrame) -> pd.Series:
    ranked = summary.sort_values(
        ["mean_cv_roc_auc", "std_cv_roc_auc", "candidate_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def _candidate_prediction_rows(
    development: pd.DataFrame,
    candidate_id: int,
    scores: np.ndarray,
    score_type: str,
) -> pd.DataFrame:
    output = development[["row_index", "compound_id", "label", "fold"]].copy()
    output.insert(0, "candidate_id", candidate_id)
    output["score"] = np.asarray(scores, dtype=float)
    output["score_type"] = score_type
    return output


def tune_classical_experiment(
    config: dict[str, Any],
    representation: str,
    model_name: str,
    candidates: list[dict[str, Any]],
    quick: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    development = _load_development(config)
    row_indices = development["row_index"].to_numpy(dtype=int)
    labels = development["label"].to_numpy(dtype=int)
    folds = development["fold"].to_numpy(dtype=int)
    features = load_feature_matrix(config, representation)[row_indices]
    base_seed = int(config["seeds"]["global"])
    n_folds = int(config["validation"]["cv_folds"])
    experiment_dir = _experiment_dir(config, representation, model_name, quick)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_id, params in enumerate(candidates):
        candidate_scores = np.full(len(development), np.nan, dtype=float)
        for fold in range(n_folds):
            fold_start = time.perf_counter()
            validation_mask = folds == fold
            training_mask = ~validation_mask
            estimator = _classical_pipeline(
                model_name,
                params,
                base_seed + candidate_id * 100 + fold,
                labels[training_mask],
                representation,
                svm_probability=model_name != "svm",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator.fit(features[training_mask], labels[training_mask])
            if model_name == "svm":
                fold_scores = estimator.decision_function(features[validation_mask])
                score_type = "decision_function"
            else:
                fold_scores = estimator.predict_proba(features[validation_mask])[:, 1]
                score_type = "probability"
            candidate_scores[validation_mask] = fold_scores
            tuning_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "params_json": json.dumps(params, sort_keys=True),
                    "n_train": int(training_mask.sum()),
                    "n_validation": int(validation_mask.sum()),
                    "validation_roc_auc": float(
                        roc_auc_score(labels[validation_mask], fold_scores)
                    ),
                    "runtime_seconds": float(time.perf_counter() - fold_start),
                }
            )
        if np.isnan(candidate_scores).any():
            raise RuntimeError(f"Candidate {candidate_id} has missing OOF scores.")
        prediction_frames.append(
            _candidate_prediction_rows(
                development, candidate_id, candidate_scores, score_type
            )
        )

    tuning = pd.DataFrame(tuning_rows)
    summary = (
        tuning.groupby(["candidate_id", "params_json"], as_index=False)
        .agg(
            mean_cv_roc_auc=("validation_roc_auc", "mean"),
            std_cv_roc_auc=("validation_roc_auc", "std"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values("candidate_id")
    )
    selected = _select_candidate(summary)
    selected_id = int(selected["candidate_id"])
    selected_params = candidates[selected_id]
    candidate_predictions = pd.concat(prediction_frames, ignore_index=True)

    if model_name == "svm":
        selected_probability = np.full(len(development), np.nan, dtype=float)
        for fold in range(n_folds):
            validation_mask = folds == fold
            training_mask = ~validation_mask
            estimator = _classical_pipeline(
                model_name,
                selected_params,
                base_seed + 1000 + fold,
                labels[training_mask],
                representation,
                svm_probability=True,
            )
            estimator.fit(features[training_mask], labels[training_mask])
            selected_probability[validation_mask] = estimator.predict_proba(
                features[validation_mask]
            )[:, 1]
    else:
        selected_probability = (
            candidate_predictions.loc[
                candidate_predictions["candidate_id"].eq(selected_id)
            ]
            .sort_values("row_index")["score"]
            .to_numpy(dtype=float)
        )

    threshold = float(config["validation"]["classification_threshold"])
    oof = _selected_oof_frame(
        development,
        selected_probability,
        representation,
        model_name,
        threshold,
        _protocol_name(config),
    )
    tuning.to_csv(experiment_dir / "tuning_results.csv", index=False)
    summary.to_csv(experiment_dir / "candidate_summary.csv", index=False)
    candidate_predictions.to_csv(
        experiment_dir / "candidate_oof_predictions.csv", index=False
    )
    oof.to_csv(experiment_dir / "oof_predictions.csv", index=False)
    payload = {
        "run_id": config["run"]["id"],
        "protocol": _protocol_name(config),
        "representation": representation,
        "model": model_name,
        "selection_metric": f"mean_{n_folds}fold_development_roc_auc",
        "tie_break_rule": "higher mean AUC, lower AUC SD, lower candidate_id",
        "selected_candidate_id": selected_id,
        "best_params": selected_params,
        "best_mean_cv_roc_auc": float(selected["mean_cv_roc_auc"]),
        "best_std_cv_roc_auc": float(selected["std_cv_roc_auc"]),
        "development_oof_metrics": calculate_metrics(
            labels, selected_probability, threshold
        ),
        "n_development": int(len(development)),
        "n_folds": n_folds,
        "test_data_loaded": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }
    write_json(payload, experiment_dir / "selected_params.json")
    return payload


def tune_dnn_experiment(
    config: dict[str, Any],
    representation: str,
    candidates: list[dict[str, Any]],
    quick: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    development = _load_development(config)
    row_indices = development["row_index"].to_numpy(dtype=int)
    labels = development["label"].to_numpy(dtype=int)
    folds = development["fold"].to_numpy(dtype=int)
    features = load_feature_matrix(config, representation)[row_indices]
    base_seed = int(config["seeds"]["global"])
    n_folds = int(config["validation"]["cv_folds"])
    experiment_dir = _experiment_dir(config, representation, "dnn", quick)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_id, params in enumerate(candidates):
        candidate_probability = np.full(len(development), np.nan, dtype=float)
        for fold in range(n_folds):
            fold_start = time.perf_counter()
            validation_mask = folds == fold
            training_mask = ~validation_mask
            train_features, validation_features, _ = _dnn_transform(
                features[training_mask], features[validation_mask], representation
            )
            fit = fit_with_early_stopping(
                train_features,
                labels[training_mask],
                validation_features,
                labels[validation_mask],
                params,
                config["dnn_training"],
                base_seed + candidate_id * 100 + fold,
            )
            model = FeedForwardDNN(
                input_dim=train_features.shape[1],
                hidden_layers=list(params["hidden_layers"]),
                dropout=float(params["dropout"]),
                batch_norm=bool(params.get("batch_norm", False)),
            )
            model.load_state_dict(fit.state_dict)
            fold_probability = predict_probabilities(model, validation_features)
            candidate_probability[validation_mask] = fold_probability
            tuning_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "params_json": json.dumps(params, sort_keys=True),
                    "n_train": int(training_mask.sum()),
                    "n_validation": int(validation_mask.sum()),
                    "validation_roc_auc": float(
                        roc_auc_score(labels[validation_mask], fold_probability)
                    ),
                    "best_epoch": int(fit.best_epoch),
                    "best_validation_auc_during_training": float(
                        fit.best_validation_auc
                    ),
                    "runtime_seconds": float(time.perf_counter() - fold_start),
                }
            )
            history_rows.extend(
                {
                    "candidate_id": candidate_id,
                    "fold": fold,
                    **history,
                }
                for history in fit.history
            )
        if np.isnan(candidate_probability).any():
            raise RuntimeError(f"DNN candidate {candidate_id} has missing OOF scores.")
        prediction_frames.append(
            _candidate_prediction_rows(
                development,
                candidate_id,
                candidate_probability,
                "probability",
            )
        )

    tuning = pd.DataFrame(tuning_rows)
    summary = (
        tuning.groupby(["candidate_id", "params_json"], as_index=False)
        .agg(
            mean_cv_roc_auc=("validation_roc_auc", "mean"),
            std_cv_roc_auc=("validation_roc_auc", "std"),
            median_best_epoch=("best_epoch", "median"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values("candidate_id")
    )
    selected = _select_candidate(summary)
    selected_id = int(selected["candidate_id"])
    selected_params = candidates[selected_id]
    candidate_predictions = pd.concat(prediction_frames, ignore_index=True)
    selected_probability = (
        candidate_predictions.loc[candidate_predictions["candidate_id"].eq(selected_id)]
        .sort_values("row_index")["score"]
        .to_numpy(dtype=float)
    )
    selected_epochs = (
        tuning.loc[tuning["candidate_id"].eq(selected_id)]
        .sort_values("fold")["best_epoch"]
        .astype(int)
        .tolist()
    )
    final_epochs = max(1, int(np.median(selected_epochs)))
    threshold = float(config["validation"]["classification_threshold"])
    oof = _selected_oof_frame(
        development,
        selected_probability,
        representation,
        "dnn",
        threshold,
        _protocol_name(config),
    )
    tuning.to_csv(experiment_dir / "tuning_results.csv", index=False)
    summary.to_csv(experiment_dir / "candidate_summary.csv", index=False)
    candidate_predictions.to_csv(
        experiment_dir / "candidate_oof_predictions.csv", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        experiment_dir / "training_history.csv", index=False
    )
    oof.to_csv(experiment_dir / "oof_predictions.csv", index=False)
    payload = {
        "run_id": config["run"]["id"],
        "protocol": _protocol_name(config),
        "representation": representation,
        "model": "dnn",
        "selection_metric": f"mean_{n_folds}fold_development_roc_auc",
        "tie_break_rule": "higher mean AUC, lower AUC SD, lower candidate_id",
        "selected_candidate_id": selected_id,
        "best_params": selected_params,
        "best_mean_cv_roc_auc": float(selected["mean_cv_roc_auc"]),
        "best_std_cv_roc_auc": float(selected["std_cv_roc_auc"]),
        "best_epochs_by_fold": selected_epochs,
        "final_refit_epochs": final_epochs,
        "final_ensemble_seeds": [int(seed) for seed in config["seeds"]["dnn_final"]],
        "development_oof_metrics": calculate_metrics(
            labels, selected_probability, threshold
        ),
        "n_development": int(len(development)),
        "n_folds": n_folds,
        "test_data_loaded": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }
    write_json(payload, experiment_dir / "selected_params.json")
    return payload


def tune_experiments(
    config: dict[str, Any],
    representations: list[str] | None = None,
    models: list[str] | None = None,
    quick: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    selected_representations = representations or list(REPRESENTATIONS)
    selected_models = models or list(MODELS)
    invalid_representations = sorted(
        set(selected_representations) - set(REPRESENTATIONS)
    )
    invalid_models = sorted(set(selected_models) - set(MODELS))
    if invalid_representations or invalid_models:
        raise ValueError(
            f"Invalid selections: representations={invalid_representations}, "
            f"models={invalid_models}"
        )

    results: list[dict[str, Any]] = []
    for representation in selected_representations:
        for model_name in selected_models:
            experiment_dir = _experiment_dir(
                config, representation, model_name, quick
            )
            selected_path = experiment_dir / "selected_params.json"
            if selected_path.exists() and not force:
                with selected_path.open("r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
                print(f"SKIP existing tuning: {representation}/{model_name}", flush=True)
                continue
            candidates = list(config["models"][model_name])
            if quick:
                candidates = candidates[:1]
            print(f"TUNE {representation}/{model_name}", flush=True)
            if model_name == "dnn":
                result = tune_dnn_experiment(
                    config, representation, candidates, quick=quick
                )
            else:
                result = tune_classical_experiment(
                    config,
                    representation,
                    model_name,
                    candidates,
                    quick=quick,
                )
            results.append(result)
            print(
                f"DONE tuning {representation}/{model_name}: "
                f"CV AUC={result['best_mean_cv_roc_auc']:.4f}",
                flush=True,
            )
    return results


def _hash_files(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }


def _modeling_source_paths(config: dict[str, Any]) -> list[Path]:
    root = project_path(config, ".")
    relative = [
        "configs/experiment.json",
        "configs/rdkit_descriptors_166.txt",
        "nephrotox/config.py",
        "nephrotox/data.py",
        "nephrotox/features.py",
        "nephrotox/metrics.py",
        "nephrotox/dnn.py",
        "nephrotox/training.py",
        "nephrotox/locked_pipeline.py",
    ]
    return [root / value for value in relative]


def lock_candidates(config: dict[str, Any]) -> dict[str, Any]:
    root = project_path(config, ".")
    results_root = _results_root(config)
    lock_dir = results_root / "lock"
    lock_path = lock_dir / "locked_candidates.json"
    if lock_path.exists():
        with lock_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    experiments: list[dict[str, Any]] = []
    tuning_paths: list[Path] = []
    for representation in REPRESENTATIONS:
        for model_name in MODELS:
            experiment_dir = _experiment_dir(config, representation, model_name)
            required = [
                experiment_dir / "selected_params.json",
                experiment_dir / "tuning_results.csv",
                experiment_dir / "candidate_summary.csv",
                experiment_dir / "candidate_oof_predictions.csv",
                experiment_dir / "oof_predictions.csv",
            ]
            if model_name == "dnn":
                required.append(experiment_dir / "training_history.csv")
            missing = [path for path in required if not path.exists()]
            if missing:
                raise RuntimeError(
                    f"Cannot lock before all tuning artifacts exist: {missing}"
                )
            with required[0].open("r", encoding="utf-8") as handle:
                selected = json.load(handle)
            if selected.get("test_data_loaded") is not False:
                raise RuntimeError(f"Invalid test-access flag in {required[0]}")
            oof = pd.read_csv(experiment_dir / "oof_predictions.csv")
            if len(oof) != 1482 or oof["row_index"].nunique() != 1482:
                raise RuntimeError(f"Incomplete OOF predictions in {experiment_dir}")
            experiments.append(selected)
            tuning_paths.extend(required)

    protocol_manifest = project_path(config, config["run"]["protocol_manifest"])
    feature_manifest = project_path(
        config, "data/processed/features/feature_manifest.json"
    )
    feature_paths = [
        project_path(config, f"data/processed/features/{name}.npz")
        for name in REPRESENTATIONS
    ]
    immutable_paths = (
        _modeling_source_paths(config)
        + [
            protocol_manifest,
            feature_manifest,
            project_path(config, config["run"]["split_file"]),
            project_path(config, config["run"]["development_file"]),
            project_path(config, config["run"]["test_features_file"]),
        ]
        + feature_paths
        + tuning_paths
    )
    missing_immutable = [path for path in immutable_paths if not path.exists()]
    if missing_immutable:
        raise RuntimeError(f"Files required for locking are missing: {missing_immutable}")

    payload = {
        "run_id": config["run"]["id"],
        "protocol": _protocol_name(config),
        "locked_at_utc": _utc_now(),
        "lock_scope": (
            "feature strategy, split, folds, source implementation, hyperparameters, "
            "DNN architecture, final refit rule, and all development-only outputs"
        ),
        "selection_metric": config["validation"]["selection_metric"],
        "classification_threshold": float(
            config["validation"]["classification_threshold"]
        ),
        "test_access_before_lock": False,
        "n_experiments": len(experiments),
        "experiments": experiments,
        "locked_file_sha256": _hash_files(immutable_paths, root),
        "sealed_test_label_sha256": json.loads(
            protocol_manifest.read_text(encoding="utf-8")
        )["files"]["sealed_test_labels"]["sha256"],
    }
    lock_dir.mkdir(parents=True, exist_ok=True)
    write_json(payload, lock_path)
    return payload


def _verify_lock(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    root = project_path(config, ".")
    lock_path = _results_root(config) / "lock/locked_candidates.json"
    if not lock_path.exists():
        raise RuntimeError("The test set remains sealed because no lock manifest exists.")
    lock_hash = sha256_file(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    changed: list[str] = []
    for relative, expected in payload["locked_file_sha256"].items():
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(
            "The locked configuration has changed; refusing to open the test labels: "
            + ", ".join(changed)
        )
    sealed_path = project_path(config, config["run"]["sealed_test_labels_file"])
    if sha256_file(sealed_path) != payload["sealed_test_label_sha256"]:
        raise RuntimeError("The sealed test label file hash does not match the lock.")
    return payload, lock_hash


def _model_artifact_manifest(paths: list[Path], base: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(base)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _test_prediction_frame(
    test_features: pd.DataFrame,
    test_labels: pd.DataFrame,
    probabilities: np.ndarray,
    representation: str,
    model_name: str,
    threshold: float,
    protocol: str,
) -> pd.DataFrame:
    output = test_features.merge(
        test_labels,
        on=["row_index", "compound_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values("row_index")
    output["probability"] = np.asarray(probabilities, dtype=float)
    output["prediction_0.5"] = (output["probability"] >= threshold).astype(int)
    output["fold"] = -1
    output["protocol"] = protocol
    output["representation"] = representation
    output["model"] = model_name
    output["partition"] = "locked_test"
    return output


def evaluate_locked_test(config: dict[str, Any]) -> dict[str, Any]:
    lock, lock_hash = _verify_lock(config)
    results_root = _results_root(config)
    evaluation_root = results_root / "evaluation"
    started_path = evaluation_root / "evaluation_started.json"
    completed_path = evaluation_root / "test_evaluation_manifest.json"
    if completed_path.exists():
        raise RuntimeError(
            "The locked test evaluation is already complete; refusing a second evaluation."
        )
    if started_path.exists():
        started = json.loads(started_path.read_text(encoding="utf-8"))
        if started["lock_manifest_sha256"] != lock_hash:
            raise RuntimeError("A different locked evaluation was already started.")
    else:
        started = {
            "run_id": config["run"]["id"],
            "started_at_utc": _utc_now(),
            "lock_manifest_sha256": lock_hash,
            "resume_policy": "resume incomplete combinations under the identical lock only",
        }
        write_json(started, started_path)

    development = _load_development(config)
    test_features = pd.read_csv(
        project_path(config, config["run"]["test_features_file"])
    ).sort_values("row_index")
    test_labels = pd.read_csv(
        project_path(config, config["run"]["sealed_test_labels_file"])
    ).sort_values("row_index")
    if len(test_features) != 371 or len(test_labels) != 371:
        raise RuntimeError("The locked test partition must contain exactly 371 compounds.")
    development_rows = development["row_index"].to_numpy(dtype=int)
    development_labels = development["label"].to_numpy(dtype=int)
    test_rows = test_features["row_index"].to_numpy(dtype=int)
    test_y = test_labels["label"].to_numpy(dtype=int)
    threshold = float(config["validation"]["classification_threshold"])

    experiment_results: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        all_features = load_feature_matrix(config, representation)
        development_X = all_features[development_rows]
        test_X = all_features[test_rows]
        for model_name in MODELS:
            experiment_dir = evaluation_root / representation / model_name
            metrics_path = experiment_dir / "metrics.json"
            if metrics_path.exists():
                experiment_results.append(
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                )
                continue
            start = time.perf_counter()
            selection_path = (
                _experiment_dir(config, representation, model_name)
                / "selected_params.json"
            )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            params = selection["best_params"]
            experiment_dir.mkdir(parents=True, exist_ok=True)
            model_paths: list[Path] = []

            if model_name == "dnn":
                transformed_dev, transformed_test, preprocessing = _dnn_transform(
                    development_X, test_X, representation
                )
                probabilities: list[np.ndarray] = []
                for seed in selection["final_ensemble_seeds"]:
                    model = fit_fixed_epochs(
                        transformed_dev,
                        development_labels,
                        params,
                        int(selection["final_refit_epochs"]),
                        int(seed),
                    )
                    probabilities.append(
                        predict_probabilities(model, transformed_test)
                    )
                    model_path = experiment_dir / f"final_model_seed_{seed}.pt"
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "input_dim": int(transformed_dev.shape[1]),
                            "params": params,
                            "epochs": int(selection["final_refit_epochs"]),
                            "seed": int(seed),
                            "representation": representation,
                        },
                        model_path,
                    )
                    model_paths.append(model_path)
                preprocessing_path = experiment_dir / "preprocessing.joblib"
                joblib.dump(preprocessing, preprocessing_path, compress=3)
                model_paths.append(preprocessing_path)
                test_probability = np.mean(np.vstack(probabilities), axis=0)
            else:
                estimator = _classical_pipeline(
                    model_name,
                    params,
                    int(config["seeds"]["global"]),
                    development_labels,
                    representation,
                    svm_probability=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    estimator.fit(development_X, development_labels)
                test_probability = estimator.predict_proba(test_X)[:, 1]
                model_path = experiment_dir / "final_model.joblib"
                joblib.dump(estimator, model_path, compress=3)
                model_paths.append(model_path)

            predictions = _test_prediction_frame(
                test_features,
                test_labels,
                test_probability,
                representation,
                model_name,
                threshold,
                _protocol_name(config),
            )
            predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
            pd.DataFrame(calibration_table(test_y, test_probability)).to_csv(
                experiment_dir / "test_calibration.csv", index=False
            )
            result = {
                "run_id": config["run"]["id"],
                "protocol": _protocol_name(config),
                "representation": representation,
                "model": model_name,
                "lock_manifest_sha256": lock_hash,
                "selected_candidate_id": selection["selected_candidate_id"],
                "best_params": params,
                "best_mean_cv_roc_auc": selection["best_mean_cv_roc_auc"],
                "best_std_cv_roc_auc": selection["best_std_cv_roc_auc"],
                "development_oof_metrics": selection["development_oof_metrics"],
                "locked_test_metrics": calculate_metrics(
                    test_y, test_probability, threshold
                ),
                "n_development": int(len(development)),
                "n_test": int(len(test_features)),
                "classification_threshold": threshold,
                "test_set_used_for_tuning": False,
                "model_artifacts": _model_artifact_manifest(
                    model_paths, results_root
                ),
                "runtime_seconds": float(time.perf_counter() - start),
            }
            write_json(result, metrics_path)
            experiment_results.append(result)
            print(f"EVALUATED {representation}/{model_name}", flush=True)

    if len(experiment_results) != len(REPRESENTATIONS) * len(MODELS):
        raise RuntimeError("The one-time evaluation did not produce all 25 experiments.")
    completed = {
        **started,
        "completed_at_utc": _utc_now(),
        "n_experiments": len(experiment_results),
        "test_evaluated_once_after_lock": True,
        "test_size": int(len(test_features)),
        "test_label_counts": {
            str(key): int(value)
            for key, value in test_labels["label"].value_counts().sort_index().items()
        },
        "experiments": [
            {
                "representation": result["representation"],
                "model": result["model"],
                "metrics_sha256": sha256_file(
                    evaluation_root
                    / result["representation"]
                    / result["model"]
                    / "metrics.json"
                ),
            }
            for result in experiment_results
        ],
    }
    write_json(completed, completed_path)
    return completed


def lock_digest(config: dict[str, Any]) -> str:
    """Return the current lock hash for reporting and validation."""
    path = _results_root(config) / "lock/locked_candidates.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
