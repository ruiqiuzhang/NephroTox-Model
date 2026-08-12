from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import project_path, write_json
from .dnn import (
    FeedForwardDNN,
    fit_fixed_epochs,
    fit_with_early_stopping,
    predict_probabilities,
)
from .features import load_feature_matrix
from .metrics import calculate_metrics, calibration_table


REPRESENTATIONS = [
    "morgan2048",
    "maccs166",
    "atompair1024",
    "torsion1024",
    "rdkit166",
]
MODELS = ["rf", "svm", "mlp", "xgb", "dnn"]
PROTOCOLS = ["random", "scaffold"]


def _classical_estimator(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    y_train: np.ndarray,
    svm_probability: bool = True,
) -> BaseEstimator:
    if model_name == "rf":
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "svm":
        return SVC(
            **params,
            kernel="rbf",
            probability=svm_probability,
            class_weight="balanced",
            random_state=seed,
            cache_size=2048,
        )
    if model_name == "mlp":
        converted = dict(params)
        converted["hidden_layer_sizes"] = tuple(converted["hidden_layer_sizes"])
        return MLPClassifier(
            **converted,
            activation="relu",
            solver="adam",
            batch_size=128,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
        )
    if model_name == "xgb":
        n_positive = int(np.sum(y_train == 1))
        n_negative = int(np.sum(y_train == 0))
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=n_negative / n_positive,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classical model: {model_name}")


def _classical_pipeline(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    y_train: np.ndarray,
    representation: str,
    svm_probability: bool = True,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if representation == "rdkit166":
        steps.append(("imputer", SimpleImputer(strategy="median")))
        if model_name in {"svm", "mlp"}:
            steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            _classical_estimator(
                model_name,
                params,
                seed,
                y_train,
                svm_probability=svm_probability,
            ),
        )
    )
    return Pipeline(steps)


def _dnn_transform(
    X_train: np.ndarray,
    X_other: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if representation != "rdkit166":
        return (
            X_train.astype(np.float32, copy=False),
            X_other.astype(np.float32, copy=False),
            {"imputer": None, "scaler": None},
        )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(imputer.fit_transform(X_train))
    transformed_other = scaler.transform(imputer.transform(X_other))
    return (
        transformed_train.astype(np.float32),
        transformed_other.astype(np.float32),
        {"imputer": imputer, "scaler": scaler},
    )


def _prediction_frame(
    data: pd.DataFrame,
    rows: np.ndarray,
    probabilities: np.ndarray,
    partition: str,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    indexed_assignment = assignment.set_index("row_index")
    output = data.loc[
        rows,
        [
            "row_index",
            "compound_id",
            "name",
            "label",
            "standardized_smiles",
            "connectivity_key",
            "murcko_scaffold",
            "scaffold_group",
        ],
    ].copy()
    output["partition"] = partition
    output["fold"] = indexed_assignment.loc[rows, "fold"].to_numpy(dtype=int)
    output["probability"] = np.asarray(probabilities, dtype=float)
    output["prediction_0.5"] = (output["probability"] >= 0.5).astype(int)
    return output.reset_index(drop=True)


def _save_common_outputs(
    experiment_dir: Path,
    data: pd.DataFrame,
    assignment: pd.DataFrame,
    development_rows: np.ndarray,
    test_rows: np.ndarray,
    oof_probability: np.ndarray,
    test_probability: np.ndarray,
    tuning: pd.DataFrame,
    metadata: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    oof_frame = _prediction_frame(
        data, development_rows, oof_probability, "development_oof", assignment
    )
    test_frame = _prediction_frame(
        data, test_rows, test_probability, "locked_test", assignment
    )
    oof_frame.to_csv(experiment_dir / "oof_predictions.csv", index=False)
    test_frame.to_csv(experiment_dir / "test_predictions.csv", index=False)
    tuning.to_csv(experiment_dir / "tuning_results.csv", index=False)
    pd.DataFrame(
        calibration_table(data.loc[test_rows, "label"].to_numpy(), test_probability)
    ).to_csv(experiment_dir / "test_calibration.csv", index=False)

    result = {
        **metadata,
        "development_oof_metrics": calculate_metrics(
            data.loc[development_rows, "label"].to_numpy(),
            oof_probability,
            threshold,
        ),
        "locked_test_metrics": calculate_metrics(
            data.loc[test_rows, "label"].to_numpy(),
            test_probability,
            threshold,
        ),
        "n_development": int(len(development_rows)),
        "n_test": int(len(test_rows)),
    }
    write_json(result, experiment_dir / "metrics.json")
    return result


def run_classical_experiment(
    config: dict[str, Any],
    protocol: str,
    representation: str,
    model_name: str,
    candidates: list[dict[str, Any]],
    output_root: Path,
    save_model: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    seed = int(config["seeds"]["global"])
    threshold = float(config["validation"]["classification_threshold"])
    data = pd.read_csv(project_path(config, "data/processed/model_ready.csv"))
    assignment = pd.read_csv(
        project_path(config, f"data/splits/{protocol}_holdout.csv")
    )
    X = load_feature_matrix(config, representation)
    y = data["label"].to_numpy(dtype=int)
    development_rows = assignment.loc[
        assignment["partition"].eq("development"), "row_index"
    ].to_numpy(dtype=int)
    test_rows = assignment.loc[
        assignment["partition"].eq("test"), "row_index"
    ].to_numpy(dtype=int)
    fold_by_row = assignment.set_index("row_index")["fold"]

    tuning_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    best_oof: np.ndarray | None = None
    for candidate_id, params in enumerate(candidates):
        oof = np.full(len(development_rows), np.nan, dtype=float)
        fold_scores: list[float] = []
        for fold in range(int(config["validation"]["cv_folds"])):
            print(
                f"  candidate {candidate_id + 1}/{len(candidates)}, fold {fold + 1}/5",
                flush=True,
            )
            validation_mask = (
                fold_by_row.loc[development_rows].to_numpy(dtype=int) == fold
            )
            train_rows = development_rows[~validation_mask]
            validation_rows = development_rows[validation_mask]
            estimator = _classical_pipeline(
                model_name,
                params,
                seed + candidate_id * 100 + fold,
                y[train_rows],
                representation,
                svm_probability=model_name != "svm",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator.fit(X[train_rows], y[train_rows])
            if model_name == "svm":
                selection_score = estimator.decision_function(X[validation_rows])
            else:
                selection_score = estimator.predict_proba(X[validation_rows])[:, 1]
            oof[validation_mask] = selection_score
            fold_scores.append(float(roc_auc_score(y[validation_rows], selection_score)))
        if np.isnan(oof).any():
            raise RuntimeError("At least one development row lacks an OOF prediction.")
        mean_score = float(np.mean(fold_scores))
        tuning_rows.append(
            {
                "candidate_id": candidate_id,
                "params_json": json.dumps(params, sort_keys=True),
                **{f"fold_{fold}_roc_auc": score for fold, score in enumerate(fold_scores)},
                "mean_cv_roc_auc": mean_score,
                "std_cv_roc_auc": float(np.std(fold_scores, ddof=1)),
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_oof = oof.copy()
    if best_params is None or best_oof is None:
        raise RuntimeError("No classical candidate completed successfully.")

    # SVM candidates are ranked by their monotonic decision scores. Probability
    # fitting is intentionally performed only for the selected candidate, which
    # avoids nesting libsvm's internal calibration inside every grid candidate.
    if model_name == "svm":
        best_oof = np.full(len(development_rows), np.nan, dtype=float)
        for fold in range(int(config["validation"]["cv_folds"])):
            print(f"  selected SVM probability refit, fold {fold + 1}/5", flush=True)
            validation_mask = (
                fold_by_row.loc[development_rows].to_numpy(dtype=int) == fold
            )
            train_rows = development_rows[~validation_mask]
            validation_rows = development_rows[validation_mask]
            calibrated_estimator = _classical_pipeline(
                model_name,
                best_params,
                seed + 1000 + fold,
                y[train_rows],
                representation,
                svm_probability=True,
            )
            calibrated_estimator.fit(X[train_rows], y[train_rows])
            best_oof[validation_mask] = calibrated_estimator.predict_proba(
                X[validation_rows]
            )[:, 1]

    final_estimator = _classical_pipeline(
        model_name,
        best_params,
        seed,
        y[development_rows],
        representation,
        svm_probability=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_estimator.fit(X[development_rows], y[development_rows])
    test_probability = final_estimator.predict_proba(X[test_rows])[:, 1]
    experiment_dir = output_root / protocol / representation / model_name
    if save_model:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_estimator, experiment_dir / "final_model.joblib", compress=3)
    metadata = {
        "protocol": protocol,
        "representation": representation,
        "model": model_name,
        "selection_metric": "mean_5fold_cv_roc_auc",
        "best_mean_cv_roc_auc": best_score,
        "best_params": best_params,
        "runtime_seconds": float(time.perf_counter() - start),
        "test_set_used_for_tuning": False,
    }
    return _save_common_outputs(
        experiment_dir,
        data,
        assignment,
        development_rows,
        test_rows,
        best_oof,
        test_probability,
        pd.DataFrame(tuning_rows),
        metadata,
        threshold,
    )


def run_dnn_experiment(
    config: dict[str, Any],
    protocol: str,
    representation: str,
    candidates: list[dict[str, Any]],
    output_root: Path,
    save_model: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    base_seed = int(config["seeds"]["global"])
    threshold = float(config["validation"]["classification_threshold"])
    data = pd.read_csv(project_path(config, "data/processed/model_ready.csv"))
    assignment = pd.read_csv(
        project_path(config, f"data/splits/{protocol}_holdout.csv")
    )
    X = load_feature_matrix(config, representation)
    y = data["label"].to_numpy(dtype=int)
    development_rows = assignment.loc[
        assignment["partition"].eq("development"), "row_index"
    ].to_numpy(dtype=int)
    test_rows = assignment.loc[
        assignment["partition"].eq("test"), "row_index"
    ].to_numpy(dtype=int)
    fold_by_row = assignment.set_index("row_index")["fold"]

    tuning_rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    best_oof: np.ndarray | None = None
    best_epochs: list[int] | None = None
    for candidate_id, params in enumerate(candidates):
        oof = np.full(len(development_rows), np.nan, dtype=float)
        fold_scores: list[float] = []
        fold_epochs: list[int] = []
        for fold in range(int(config["validation"]["cv_folds"])):
            print(
                f"  candidate {candidate_id + 1}/{len(candidates)}, fold {fold + 1}/5",
                flush=True,
            )
            validation_mask = (
                fold_by_row.loc[development_rows].to_numpy(dtype=int) == fold
            )
            train_rows = development_rows[~validation_mask]
            validation_rows = development_rows[validation_mask]
            train_features, validation_features, _ = _dnn_transform(
                X[train_rows], X[validation_rows], representation
            )
            fit = fit_with_early_stopping(
                train_features,
                y[train_rows],
                validation_features,
                y[validation_rows],
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
            probability = predict_probabilities(model, validation_features)
            oof[validation_mask] = probability
            fold_scores.append(float(roc_auc_score(y[validation_rows], probability)))
            fold_epochs.append(int(fit.best_epoch))
        if np.isnan(oof).any():
            raise RuntimeError("At least one development row lacks a DNN OOF prediction.")
        mean_score = float(np.mean(fold_scores))
        tuning_rows.append(
            {
                "candidate_id": candidate_id,
                "params_json": json.dumps(params, sort_keys=True),
                **{f"fold_{fold}_roc_auc": score for fold, score in enumerate(fold_scores)},
                **{f"fold_{fold}_best_epoch": epoch for fold, epoch in enumerate(fold_epochs)},
                "mean_cv_roc_auc": mean_score,
                "std_cv_roc_auc": float(np.std(fold_scores, ddof=1)),
                "median_best_epoch": int(np.median(fold_epochs)),
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_oof = oof.copy()
            best_epochs = fold_epochs.copy()
    if best_params is None or best_oof is None or best_epochs is None:
        raise RuntimeError("No DNN candidate completed successfully.")

    final_epochs = max(1, int(np.median(best_epochs)))
    development_features, test_features, preprocessing = _dnn_transform(
        X[development_rows], X[test_rows], representation
    )
    final_probabilities: list[np.ndarray] = []
    experiment_dir = output_root / protocol / representation / "dnn"
    for final_seed in config["seeds"]["dnn_final"]:
        model = fit_fixed_epochs(
            development_features,
            y[development_rows],
            best_params,
            final_epochs,
            int(final_seed),
        )
        final_probabilities.append(predict_probabilities(model, test_features))
        if save_model:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": int(development_features.shape[1]),
                    "params": best_params,
                    "epochs": final_epochs,
                    "seed": int(final_seed),
                },
                experiment_dir / f"final_model_seed_{final_seed}.pt",
            )
    if save_model:
        joblib.dump(preprocessing, experiment_dir / "preprocessing.joblib", compress=3)
    test_probability = np.mean(np.vstack(final_probabilities), axis=0)
    metadata = {
        "protocol": protocol,
        "representation": representation,
        "model": "dnn",
        "selection_metric": "mean_5fold_cv_roc_auc",
        "best_mean_cv_roc_auc": best_score,
        "best_params": best_params,
        "best_epochs_by_fold": best_epochs,
        "final_refit_epochs": final_epochs,
        "final_ensemble_seeds": [int(seed) for seed in config["seeds"]["dnn_final"]],
        "runtime_seconds": float(time.perf_counter() - start),
        "test_set_used_for_tuning": False,
    }
    return _save_common_outputs(
        experiment_dir,
        data,
        assignment,
        development_rows,
        test_rows,
        best_oof,
        test_probability,
        pd.DataFrame(tuning_rows),
        metadata,
        threshold,
    )


def train_experiments(
    config: dict[str, Any],
    protocols: list[str],
    representations: list[str],
    models: list[str],
    quick: bool = False,
    force: bool = False,
    save_models: bool = False,
) -> list[dict[str, Any]]:
    invalid_protocols = sorted(set(protocols) - set(PROTOCOLS))
    invalid_representations = sorted(set(representations) - set(REPRESENTATIONS))
    invalid_models = sorted(set(models) - set(MODELS))
    if invalid_protocols or invalid_representations or invalid_models:
        raise ValueError(
            "Invalid selections: "
            f"protocols={invalid_protocols}, representations={invalid_representations}, "
            f"models={invalid_models}"
        )
    output_root = project_path(
        config, "results/smoke" if quick else "results/experiments"
    )
    results: list[dict[str, Any]] = []
    for protocol in protocols:
        for representation in representations:
            for model_name in models:
                experiment_dir = output_root / protocol / representation / model_name
                metrics_path = experiment_dir / "metrics.json"
                if metrics_path.exists() and not force:
                    print(f"SKIP existing: {protocol}/{representation}/{model_name}", flush=True)
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        results.append(json.load(handle))
                    continue
                candidates = list(config["models"][model_name])
                if quick:
                    candidates = candidates[:1]
                print(f"RUN {protocol}/{representation}/{model_name}", flush=True)
                if model_name == "dnn":
                    result = run_dnn_experiment(
                        config,
                        protocol,
                        representation,
                        candidates,
                        output_root,
                        save_models,
                    )
                else:
                    result = run_classical_experiment(
                        config,
                        protocol,
                        representation,
                        model_name,
                        candidates,
                        output_root,
                        save_models,
                    )
                results.append(result)
                print(
                    f"DONE {protocol}/{representation}/{model_name}: "
                    f"CV AUC={result['best_mean_cv_roc_auc']:.4f}, "
                    f"test AUC={result['locked_test_metrics']['roc_auc']:.4f}",
                    flush=True,
                )
    return results
