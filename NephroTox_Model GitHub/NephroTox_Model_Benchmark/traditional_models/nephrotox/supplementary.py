from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_path, write_json
from .locked_pipeline import MODELS, REPRESENTATIONS


MODEL_LABELS = {"rf": "RF", "svm": "SVM", "mlp": "MLP", "xgb": "XGBoost", "dnn": "DNN"}
REPRESENTATION_LABELS = {
    "morgan2048": "Morgan",
    "maccs166": "MACCS",
    "atompair1024": "AtomPairs",
    "torsion1024": "TopologicalTorsion",
    "rdkit166": "Descriptor",
}
MODEL_ORDER = ["rf", "svm", "mlp", "xgb", "dnn"]


def _results_root(config: dict[str, Any]) -> Path:
    return project_path(config, config["run"]["results_root"])


def export_split_tables(config: dict[str, Any]) -> dict[str, Any]:
    """Export public split and fold tables aligned to the locked protocol."""
    split = pd.read_csv(project_path(config, config["run"]["split_file"]))
    data = pd.read_csv(project_path(config, "data/processed/model_ready.csv"))
    identity = data[
        ["row_index", "compound_id", "name", "standardized_smiles", "label"]
    ]
    aligned = identity.merge(
        split[["row_index", "compound_id", "partition", "fold"]],
        on=["row_index", "compound_id"],
        validate="one_to_one",
    ).sort_values("row_index")

    dataset_split = aligned[
        ["row_index", "compound_id", "name", "standardized_smiles", "label", "partition"]
    ].copy()
    dataset_split["dataset_split"] = dataset_split.pop("partition").replace(
        {"test": "independent_test"}
    )
    dataset_split["split_seed"] = int(config["seeds"]["global"])
    dataset_split["split_protocol"] = "stratified_random_80_20"

    cv_folds = aligned.loc[aligned["partition"].eq("development")].copy()
    cv_folds["fold_zero_based"] = cv_folds["fold"].astype(int)
    cv_folds["cv_fold"] = cv_folds["fold_zero_based"] + 1
    cv_folds["cv_role"] = "validation in listed fold; training in the other nine folds"
    cv_folds = cv_folds[
        [
            "row_index",
            "compound_id",
            "name",
            "standardized_smiles",
            "label",
            "cv_fold",
            "fold_zero_based",
            "cv_role",
        ]
    ].sort_values(["cv_fold", "row_index"])

    output_dir = project_path(config, "data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset_split.csv"
    folds_path = output_dir / "cv_folds.csv"
    dataset_split.to_csv(dataset_path, index=False)
    cv_folds.to_csv(folds_path, index=False)

    if len(dataset_split) != 1853:
        raise RuntimeError("dataset_split.csv must contain 1,853 compounds.")
    if dataset_split["dataset_split"].value_counts().to_dict() != {
        "development": 1482,
        "independent_test": 371,
    }:
        raise RuntimeError("Unexpected development/test counts in dataset_split.csv.")
    if len(cv_folds) != 1482 or set(cv_folds["cv_fold"]) != set(range(1, 11)):
        raise RuntimeError("cv_folds.csv must contain 1,482 development compounds in folds 1-10.")
    return {
        "dataset_split": str(dataset_path),
        "cv_folds": str(folds_path),
        "dataset_split_rows": len(dataset_split),
        "cv_folds_rows": len(cv_folds),
    }


def _selection_roles(summary: pd.DataFrame) -> dict[tuple[str, str], str]:
    development_best = summary.sort_values(
        ["best_mean_cv_roc_auc", "best_std_cv_roc_auc"], ascending=[False, True]
    ).iloc[0]
    test_best = summary.sort_values(
        ["test_roc_auc", "best_mean_cv_roc_auc"], ascending=[False, False]
    ).iloc[0]
    roles = {
        (development_best["representation"], development_best["model"]):
            "CV-selected primary model",
        (test_best["representation"], test_best["model"]):
            "Test-best descriptive model",
    }
    return roles


def export_supplementary_tables(
    config: dict[str, Any], summary: pd.DataFrame
) -> dict[str, Any]:
    """Create machine-readable source tables for Supplementary Tables S1-S2."""
    results_root = _results_root(config)
    output_dir = results_root / "supplementary_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = _selection_roles(summary)
    model_order = {name: index for index, name in enumerate(MODEL_ORDER)}
    representation_order = {
        name: index for index, name in enumerate(REPRESENTATIONS)
    }
    ordered = summary.copy()
    ordered["model_order"] = ordered["model"].map(model_order)
    ordered["representation_order"] = ordered["representation"].map(
        representation_order
    )
    ordered = ordered.sort_values(["model_order", "representation_order"])
    ordered["selection_role"] = [
        roles.get((representation, model), "Candidate model")
        for representation, model in zip(ordered["representation"], ordered["model"])
    ]

    s1_publication = pd.DataFrame(
        {
            "Model": ordered["model"].map(MODEL_LABELS),
            "Molecular representation": ordered["representation"].map(
                REPRESENTATION_LABELS
            ),
            "Selection role": ordered["selection_role"],
            "Development 10-fold CV AUC, mean ± SD": [
                f"{mean:.3f} ± {sd:.3f}"
                for mean, sd in zip(
                    ordered["best_mean_cv_roc_auc"], ordered["best_std_cv_roc_auc"]
                )
            ],
            "Independent test AUC (bootstrap 95% CI)": [
                f"{value:.3f} ({low:.3f}–{high:.3f})"
                for value, low, high in zip(
                    ordered["test_roc_auc"],
                    ordered["test_roc_auc_ci_low"],
                    ordered["test_roc_auc_ci_high"],
                )
            ],
            "ACC": ordered["test_accuracy"],
            "SE": ordered["test_sensitivity"],
            "SP": ordered["test_specificity"],
            "Precision": ordered["test_precision"],
            "F1": ordered["test_f1"],
            "MCC": ordered["test_mcc"],
            "BA": ordered["test_balanced_accuracy"],
        }
    )
    s1_publication_path = output_dir / "Supplementary_Table_S1.csv"
    s1_publication.to_csv(s1_publication_path, index=False)

    metric_names = [
        "roc_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
        "balanced_accuracy",
    ]
    s1_numeric_columns = [
        "representation",
        "representation_label",
        "model",
        "model_label",
        "selection_role",
        "best_mean_cv_roc_auc",
        "best_std_cv_roc_auc",
        "selected_candidate_id",
    ]
    for metric in metric_names:
        s1_numeric_columns.extend(
            [
                f"test_{metric}",
                f"test_{metric}_ci_low",
                f"test_{metric}_ci_high",
            ]
        )
    s1_numeric = ordered[s1_numeric_columns].copy()
    s1_numeric_path = output_dir / "Supplementary_Table_S1_full_numeric.csv"
    s1_numeric.to_csv(s1_numeric_path, index=False)

    search_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        for candidate_id, candidate in enumerate(config["models"][model_name]):
            search_rows.append(
                {
                    "Model": MODEL_LABELS[model_name],
                    "Candidate number": candidate_id + 1,
                    "Candidate ID in code": candidate_id,
                    "Prespecified configuration": json.dumps(
                        candidate, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    s2_search = pd.DataFrame(search_rows)
    s2_search_path = output_dir / "Supplementary_Table_S2A_search_spaces.csv"
    s2_search.to_csv(s2_search_path, index=False)

    optimized_rows: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        selected_path = (
            results_root
            / "tuning"
            / row["representation"]
            / row["model"]
            / "selected_params.json"
        )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        optimized_rows.append(
            {
                "Model": MODEL_LABELS[row["model"]],
                "Molecular representation": REPRESENTATION_LABELS[
                    row["representation"]
                ],
                "Selection role": roles.get(
                    (row["representation"], row["model"]), "Candidate model"
                ),
                "Selected candidate number": int(row["selected_candidate_id"]) + 1,
                "Selected candidate ID in code": int(row["selected_candidate_id"]),
                "Optimized configuration": json.dumps(
                    selected["best_params"], ensure_ascii=False, sort_keys=True
                ),
                "Development CV AUC mean": float(row["best_mean_cv_roc_auc"]),
                "Development CV AUC SD": float(row["best_std_cv_roc_auc"]),
                "DNN final refit epochs": selected.get("final_refit_epochs", ""),
                "DNN final ensemble seeds": json.dumps(
                    selected.get("final_ensemble_seeds", []), ensure_ascii=False
                ),
            }
        )
    s2_optimized = pd.DataFrame(optimized_rows)
    s2_optimized_path = output_dir / "Supplementary_Table_S2B_optimized.csv"
    s2_optimized.to_csv(s2_optimized_path, index=False)

    metadata = {
        "table_s1_caption": (
            "Supplementary Table S1. Cross-validation and independent test-set "
            "performance of conventional machine-learning and deep-neural-network "
            "nephrotoxicity prediction models."
        ),
        "table_s1_note": (
            "Model selection used mean ROC-AUC across 10 development folds. CV "
            "variability is the SD across folds. Test AUC 95% CI used 2,000 "
            "stratified bootstrap replicates. Threshold metrics used p >= 0.5."
        ),
        "table_s2_caption": (
            "Supplementary Table S2. Hyperparameter search spaces and optimized "
            "configurations of conventional machine-learning and deep-neural-network "
            "nephrotoxicity prediction models."
        ),
        "table_s2_note": (
            "The search used three prespecified candidate configurations per model, "
            "rather than a Cartesian-product grid. Candidate IDs in code are zero-based."
        ),
        "files": {
            "s1_publication": str(s1_publication_path),
            "s1_full_numeric": str(s1_numeric_path),
            "s2_search_spaces": str(s2_search_path),
            "s2_optimized": str(s2_optimized_path),
        },
    }
    write_json(metadata, output_dir / "supplementary_table_metadata.json")
    return metadata

