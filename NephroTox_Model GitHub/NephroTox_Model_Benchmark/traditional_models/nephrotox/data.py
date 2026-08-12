from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from .config import project_path, write_json


REQUIRED_COLUMNS = [
    "compound_id",
    "name",
    "label",
    "standardized_smiles",
    "standardized_InChIKey",
    "connectivity_key",
    "MW",
    "heavy_atoms",
    "radical_electrons",
    "QC_status",
    "source_excel_row",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold if scaffold else "__ACYCLIC__"


def load_and_validate_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_cfg = config["dataset"]
    workbook = project_path(config, dataset_cfg["workbook"])
    actual_hash = sha256_file(workbook)
    expected_hash = dataset_cfg["expected_sha256"].lower()
    if actual_hash.lower() != expected_hash:
        raise ValueError(
            "Dataset hash mismatch. Refusing to run on an unreviewed workbook: "
            f"expected {expected_hash}, observed {actual_hash}."
        )

    frame = pd.read_excel(workbook, sheet_name=dataset_cfg["sheet"])
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) != int(dataset_cfg["expected_rows"]):
        raise ValueError(
            f"Expected {dataset_cfg['expected_rows']} rows, observed {len(frame)}."
        )
    if frame[REQUIRED_COLUMNS].isna().any().any():
        bad = frame[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"Missing values detected:\n{bad[bad > 0]}")
    if set(frame["label"].astype(int).unique()) != {0, 1}:
        raise ValueError("The label column must contain exactly binary values {0, 1}.")
    if not frame["QC_status"].astype(str).eq("PASS").all():
        raise ValueError("At least one row did not pass source quality control.")

    unique_fields = [
        "compound_id",
        "standardized_smiles",
        "standardized_InChIKey",
        "connectivity_key",
    ]
    duplicate_counts = {
        column: int(frame[column].duplicated().sum()) for column in unique_fields
    }
    if any(duplicate_counts.values()):
        raise ValueError(f"Duplicate identifiers detected: {duplicate_counts}")

    molecules = [Chem.MolFromSmiles(value) for value in frame["standardized_smiles"]]
    invalid_rows = [index for index, mol in enumerate(molecules) if mol is None]
    if invalid_rows:
        raise ValueError(f"Invalid standardized SMILES at zero-based rows: {invalid_rows}")

    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    result["label"] = result["label"].astype(np.int8)
    result["murcko_scaffold"] = [
        murcko_scaffold(smiles, include_chirality=False)
        for smiles in result["standardized_smiles"]
    ]
    chiral_scaffolds = [
        murcko_scaffold(smiles, include_chirality=True)
        for smiles in result["standardized_smiles"]
    ]
    result["scaffold_group"] = [
        f"ACYCLIC::{compound_id}" if scaffold == "__ACYCLIC__" else scaffold
        for scaffold, compound_id in zip(chiral_scaffolds, result["compound_id"])
    ]
    result.insert(0, "row_index", np.arange(len(result), dtype=int))

    audit = {
        "workbook": str(workbook),
        "sheet": dataset_cfg["sheet"],
        "sha256": actual_hash,
        "n_rows": int(len(result)),
        "label_counts": {
            str(key): int(value)
            for key, value in result["label"].value_counts().sort_index().items()
        },
        "n_unique_scaffolds": int(result["murcko_scaffold"].nunique()),
        "n_scaffold_split_groups": int(result["scaffold_group"].nunique()),
        "n_acyclic": int(result["murcko_scaffold"].eq("__ACYCLIC__").sum()),
        "duplicate_counts": duplicate_counts,
        "invalid_smiles": 0,
    }
    return result, audit


def _assignment_frame(
    data: pd.DataFrame,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    folds: np.ndarray,
    n_splits: int,
) -> pd.DataFrame:
    assignment = data[
        ["row_index", "compound_id", "label", "murcko_scaffold", "scaffold_group"]
    ].copy()
    assignment["partition"] = "test"
    assignment["fold"] = -1
    assignment.loc[development_indices, "partition"] = "development"
    assignment.loc[development_indices, "fold"] = folds.astype(int)
    assignment = assignment.sort_values("row_index").reset_index(drop=True)
    if set(assignment.loc[development_indices, "fold"].unique()) != set(range(n_splits)):
        raise RuntimeError(f"Expected {n_splits} development folds.")
    if len(set(development_indices).intersection(set(test_indices))):
        raise RuntimeError("Development/test overlap detected.")
    return assignment


def make_random_split(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seeds"]["global"])
    test_fraction = float(config["validation"]["test_fraction"])
    n_splits = int(config["validation"]["cv_folds"])
    indices = np.arange(len(data))
    development, test = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=data["label"].to_numpy(),
    )
    development = np.sort(development)
    test = np.sort(test)
    folds = np.full(len(development), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_dev = data.loc[development, "label"].to_numpy()
    for fold, (_, validation_position) in enumerate(splitter.split(development, y_dev)):
        folds[validation_position] = fold
    return _assignment_frame(data, development, test, folds, n_splits)


def make_scaffold_split(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    holdout_seed = int(config["seeds"]["scaffold_holdout"])
    test_fraction = float(config["validation"]["test_fraction"])
    n_splits = int(config["validation"]["cv_folds"])
    indices = np.arange(len(data))
    groups = data["scaffold_group"].to_numpy()
    holdout = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=holdout_seed
    )
    development_position, test_position = next(
        holdout.split(indices, data["label"].to_numpy(), groups)
    )
    development = np.sort(indices[development_position])
    test = np.sort(indices[test_position])

    folds = np.full(len(development), -1, dtype=int)
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=holdout_seed
    )
    y_dev = data.loc[development, "label"].to_numpy()
    group_dev = data.loc[development, "scaffold_group"].to_numpy()
    for fold, (_, validation_position) in enumerate(
        cv.split(development, y_dev, group_dev)
    ):
        folds[validation_position] = fold
    assignment = _assignment_frame(data, development, test, folds, n_splits)

    dev_scaffolds = set(
        assignment.loc[assignment["partition"].eq("development"), "scaffold_group"]
    )
    test_scaffolds = set(
        assignment.loc[assignment["partition"].eq("test"), "scaffold_group"]
    )
    if dev_scaffolds.intersection(test_scaffolds):
        raise RuntimeError("Scaffold leakage detected across development and test.")
    for fold in range(n_splits):
        val_scaffolds = set(
            assignment.loc[assignment["fold"].eq(fold), "scaffold_group"]
        )
        other_scaffolds = set(
            assignment.loc[
                assignment["partition"].eq("development")
                & assignment["fold"].ne(fold),
                "scaffold_group",
            ]
        )
        if val_scaffolds.intersection(other_scaffolds):
            raise RuntimeError(f"Scaffold leakage detected in CV fold {fold}.")
    return assignment


def summarize_split(assignment: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in ["development", "test"]:
        subset = assignment.loc[assignment["partition"].eq(partition)]
        summary[partition] = {
            "n": int(len(subset)),
            "label_0": int(subset["label"].eq(0).sum()),
            "label_1": int(subset["label"].eq(1).sum()),
            "unique_scaffolds": int(subset["scaffold_group"].nunique()),
        }
    summary["cv_folds"] = {
        str(fold): {
            "n": int(assignment["fold"].eq(fold).sum()),
            "label_0": int(
                (assignment["fold"].eq(fold) & assignment["label"].eq(0)).sum()
            ),
            "label_1": int(
                (assignment["fold"].eq(fold) & assignment["label"].eq(1)).sum()
            ),
        }
        for fold in sorted(assignment.loc[assignment["fold"].ge(0), "fold"].unique())
    }
    return summary


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    data, audit = load_and_validate_dataset(config)
    processed_path = project_path(config, "data/processed/model_ready.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)

    split_summaries: dict[str, Any] = {}
    for name, assignment in {
        "random": make_random_split(data, config),
        "scaffold": make_scaffold_split(data, config),
    }.items():
        output = project_path(config, f"data/splits/{name}_holdout.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_csv(output, index=False)
        split_summaries[name] = summarize_split(assignment)

    manifest = {"dataset_audit": audit, "splits": split_summaries}
    write_json(manifest, project_path(config, "data/processed/data_manifest.json"))
    return manifest


def _write_immutable_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a protocol file once and reject later content drift."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        try:
            pd.testing.assert_frame_equal(
                existing,
                frame.reset_index(drop=True),
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as error:
            raise RuntimeError(
                f"Locked protocol file differs from the requested content: {path}"
            ) from error
        return
    frame.to_csv(path, index=False)


def prepare_locked_scaffold_data(config: dict[str, Any]) -> dict[str, Any]:
    """Create the immutable scaffold10 protocol without touching legacy splits.

    The development table contains labels and is the only label-bearing table used
    by tuning. Test metadata and labels are stored separately; the sealed label file
    is opened only by the post-lock evaluation stage.
    """
    run = config["run"]
    data, audit = load_and_validate_dataset(config)
    _write_immutable_csv(
        data, project_path(config, "data/processed/model_ready.csv")
    )
    assignment = make_scaffold_split(data, config)
    summary = summarize_split(assignment)
    target_test_size = int(config["validation"]["target_test_size"])
    if summary["test"]["n"] != target_test_size:
        raise RuntimeError(
            f"Expected exactly {target_test_size} locked test compounds, "
            f"observed {summary['test']['n']}."
        )

    development_assignment = assignment.loc[
        assignment["partition"].eq("development"), ["row_index", "fold"]
    ]
    development = data.merge(
        development_assignment, on="row_index", how="inner", validate="one_to_one"
    ).sort_values("row_index")
    test_rows = assignment.loc[
        assignment["partition"].eq("test"), "row_index"
    ].to_numpy(dtype=int)
    test_features = data.loc[test_rows].drop(columns=["label"]).sort_values("row_index")
    test_labels = data.loc[test_rows, ["row_index", "compound_id", "label"]].sort_values(
        "row_index"
    )
    public_assignment = assignment.drop(columns=["label"])

    split_path = project_path(config, run["split_file"])
    development_path = project_path(config, run["development_file"])
    test_features_path = project_path(config, run["test_features_file"])
    test_labels_path = project_path(config, run["sealed_test_labels_file"])
    _write_immutable_csv(public_assignment, split_path)
    _write_immutable_csv(development, development_path)
    _write_immutable_csv(test_features, test_features_path)
    _write_immutable_csv(test_labels, test_labels_path)

    dev_groups = set(development["scaffold_group"])
    test_groups = set(test_features["scaffold_group"])
    fold_overlap = {}
    for fold in range(int(config["validation"]["cv_folds"])):
        validation_groups = set(development.loc[development["fold"].eq(fold), "scaffold_group"])
        training_groups = set(development.loc[development["fold"].ne(fold), "scaffold_group"])
        fold_overlap[str(fold)] = len(validation_groups.intersection(training_groups))

    manifest = {
        "run_id": run["id"],
        "dataset_audit": audit,
        "split_method": "GroupShuffleSplit by chirality-aware Bemis-Murcko scaffold group",
        "acyclic_group_rule": "one unique scaffold group per acyclic compound",
        "split_seed": int(config["seeds"]["scaffold_holdout"]),
        "seed_selection": {
            "search_range": [20260810, 20310809],
            "selection_rule": (
                "lexicographically minimize absolute test-size error, absolute "
                "test-prevalence error, then seed; model outcomes were not computed"
            ),
            "target_test_size": target_test_size,
        },
        "cv_method": "StratifiedGroupKFold",
        "cv_folds": int(config["validation"]["cv_folds"]),
        "summary": summary,
        "scaffold_overlap": {
            "development_vs_test": len(dev_groups.intersection(test_groups)),
            "training_vs_validation_by_fold": fold_overlap,
        },
        "files": {
            "split": {"path": run["split_file"], "sha256": sha256_file(split_path)},
            "development": {
                "path": run["development_file"],
                "sha256": sha256_file(development_path),
            },
            "test_features": {
                "path": run["test_features_file"],
                "sha256": sha256_file(test_features_path),
            },
            "sealed_test_labels": {
                "path": run["sealed_test_labels_file"],
                "sha256": sha256_file(test_labels_path),
            },
        },
    }
    if manifest["scaffold_overlap"]["development_vs_test"] != 0 or any(
        fold_overlap.values()
    ):
        raise RuntimeError("Scaffold leakage detected in the locked protocol.")
    manifest_path = project_path(config, run["protocol_manifest"])
    write_json(manifest, manifest_path)
    return manifest


def prepare_locked_random_data(config: dict[str, Any]) -> dict[str, Any]:
    """Create the immutable stratified-random 80:20 protocol.

    The independent test labels are written to a separate sealed file. Model
    tuning reads only the development table and therefore cannot use test labels.
    """
    run = config["run"]
    data, audit = load_and_validate_dataset(config)
    _write_immutable_csv(
        data, project_path(config, "data/processed/model_ready.csv")
    )
    assignment = make_random_split(data, config)
    summary = summarize_split(assignment)
    target_test_size = int(config["validation"]["target_test_size"])
    if summary["test"]["n"] != target_test_size:
        raise RuntimeError(
            f"Expected exactly {target_test_size} locked test compounds, "
            f"observed {summary['test']['n']}."
        )

    development_assignment = assignment.loc[
        assignment["partition"].eq("development"), ["row_index", "fold"]
    ]
    development = data.merge(
        development_assignment, on="row_index", how="inner", validate="one_to_one"
    ).sort_values("row_index")
    test_rows = assignment.loc[
        assignment["partition"].eq("test"), "row_index"
    ].to_numpy(dtype=int)
    test_features = data.loc[test_rows].drop(columns=["label"]).sort_values("row_index")
    test_labels = data.loc[test_rows, ["row_index", "compound_id", "label"]].sort_values(
        "row_index"
    )
    public_assignment = assignment.drop(columns=["label"])

    split_path = project_path(config, run["split_file"])
    development_path = project_path(config, run["development_file"])
    test_features_path = project_path(config, run["test_features_file"])
    test_labels_path = project_path(config, run["sealed_test_labels_file"])
    _write_immutable_csv(public_assignment, split_path)
    _write_immutable_csv(development, development_path)
    _write_immutable_csv(test_features, test_features_path)
    _write_immutable_csv(test_labels, test_labels_path)

    dev_groups = set(development["scaffold_group"])
    test_groups = set(test_features["scaffold_group"])
    manifest = {
        "run_id": run["id"],
        "dataset_audit": audit,
        "split_method": "stratified random train_test_split",
        "split_ratio": {"development": 0.80, "independent_test": 0.20},
        "split_seed": int(config["seeds"]["global"]),
        "cv_method": "StratifiedKFold within development",
        "cv_folds": int(config["validation"]["cv_folds"]),
        "summary": summary,
        "chemical_space_audit": {
            "development_test_scaffold_group_overlap": len(
                dev_groups.intersection(test_groups)
            ),
            "note": (
                "Scaffold overlap is reported for transparency but is not constrained "
                "under the prespecified stratified-random protocol."
            ),
        },
        "files": {
            "split": {"path": run["split_file"], "sha256": sha256_file(split_path)},
            "development": {
                "path": run["development_file"],
                "sha256": sha256_file(development_path),
            },
            "test_features": {
                "path": run["test_features_file"],
                "sha256": sha256_file(test_features_path),
            },
            "sealed_test_labels": {
                "path": run["sealed_test_labels_file"],
                "sha256": sha256_file(test_labels_path),
            },
        },
    }
    write_json(manifest, project_path(config, run["protocol_manifest"]))
    return manifest
