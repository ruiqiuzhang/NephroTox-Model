from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from .config import project_path, write_json


def _fingerprint_matrix(molecules: list[Chem.Mol], generator: Any) -> np.ndarray:
    n_bits = int(generator.GetOptions().fpSize)
    matrix = np.zeros((len(molecules), n_bits), dtype=np.uint8)
    for row, mol in enumerate(molecules):
        vector = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        matrix[row] = vector
    return matrix


def morgan2048(molecules: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    return _fingerprint_matrix(molecules, generator)


def maccs166(molecules: list[Chem.Mol]) -> np.ndarray:
    matrix = np.zeros((len(molecules), 166), dtype=np.uint8)
    for row, mol in enumerate(molecules):
        vector = np.zeros(167, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), vector)
        matrix[row] = vector[1:]
    return matrix


def atompair1024(molecules: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=1024, includeChirality=True
    )
    return _fingerprint_matrix(molecules, generator)


def torsion1024(molecules: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=1024, includeChirality=True
    )
    return _fingerprint_matrix(molecules, generator)


def rdkit166(molecules: list[Chem.Mol], descriptor_file: Path) -> np.ndarray:
    names = [
        line.strip()
        for line in descriptor_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != 166 or len(set(names)) != 166:
        raise ValueError("The frozen descriptor file must contain 166 unique names.")
    available: dict[str, Callable[[Chem.Mol], float]] = dict(Descriptors.descList)
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Descriptors unavailable in this RDKit version: {missing}")
    matrix = np.full((len(molecules), len(names)), np.nan, dtype=np.float32)
    for row, mol in enumerate(molecules):
        for column, name in enumerate(names):
            try:
                value = float(available[name](mol))
                matrix[row, column] = value if np.isfinite(value) else np.nan
            except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
                matrix[row, column] = np.nan
    return matrix


def generate_features(config: dict[str, Any]) -> dict[str, Any]:
    data_path = project_path(config, "data/processed/model_ready.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/00_prepare_data.py before feature generation.")
    data = pd.read_csv(data_path)
    molecules = [Chem.MolFromSmiles(value) for value in data["standardized_smiles"]]
    if any(mol is None for mol in molecules):
        raise ValueError("Invalid SMILES reached the feature generation stage.")

    feature_root = project_path(config, "data/processed/features")
    feature_root.mkdir(parents=True, exist_ok=True)
    descriptor_file = project_path(
        config, config["representations"]["rdkit166"]["descriptor_file"]
    )
    builders = {
        "morgan2048": lambda: morgan2048(molecules),
        "maccs166": lambda: maccs166(molecules),
        "atompair1024": lambda: atompair1024(molecules),
        "torsion1024": lambda: torsion1024(molecules),
        "rdkit166": lambda: rdkit166(molecules, descriptor_file),
    }
    manifest: dict[str, Any] = {
        "rdkit_version": rdBase.rdkitVersion,
        "n_samples": int(len(data)),
        "representations": {},
    }
    for name, builder in builders.items():
        matrix = builder()
        if matrix.shape[0] != len(data):
            raise RuntimeError(f"Row mismatch for representation {name}.")
        output = feature_root / f"{name}.npz"
        np.savez_compressed(
            output,
            X=matrix,
            row_index=data["row_index"].to_numpy(dtype=int),
        )
        manifest["representations"][name] = {
            "path": str(output.relative_to(project_path(config, "."))),
            "shape": [int(value) for value in matrix.shape],
            "dtype": str(matrix.dtype),
            "missing_values": int(np.isnan(matrix.astype(float)).sum()),
        }
    write_json(manifest, feature_root / "feature_manifest.json")
    return manifest


def load_feature_matrix(config: dict[str, Any], representation: str) -> np.ndarray:
    path = project_path(config, f"data/processed/features/{representation}.npz")
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with np.load(path) as payload:
        matrix = payload["X"]
        row_index = payload["row_index"]
    if not np.array_equal(row_index, np.arange(len(row_index))):
        raise RuntimeError("Feature row order no longer matches the audited dataset.")
    return matrix
