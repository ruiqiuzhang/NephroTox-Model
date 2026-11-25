"""Predict with one saved final model and its recorded preprocessing.

CSV input needs compound_id and standardized_smiles. Chemistry standardization
must precede inference; this script rejects mixtures and never completes stereo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
PAPER_MODELS = ROOT.parent / "reference" / "modeling"
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.features import compute_descriptors, fingerprint_matrix, load_descriptor_names


def inference_features(smiles, representation):
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    if any(
        mol is None
        or len(Chem.GetMolFrags(mol)) != 1
        or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
        for mol in molecules
    ):
        raise ValueError("Inference requires complete single-component structures with defined atoms")
    if representation == "graph":
        return np.asarray(smiles)
    if representation == "rdkit166":
        names = load_descriptor_names(ROOT / "configs/rdkit_descriptors_166.txt")
        row_ids = [str(i) for i in range(len(molecules))]
        matrix, failures = compute_descriptors(molecules, row_ids, names)
        if failures:
            raise ValueError(f"Descriptor calculation failed: {failures}")
        return matrix
    if "count" not in representation:
        return fingerprint_matrix(molecules, representation)

    generators = {
        "morgan_count2048": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
        "atompair_count1024": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
        "torsion_count1024": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True),
    }
    generator = generators[representation]()
    width = 2048 if "2048" in representation else 1024
    matrix = np.zeros((len(molecules), width), dtype=np.int32)
    for i, mol in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(mol), matrix[i])
    return matrix


def read_queries(path: str | Path) -> list[dict]:
    """Read compound IDs and standardized SMILES from a UTF-8 CSV."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def predict_records(records: list[dict], model_dir: str | Path | None = None) -> list[dict]:
    """Return probabilities and labels in input order using a saved final model."""
    if not records or any(not row.get("compound_id") or not row.get("standardized_smiles") for row in records):
        raise ValueError("Input requires compound_id and standardized_smiles for every query")
    if len({row["compound_id"] for row in records}) != len(records):
        raise ValueError("Duplicate query compound_id")

    if model_dir is None:
        selection = json.loads((PAPER_MODELS / "reports" / "primary_selection.json").read_text(encoding="utf-8"))
        model_dir = PAPER_MODELS / selection["primary_model_id"]
    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    path = directory / metadata["model_file"]
    if not metadata.get("complete") or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model is incomplete or does not match its metadata")

    matrix = inference_features([row["standardized_smiles"] for row in records], metadata["representation"])
    estimator = joblib.load(path)
    probabilities = estimator.predict_proba(matrix)[:, 1]
    threshold = 0.5
    return [
        {
            "compound_id": row["compound_id"],
            "standardized_smiles": row["standardized_smiles"],
            "model_id": metadata["model_id"],
            "probability": float(probability),
            "predicted_label": int(probability >= threshold),
            "threshold": threshold,
        }
        for row, probability in zip(records, probabilities, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Saved model directory (default: the manuscript primary model)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles", help="One standardized, single-component isomeric SMILES")
    inputs.add_argument("--input", type=Path, help="CSV containing compound_id and standardized_smiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input is not None:
        records = read_queries(args.input)
    else:
        records = [{"compound_id": "QUERY001", "standardized_smiles": args.smiles}]
    predictions = predict_records(records, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(args.output)


if __name__ == "__main__":
    main()
