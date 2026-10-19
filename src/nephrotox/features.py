"""Calculate five raw molecular representations from the supplied source rows.

The caller supplies the rows read from the DATA workbook. This module never
opens a chemical dataset, a prior feature cache, or an external database.
Specified stereochemistry and the original SMILES strings are retained. No
fragment removal, neutralisation, stereo completion, imputation, scaling or
model fitting takes place here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator


FINGERPRINT_WIDTHS = {
    "morgan2048": 2048,
    "maccs166": 166,
    "atompair1024": 1024,
    "torsion1024": 1024,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_descriptor_names(path: str | Path) -> list[str]:
    """Read the explicit frozen list; never truncate RDKit's global registry."""
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != 166 or len(set(names)) != 166:
        raise ValueError("The descriptor specification must contain 166 unique names")
    unavailable = sorted(set(names) - set(dict(Descriptors.descList)))
    if unavailable:
        raise ValueError(f"Descriptors unavailable in this RDKit version: {unavailable}")
    return names


def encode_bits(vector: np.ndarray) -> str:
    """Matrix bit zero is the leftmost character; leading zeroes are retained."""
    vector = np.asarray(vector)
    if vector.ndim != 1 or not np.isin(vector, (0, 1)).all():
        raise ValueError("A fingerprint must be a one-dimensional binary vector")
    return "".join(map(str, vector.astype(np.uint8).tolist()))


def decode_bits(bits: str, width: int) -> np.ndarray:
    if not isinstance(bits, str) or len(bits) != width or set(bits) - {"0", "1"}:
        raise ValueError(f"Expected exactly {width} binary text characters")
    return np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48


def fingerprint_matrix(molecules: list[Chem.Mol], representation: str) -> np.ndarray:
    if representation not in FINGERPRINT_WIDTHS:
        raise ValueError(f"Unknown fingerprint representation: {representation}")
    width = FINGERPRINT_WIDTHS[representation]
    matrix = np.zeros((len(molecules), width), dtype=np.uint8)
    if representation == "maccs166":
        for i, mol in enumerate(molecules):
            full = np.zeros(167, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), full)
            matrix[i] = full[1:]
    else:
        generators = {
            "morgan2048": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
            "atompair1024": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
            "torsion1024": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True),
        }
        generator = generators[representation]()
        for i, mol in enumerate(molecules):
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[i])
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"Nonbinary values produced by {representation}")
    return matrix


def compute_descriptors(molecules: list[Chem.Mol], ids: list[str], names: list[str]) -> tuple[np.ndarray, list[dict]]:
    functions = dict(Descriptors.descList)
    matrix = np.full((len(molecules), len(names)), np.nan, dtype=np.float64)
    failures = []
    with rdBase.BlockLogs():
        for i, mol in enumerate(molecules):
            for j, name in enumerate(names):
                try:
                    value = float(functions[name](mol))
                    if np.isfinite(value):
                        matrix[i, j] = value
                    else:
                        failures.append({"compound_id": ids[i], "row_index": i, "descriptor": name,
                                         "reason": "descriptor_returned_nonfinite", "value": str(value)})
                except Exception as error:
                    failures.append({"compound_id": ids[i], "row_index": i, "descriptor": name,
                                     "reason": type(error).__name__, "detail": str(error)})
    return matrix, failures


def _snapshot(mol: Chem.Mol) -> dict:
    # FindMolChiralCenters(useLegacyImplementation=False) can overwrite atom
    # _CIPCode properties. Fingerprint generators consume those properties, so
    # every diagnostic operation must run on an independent molecular copy.
    mol = Chem.Mol(mol)
    return {
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "chiral_centres": Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False),
        "specified_bonds": [(bond.GetIdx(), str(bond.GetStereo())) for bond in mol.GetBonds()
                            if bond.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)],
        "fragment_count": len(Chem.GetMolFrags(mol)),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def _numeric_hash(matrix: np.ndarray) -> str:
    """Hash canonical numerical bytes, avoiding ZIP timestamp comparisons."""
    dtype = np.dtype("<f8") if np.issubdtype(matrix.dtype, np.floating) else np.dtype("u1")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(matrix.shape), "dtype": dtype.str}, sort_keys=True).encode())
    digest.update(np.asarray(matrix, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def compute(rows: list[dict], output_dir: str | Path, descriptor_file: str | Path,
            source_info: dict | None = None) -> dict:
    """Compute all raw matrices, save aligned NPZ/JSON files, return the manifest.

    A nonfinite descriptor is reported and aborts publication; no missing value
    is replaced by zero or imputed. source_info is supplied by the DATA reader.
    Files are staged only after every representation and graph check succeeds.
    """
    if not rows:
        raise ValueError("No source records supplied")
    ids = [row["compound_id"] for row in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Source compound IDs must be nonempty and unique")
    smiles = [row["standardized_smiles"] for row in rows]
    if any(not isinstance(value, str) or not value for value in smiles):
        raise ValueError("Every source SMILES must be a nonempty string")
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    invalid = [ids[i] for i, mol in enumerate(molecules) if mol is None]
    if invalid:
        raise ValueError(f"Invalid source SMILES: {invalid}")
    before = [_snapshot(mol) for mol in molecules]
    names = load_descriptor_names(descriptor_file)
    matrices = {name: fingerprint_matrix(molecules, name) for name in FINGERPRINT_WIDTHS}
    descriptors, failures = compute_descriptors(molecules, ids, names)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if failures or not np.isfinite(descriptors).all():
        _json(output_dir / "descriptor_failures.json", {"source": source_info or {}, "failures": failures})
        raise ValueError(f"Nonfinite descriptors detected; inspect descriptor_failures.json ({len(failures)} failures)")
    matrices["rdkit166"] = descriptors
    after = [_snapshot(mol) for mol in molecules]
    if before != after:
        raise ValueError("Molecular graph or stereochemistry changed during feature calculation")
    fingerprint_rows = []
    for i, cid in enumerate(ids):
        record = {"compound_id": cid, "row_index": i}
        for name, width in FINGERPRINT_WIDTHS.items():
            record[name] = encode_bits(matrices[name][i])
            if not np.array_equal(decode_bits(record[name], width), matrices[name][i]):
                raise ValueError(f"Fingerprint round-trip failed: {cid} {name}")
        fingerprint_rows.append(record)
    descriptor_rows = [{"compound_id": cid, "row_index": i,
                        **{name: float(descriptors[i, j]) for j, name in enumerate(names)}}
                       for i, cid in enumerate(ids)]
    records_sha256 = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    manifest = {
        "n_samples": len(rows), "source": source_info or {}, "input_records_sha256": records_sha256,
        "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
        "row_index_definition": "Zero-based DATA worksheet record order; IDs and original SMILES saved in every matrix",
        "source_smiles_preserved_verbatim": True, "molecular_graph_stereo_unchanged": True,
        "preprocessing": "Raw features only; no imputation, scaling, selection or model fitting",
        "descriptor_missingness": failures, "nonfinite_descriptor_values": 0,
        "descriptor_file_sha256": _sha256(Path(descriptor_file)),
        "fingerprint_bitstrings_roundtrip_verified": True, "representations": {},
    }
    # The staging directory contains only files created by this call. Publication
    # never changes the caller's DATA workbook or any other input.
    with tempfile.TemporaryDirectory(prefix=".features-stage-", dir=output_dir) as staging:
        staging = Path(staging)
        for name, matrix in matrices.items():
            file = staging / f"{name}.npz"
            metadata = {"X": matrix, "compound_id": np.asarray(ids), "row_index": np.arange(len(ids)),
                        "standardized_smiles": np.asarray(smiles),
                        "descriptor_file_sha256": np.asarray(manifest["descriptor_file_sha256"]),
                        "source_info_json": np.asarray(json.dumps(source_info or {}, ensure_ascii=False, sort_keys=True, allow_nan=False))}
            if name == "rdkit166":
                metadata["descriptor_names"] = np.asarray(names)
            np.savez_compressed(file, **metadata)
            definition = {"file": file.name, "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                          "file_sha256": _sha256(file), "numeric_sha256": _numeric_hash(matrix),
                          "missing_values": 0}
            if name == "rdkit166":
                definition.update({"descriptor_order": names, "precision": "float64", "independent_chirality_encoding": False})
            else:
                definition.update({"include_chirality": name != "maccs166", "bit_order": "Left character is matrix bit 0; MACCS matrix bit 0 corresponds to original MACCS bit 1"})
                if name == "morgan2048":
                    definition["radius"] = 2
            manifest["representations"][name] = definition
        manifest["stereochemistry_note"] = "MACCS166 and these 166 descriptors do not independently distinguish enantiomers. Other fingerprints enable chirality but can still collide. No unspecified centre was completed."
        payload = {
            "fingerprint_headers": ["compound_id", "row_index", *FINGERPRINT_WIDTHS],
            "fingerprints": fingerprint_rows,
            "rdkit166_headers": ["compound_id", "row_index", *names], "rdkit166": descriptor_rows,
            "descriptor_missing": failures, "manifest": manifest,
        }
        _json(staging / "features_payload.json", payload)
        _json(staging / "feature_manifest.json", manifest)
        _json(staging / "stereo_retention.json", [{"compound_id": cid, **state} for cid, state in zip(ids, before)])
        for name in [*[f"{key}.npz" for key in matrices], "features_payload.json", "stereo_retention.json", "feature_manifest.json"]:
            (staging / name).replace(output_dir / name)
    (output_dir / "descriptor_failures.json").unlink(missing_ok=True)
    return manifest
