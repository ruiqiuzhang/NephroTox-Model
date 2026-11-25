"""Read the frozen modeling CSV while preserving its typed rows and structures."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
INTEGER_COLUMNS = ("PubChem_CID", "label", "heavy_atoms")


def rows_fingerprint(rows):
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_data(path=None, manifest_path=None):
    """Return the authenticated CSV records and the provenance of the bytes read."""
    manifest_path = Path(manifest_path) if manifest_path else ROOT / "data/source_manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(path) if path else ROOT / expected["source_file"]
    if expected["source_format"] != "csv" or path.suffix.lower() != ".csv":
        raise ValueError("Expected the release modeling CSV")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("Modeling CSV SHA-256 differs from the source manifest")

    reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
    headers = next(reader, None)
    if headers != expected["source_columns"] or len(set(headers)) != len(headers):
        raise ValueError("CSV column names or order differ from the frozen source schema")
    rows = []
    for values in reader:
        line = reader.line_num
        if len(values) != len(headers) or any(value == "" for value in values):
            raise ValueError(f"Missing or extra CSV field at line {line}")
        row = dict(zip(headers, values))
        try:
            for key in INTEGER_COLUMNS:
                row[key] = int(row[key])
            row["MW"] = float(row["MW"])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric CSV field at line {line}") from exc
        if not math.isfinite(row["MW"]) or row["MW"] <= 0 or row["heavy_atoms"] <= 0:
            raise ValueError(f"Invalid molecular weight or heavy-atom count at line {line}")
        if row["label"] not in (0, 1) or row["PubChem_CID"] <= 0:
            raise ValueError(f"Invalid binary label or PubChem CID at line {line}")
        molecule = Chem.MolFromSmiles(row["standardized_smiles"])
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"Invalid or multicomponent molecule: {row['compound_id']}")
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            raise ValueError(f"Unspecified atom in molecule: {row['compound_id']}")
        rows.append(row)

    if len(rows) != expected["n_rows"] or len({row["compound_id"] for row in rows}) != len(rows):
        raise ValueError("Population or unique compound IDs differ from the release")
    if Counter(str(row["label"]) for row in rows) != expected["label_counts"]:
        raise ValueError("Modeling CSV label counts differ from the release")
    fingerprint = rows_fingerprint(rows)
    if fingerprint != expected["rows_fingerprint"]:
        raise ValueError("Typed compound records or their order differ from the archived run")
    return rows, {**expected, "rows_fingerprint": fingerprint, "all_source_values_preserved": True}
