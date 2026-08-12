from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
BANNED_SUFFIXES = {".pt", ".joblib", ".tif", ".tiff"}

REQUIRED = [
    ROOT / "README.md",
    ROOT / "tables" / "model_performance_independent_test.csv",
    ROOT / "tables" / "model_inventory.csv",
    ROOT / "tables" / "NephroTox_Model_Tables.xlsx",
    ROOT / "traditional_models" / "pyproject.toml",
    ROOT / "traditional_models" / "nephrotox" / "__init__.py",
    ROOT / "traditional_models" / "results" / "random10_locked" / "summary" / "all_metrics_with_ci.csv",
    ROOT / "molecular_graph_models" / "results" / "test_evaluation" / "independent_test_metrics_complete.csv",
    ROOT / "RELEASE_MANIFEST_SHA256.csv",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    banned = [str(path.relative_to(ROOT)) for path in files if path.suffix.lower() in BANNED_SUFFIXES]
    if banned:
        raise SystemExit(f"Excluded artifact types are present: {banned}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in files
        if path.stat().st_size >= MAX_GITHUB_FILE_BYTES
    ]
    if oversized:
        raise SystemExit(f"Files at or above GitHub's 100 MB limit: {oversized}")

    for path in files:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8-sig") as handle:
                json.load(handle)
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                list(csv.reader(handle))

    comparison = ROOT / "tables" / "model_performance_independent_test.csv"
    with comparison.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 29:
        raise SystemExit(f"Expected 29 normalized independent-test rows, found {len(rows)}")

    families = {row["model_family"] for row in rows}
    if families != {"traditional", "molecular_graph"}:
        raise SystemExit(f"Unexpected model families: {sorted(families)}")

    manifest_path = ROOT / "RELEASE_MANIFEST_SHA256.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["path"]: row for row in manifest_rows}
    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in files
        if path != manifest_path
    }
    if set(manifest) != expected_paths:
        missing_from_manifest = sorted(expected_paths - set(manifest))
        stale_manifest_rows = sorted(set(manifest) - expected_paths)
        raise SystemExit(
            "Manifest path mismatch: "
            f"missing={missing_from_manifest}, stale={stale_manifest_rows}"
        )

    for relative, row in manifest.items():
        path = ROOT / relative
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if size != int(row["size_bytes"]) or digest != row["sha256"].lower():
            raise SystemExit(f"Manifest mismatch: {relative}")

    print(
        f"Release validation passed: {len(files)} files, "
        f"{len(rows)} normalized model rows, {len(manifest)} verified checksums"
    )


if __name__ == "__main__":
    main()
