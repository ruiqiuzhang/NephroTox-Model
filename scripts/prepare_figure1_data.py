"""Recompute Figure 1 source data and seven multiplicity-corrected tests.

The sole chemical data input is the frozen DATA workbook read by
``nephrotox.data.load_data``. Source isomeric SMILES are never normalised,
de-salted, made achiral or completed. Existing feature matrices can be supplied
only as an optional consistency check, never as the source of Figure 1 values.

Run from any working directory:
    python /path/to/github提交版/scripts/prepare_figure1_data.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
import scipy
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.data import load_data  # noqa: E402


# Retain the original Figure 1 display names while defining their exact RDKit
# implementations explicitly. ALogP is the Wildman-Crippen log P estimate.
DESCRIPTORS = {
    "ALogP": ("MolLogP", "Wildman-Crippen octanol/water log P estimate", "unitless"),
    "nAR": ("NumAromaticRings", "Number of aromatic rings", "count"),
    "nHAcc": ("NumHAcceptors", "Number of hydrogen-bond acceptors", "count"),
    "nHDon": ("NumHDonors", "Number of hydrogen-bond donors", "count"),
    "nRotB": ("NumRotatableBonds", "Number of rotatable bonds; RDKit default/strict definition", "count"),
    "TPSA": ("TPSA", "Topological polar surface area; RDKit default N/O contributions", "angstrom^2"),
    "MW": ("MolWt", "Average molecular weight", "g mol^-1"),
}
IDENTITY_COLUMNS = ["compound_id", "name", "PubChem_CID", "label", "standardized_smiles"]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, preserving original test order."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P values must be a finite one-dimensional vector in [0, 1]")
    if not p_values.size:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted_ordered = np.minimum.accumulate(ordered[::-1])[::-1]
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    return adjusted


def significance(adjusted_p: float) -> str:
    if adjusted_p < 0.001:
        return "***"
    if adjusted_p < 0.01:
        return "**"
    if adjusted_p < 0.05:
        return "*"
    return "ns"


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Empty descriptor distribution")
    q1, median, q3 = np.quantile(finite, (0.25, 0.5, 0.75), method="linear")
    return {
        "n": int(len(values)), "n_finite": int(len(finite)),
        "n_missing": int(len(values) - len(finite)),
        "mean": float(np.mean(finite)), "std_ddof1": float(np.std(finite, ddof=1)),
        "median": float(median), "q1": float(q1), "q3": float(q3),
        "IQR": float(q3 - q1), "min": float(np.min(finite)), "max": float(np.max(finite)),
        "n_unique": int(len(np.unique(finite))),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    # UTF-8 BOM makes the Chinese source names readable when opened in Excel.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def check_features(path: Path, rows: list[dict], source_info: dict, matrix: np.ndarray) -> dict:
    """Compare newly calculated values against this release's raw feature file."""
    with np.load(path, allow_pickle=False) as saved:
        if "source_info_json" not in saved:
            raise ValueError("Feature consistency check has no frozen source provenance")
        cached_source = json.loads(str(saved["source_info_json"].item()))
        if cached_source["sha256"] != source_info["sha256"]:
            raise ValueError("Feature check belongs to a different source workbook")
        if saved["compound_id"].tolist() != [row["compound_id"] for row in rows]:
            raise ValueError("Feature consistency check has a different compound order")
        if saved["standardized_smiles"].tolist() != [row["standardized_smiles"] for row in rows]:
            raise ValueError("Feature consistency check has different SMILES")
        names = saved["descriptor_names"].tolist()
        result = {}
        for column, (display_name, (rdkit_name, _, _)) in enumerate(DESCRIPTORS.items()):
            cached = saved["X"][:, names.index(rdkit_name)]
            matches = np.array_equal(cached, matrix[:, column])
            result[display_name] = {
                "rdkit_descriptor": rdkit_name, "exact_float64_equality": bool(matches),
                "max_absolute_difference": float(np.max(np.abs(cached - matrix[:, column]))),
            }
            if not matches:
                raise ValueError(f"Recomputed Figure 1 values disagree with raw features: {display_name}")
    return {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "all_seven_exactly_equal": True, "descriptors": result}


def prepare(output: Path, check_feature_cache: Path | None = None) -> dict:
    rows, source_info = load_data()
    ids = [row["compound_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    molecules = [Chem.MolFromSmiles(row["standardized_smiles"]) for row in rows]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid source molecule")
    before = [Chem.MolToSmiles(molecule, isomericSmiles=True) for molecule in molecules]
    matrix = np.asarray([
        [float(getattr(Descriptors, name)(molecule)) for name, _, _ in DESCRIPTORS.values()]
        for molecule in molecules
    ], dtype=np.float64)
    if matrix.shape != (1673, 7) or not np.isfinite(matrix).all():
        raise ValueError("Expected 1,673 records and seven finite Figure 1 descriptors")
    after = [Chem.MolToSmiles(molecule, isomericSmiles=True) for molecule in molecules]
    if before != after:
        raise ValueError("Molecular graph or stereochemistry changed during descriptor calculation")

    tests, summaries = [], []
    for column, (display_name, (rdkit_name, definition, unit)) in enumerate(DESCRIPTORS.items()):
        positive = matrix[labels == 1, column]
        reference = matrix[labels == 0, column]
        # scipy's asymptotic implementation corrects the variance for tied ranks.
        # No independent-observation/effect-of-label causal claim is implied.
        test = mannwhitneyu(positive, reference, alternative="two-sided", method="asymptotic", use_continuity=True)
        delta = 2.0 * float(test.statistic) / (len(positive) * len(reference)) - 1.0
        tests.append({
            "descriptor": display_name, "rdkit_descriptor": rdkit_name,
            "positive_n": len(positive), "reference_negative_n": len(reference),
            "U_positive": float(test.statistic), "U_reference_negative": float(len(positive) * len(reference) - test.statistic),
            "p_raw": float(test.pvalue), "rank_biserial_positive_minus_reference": float(delta),
            "effect_interpretation": "P(positive > reference) - P(positive < reference); ties contribute zero",
        })
        for label, values in ((1, positive), (0, reference)):
            summaries.append({"descriptor": display_name, "rdkit_descriptor": rdkit_name,
                              "label": label, "group": "Nephrotoxic" if label else "Reference negative",
                              "definition": definition, "unit": unit, **describe(values)})
    p_adjusted = bh_adjust(np.asarray([test["p_raw"] for test in tests]))
    for test, adjusted in zip(tests, p_adjusted):
        test["p_BH_adjusted"] = float(adjusted)
        test["significance_BH"] = significance(float(adjusted))

    source_rows = [
        {**{name: row[name] for name in IDENTITY_COLUMNS},
         **{name: float(matrix[i, j]) for j, name in enumerate(DESCRIPTORS)}}
        for i, row in enumerate(rows)
    ]
    extrema = []
    for column, descriptor in enumerate(DESCRIPTORS):
        for tail, index in (("minimum", int(np.argmin(matrix[:, column]))),
                            ("maximum", int(np.argmax(matrix[:, column])))):
            extrema.append({"descriptor": descriptor, "tail": tail,
                            **{name: rows[index][name] for name in IDENTITY_COLUMNS[:-1]},
                            "value": float(matrix[index, column])})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "figure1_compound_descriptors.csv", source_rows, [*IDENTITY_COLUMNS, *DESCRIPTORS])
    write_csv(output / "figure1_statistical_tests.csv", tests, list(tests[0]))
    write_csv(output / "figure1_group_summaries.csv", summaries, list(summaries[0]))
    write_csv(output / "figure1_extrema.csv", extrema, list(extrema[0]))
    consistency = check_features(check_feature_cache.resolve(), rows, source_info, matrix) if check_feature_cache else None
    metadata = {
        "source": source_info, "n_records": len(rows),
        "label_counts": {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())},
        "positive_label": 1, "reference_negative_label": 0,
        "label_interpretation": "Group labels are the supplied DATA annotations; reference negatives are not a claim of clinical absence of nephrotoxicity.",
        "analysis_population": "All 1673 final DATA records, before development/test partition; no model fitting or feature selection.",
        "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "descriptor_definitions": {name: {"rdkit_function": "rdkit.Chem.Descriptors." + rdkit_name,
                                          "definition": definition, "unit": unit}
                                   for name, (rdkit_name, definition, unit) in DESCRIPTORS.items()},
        "ALogP_display_name_note": "The inherited panel label ALogP denotes RDKit MolLogP (Wildman-Crippen), not a separate descriptor algorithm.",
        "source_isomeric_smiles_preserved_verbatim": True, "graph_and_stereo_unchanged": True,
        "nonfinite_descriptor_values": 0, "excluded_records": [],
        "test": {"name": "Mann-Whitney U", "alternative": "two-sided", "method": "asymptotic",
                 "tie_corrected": True, "use_continuity": True, "positive_is_first_sample": True,
                 "multiple_testing": "Benjamini-Hochberg FDR across all seven descriptor tests",
                 "significance": {"***": "adjusted P < 0.001", "**": "0.001 <= adjusted P < 0.01",
                                  "*": "0.01 <= adjusted P < 0.05", "ns": "adjusted P >= 0.05"},
                 "scope_note": "Descriptive comparisons of supplied label groups; no multiplicity-uncorrected significance or causal inference."},
        "summary_quantiles": "numpy.quantile method='linear'; sample SD uses ddof=1",
        "plot_integrity": "Use the full ranges in figure1_group_summaries.csv; do not clip high-MW/polar compounds to legacy figure limits.",
        "optional_current_release_feature_check": consistency,
        "numeric_values_sha256": hashlib.sha256(np.asarray(matrix, dtype="<f8").tobytes(order="C")).hexdigest(),
    }
    write_json(output / "figure1_source_data.json", {"metadata": metadata, "records": source_rows,
                                                   "statistical_tests": tests, "group_summaries": summaries,
                                                   "extrema": extrema})
    write_json(output / "figure1_analysis_manifest.json", metadata)
    print(json.dumps({"output": str(output), "n": len(rows), "label_counts": metadata["label_counts"],
                      "missing": 0, "tests": tests,
                      "ranges": {name: {"min": float(np.min(matrix[:, j])), "max": float(np.max(matrix[:, j]))}
                                 for j, name in enumerate(DESCRIPTORS)}}, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/figure1/source_data")
    parser.add_argument("--check-feature-cache", type=Path, default=None,
                        help="Optional current DATA raw rdkit166.npz to check after direct recomputation")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.check_feature_cache)


if __name__ == "__main__":
    main()
