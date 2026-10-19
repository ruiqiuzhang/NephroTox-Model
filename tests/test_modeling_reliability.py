"""Tests for fold-honest AD, development-only associations and locked reports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rdkit import DataStructs
from scipy.stats import fisher_exact

from nephrotox.modeling_reliability import (
    applicability_domain, bh_adjust, group_bootstrap_ci, label_cliff_pairs,
    report_features, structural_key_associations, tanimoto_matrix,
)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"compound_id": f"S{i:03}", "name": f"synthetic {i}", "PubChem_CID": i + 1,
                      "label": i % 2, "standardized_smiles": "CCO"} for i in range(24)]
        self.assignments = [{"compound_id": r["compound_id"], "dataset_partition": "development" if i < 20 else "test",
                             "cv_validation_fold": i // 2 + 1 if i < 20 else None,
                             "structure_group_id": f"G{i // 2}"} for i, r in enumerate(self.rows)]
        self.matrix = np.random.default_rng(47).integers(0, 2, size=(24, 16), dtype=np.uint8)

    def test_binary_tanimoto_matches_rdkit_and_empty_vector_convention(self):
        x = np.vstack([np.zeros(16, np.uint8), self.matrix[:4]])
        result = tanimoto_matrix(x)
        fps = []
        for row in x:
            fp = DataStructs.ExplicitBitVect(len(row))
            for index in np.flatnonzero(row):
                fp.SetBit(int(index))
            fps.append(fp)
        expected = np.asarray([[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps])
        np.testing.assert_allclose(result, expected)
        with self.assertRaises(ValueError):
            tanimoto_matrix(np.ones((5, 8)) * 2)

    def test_ad_test_never_enters_development_thresholds(self):
        before = applicability_domain(self.rows, self.assignments, self.matrix)
        modified = self.matrix.copy()
        modified[20:] = 1
        after = applicability_domain(self.rows, self.assignments, modified)
        self.assertEqual(before["thresholds"], after["thresholds"])
        for record in before["records"]:
            a = int(record["compound_id"][1:])
            b = int(record["nearest_reference_id"][1:])
            self.assertNotEqual(self.assignments[a]["structure_group_id"], self.assignments[b]["structure_group_id"])
        self.assertEqual(len(before["records"]), 24)
        self.assertEqual(len(before["thresholds"]), 11)

    def test_oof_query_features_cannot_change_own_fold_threshold(self):
        before = applicability_domain(self.rows, self.assignments, self.matrix)
        modified = self.matrix.copy()
        modified[:2] = 0
        after = applicability_domain(self.rows, self.assignments, modified)
        first_before = next(r for r in before["thresholds"] if r["fold"] == 1)
        first_after = next(r for r in after["thresholds"] if r["fold"] == 1)
        self.assertEqual(first_before, first_after)
        for record in before["development_loo"]:
            self.assertNotEqual(record["compound_id"], record["nearest_other_development_id"])

    def test_structural_group_leakage_is_rejected(self):
        assignments = [dict(r) for r in self.assignments]
        assignments[-1]["structure_group_id"] = assignments[0]["structure_group_id"]
        with self.assertRaises(ValueError):
            applicability_domain(self.rows, assignments, self.matrix)

    def test_maccs_fisher_family_166_and_no_test_label_use(self):
        maccs = np.zeros((24, 166), dtype=np.uint8)
        maccs[:, 1] = [r["label"] for r in self.rows]
        before = structural_key_associations(self.rows, self.assignments, maccs)
        modified = [dict(row) for row in self.rows]
        for row in modified[20:]:
            row["label"] = 1 - row["label"]
        after = structural_key_associations(modified, self.assignments, maccs)
        self.assertEqual(before, after)
        self.assertEqual(len(before), 166)
        self.assertTrue(all(r["family_size"] == 166 for r in before))
        expected = fisher_exact([[10, 0], [0, 10]], alternative="two-sided").pvalue
        self.assertAlmostEqual(before[1]["fisher_p"], expected)
        self.assertAlmostEqual(before[1]["q_BH"], expected * 166)
        self.assertEqual(before[0]["fisher_p"], 1)
        self.assertTrue(before[0]["constant_key"])

    def test_bh_handles_ties_and_original_order(self):
        values = bh_adjust([.05, .001, .02, .02, 1])
        np.testing.assert_allclose(values, [.0625, .005, .0333333333333, .0333333333333, 1])

    def test_discordant_pairs_are_current_unique_and_sorted(self):
        similarity = np.eye(24)
        similarity[0, 1] = similarity[1, 0] = .9
        similarity[2, 3] = similarity[3, 2] = .95
        similarity[0, 2] = similarity[2, 0] = .99  # same label: excluded
        pairs = label_cliff_pairs(self.rows, similarity)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["positive_id"], "S003")
        self.assertEqual(pairs[0]["reference_negative_id"], "S002")
        self.assertEqual(pairs[0]["positive_CID"], 4)

    def test_group_bootstrap_is_reproducible_and_preserves_perfect_prediction(self):
        y = np.tile([0, 1], 8)
        p = np.tile([.1, .9], 8)
        groups = np.repeat(np.arange(8), 2)
        first = group_bootstrap_ci(y, p, groups, n_bootstrap=25, seed=19)
        self.assertEqual(first, group_bootstrap_ci(y, p, groups, n_bootstrap=25, seed=19))
        self.assertEqual(first["ROC_AUC_lower"], 1)
        self.assertEqual(first["ROC_AUC_upper"], 1)
        self.assertEqual(first["error_rate_upper"], 0)

    def test_missing_global_lock_prevents_any_reliability_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "global model lock"):
                report_features(self.rows, self.assignments, {}, directory)
            self.assertFalse((Path(directory) / "reliability").exists())

    def test_report_exports_current_aligned_sources_and_counts(self):
        ids = [r["compound_id"] for r in self.rows]
        dev, test = np.arange(20), np.arange(20, 24)
        metadata = {"descriptor_rf": {"algorithm": "RF", "representation": "rdkit166", "role": "main"}}
        pdev, ptest = np.tile([.2, .8], 10), np.tile([.2, .8], 2)
        bundle = {"assignments": self.assignments, "metadata": metadata,
                  "predictions": {"descriptor_rf": {"development": pdev, "test": ptest}},
                  "development_index": dev, "test_index": test,
                  "folds": np.repeat(np.arange(1, 11), 2),
                  "groups": np.asarray([r["structure_group_id"] for r in self.assignments]),
                  "lock": {"source_rows_fingerprint": "synthetic unit test"}}
        matrix = np.random.default_rng(41).integers(0, 2, size=(24, 2048), dtype=np.uint8)
        # One discordant exact-fingerprint pair exercises pair probability joins.
        matrix[1] = matrix[0]
        maccs = np.zeros((24, 166), np.uint8)
        with tempfile.TemporaryDirectory() as directory, patch(
                "nephrotox.modeling_reliability.load_locked_predictions", return_value=bundle):
            folder = Path(directory)
            (folder / "model_lock.json").write_text("synthetic test fixture")
            (folder / "reports").mkdir()
            (folder / "reports/primary_selection.json").write_text(json.dumps({
                "primary_model_id": "descriptor_rf", "test_used_for_selection": False,
                "mean_outer_auc": {"descriptor_rf": 1.0}}))
            result = report_features(self.rows, self.assignments, {"morgan2048": matrix, "maccs166": maccs},
                                     folder, n_bootstrap=5)
            self.assertEqual(result["n_development"], 20)
            self.assertEqual(result["n_test"], 4)
            self.assertEqual(result["n_label_cliff_pairs"], 1)
            import csv
            with (folder / "reliability/label_cliff_predictions.csv").open(encoding="utf-8-sig") as stream:
                pair = next(csv.DictReader(stream))
            self.assertEqual(pair["positive_id"], ids[1])
            self.assertEqual(pair["reference_negative_id"], ids[0])
            self.assertEqual(pair["positive_prediction_source"], "development_oof")
            self.assertEqual(pair["positive_probability"], "0.8")


if __name__ == "__main__":
    unittest.main()
