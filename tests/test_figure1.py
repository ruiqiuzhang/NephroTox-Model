"""Lightweight Figure 1 statistics and current DATA integrity checks.

Only the frozen current DATA workbook is read. No precomputed Figure 1 file,
historical dataset, descriptor matrix, split cache or external service is
required. The seven Figure 1 descriptors are computed once for this test class.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import false_discovery_control, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("prepare_figure1_data", ROOT / "scripts/prepare_figure1_data.py")
figure1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure1)


class Figure1StatisticalTests(unittest.TestCase):
    def test_BH_matches_independent_scipy_for_unsorted_values(self):
        values = np.asarray([0.01, 0.04, 0.03, 0.002, 0.60, 0.21, 0.9])
        np.testing.assert_allclose(figure1.bh_adjust(values), false_discovery_control(values, method="bh"),
                                   rtol=1e-15, atol=0)

    def test_BH_ties_extremes_and_empty_input(self):
        values = np.asarray([0.05, 1.0, 0.0, 1e-300, 0.05, 0.05, 1.0])
        actual = figure1.bh_adjust(values)
        np.testing.assert_allclose(actual, false_discovery_control(values, method="bh"), rtol=1e-15, atol=0)
        self.assertEqual(actual[0], actual[4])
        self.assertEqual(actual[4], actual[5])
        self.assertTrue(np.all(actual >= values))
        np.testing.assert_array_equal(figure1.bh_adjust(np.asarray([])), np.asarray([]))

    def test_BH_rejects_invalid_probability_vectors(self):
        for values in ([np.nan, 0.01], [np.inf], [-0.01], [1.01], [[0.05, 0.1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                figure1.bh_adjust(np.asarray(values))

    def test_adjusted_P_star_threshold_boundaries(self):
        for value, expected in ((0.0, "***"), (np.nextafter(0.001, 0), "***"),
                                (0.001, "**"), (np.nextafter(0.01, 0), "**"),
                                (0.01, "*"), (np.nextafter(0.05, 0), "*"),
                                (0.05, "ns"), (1.0, "ns")):
            with self.subTest(value=value):
                self.assertEqual(figure1.significance(value), expected)

    def test_tied_MWU_group_exchange_preserves_P_and_reverses_direction(self):
        positive = np.asarray([1, 1, 3, 5, 7, 7, 7], dtype=np.float64)
        reference = np.asarray([0, 1, 1, 2, 3, 3], dtype=np.float64)
        options = {"alternative": "two-sided", "method": "asymptotic", "use_continuity": True}
        forward = mannwhitneyu(positive, reference, **options)
        backward = mannwhitneyu(reference, positive, **options)
        pairs = len(positive) * len(reference)
        self.assertEqual(forward.pvalue, backward.pvalue)
        self.assertEqual(forward.statistic + backward.statistic, pairs)
        forward_effect = 2 * forward.statistic / pairs - 1
        backward_effect = 2 * backward.statistic / pairs - 1
        self.assertAlmostEqual(forward_effect, -backward_effect, places=15)
        # Pairwise counting independently verifies the rank-biserial direction.
        pairwise_effect = np.mean(np.sign(positive[:, None] - reference[None, :]))
        self.assertAlmostEqual(forward_effect, pairwise_effect, places=15)


class Figure1CurrentDATATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.info = figure1.load_data()
        cls.before = copy.deepcopy(cls.rows)
        cls.workbook = ROOT / cls.info["source_file"]
        cls.workbook_sha256_before = hashlib.sha256(cls.workbook.read_bytes()).hexdigest()
        cls.temporary = tempfile.TemporaryDirectory(prefix="nephrotox-current-DATA-figure1-")
        cls.output = Path(cls.temporary.name)
        # The real reader has already enforced the current DATA hash/schema.
        # Supplying these same objects lets us detect any source-field mutation.
        with patch.object(figure1, "load_data", return_value=(cls.rows, cls.info)):
            with contextlib.redirect_stdout(io.StringIO()):
                cls.metadata = figure1.prepare(cls.output)
        cls.payload = json.loads((cls.output / "figure1_source_data.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_current_DATA_population_and_all_source_fields_unchanged(self):
        self.assertEqual(len(self.rows), 1673)
        self.assertEqual(sum(row["label"] == 1 for row in self.rows), 842)
        self.assertEqual(sum(row["label"] == 0 for row in self.rows), 831)
        self.assertEqual(len(self.rows[0]), 12)
        self.assertEqual(self.rows, self.before)
        self.assertEqual(hashlib.sha256(self.workbook.read_bytes()).hexdigest(), self.workbook_sha256_before)
        self.assertTrue(self.metadata["source_isomeric_smiles_preserved_verbatim"])
        self.assertTrue(self.metadata["graph_and_stereo_unchanged"])
        self.assertIsNone(self.metadata["optional_current_release_feature_check"])
        for original, record in zip(self.before, self.payload["records"]):
            for field in figure1.IDENTITY_COLUMNS:
                with self.subTest(compound_id=original["compound_id"], field=field):
                    self.assertEqual(record[field], original[field])

    def test_all_seven_values_finite_and_counts_integral_in_valid_ranges(self):
        descriptors = list(figure1.DESCRIPTORS)
        values = np.asarray([[record[name] for name in descriptors] for record in self.payload["records"]])
        self.assertEqual(values.shape, (1673, 7))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(self.metadata["nonfinite_descriptor_values"], 0)
        self.assertEqual(self.metadata["excluded_records"], [])
        heavy_atoms = np.asarray([row["heavy_atoms"] for row in self.before])
        for descriptor in ("nAR", "nHAcc", "nHDon", "nRotB"):
            column = values[:, descriptors.index(descriptor)]
            with self.subTest(descriptor=descriptor):
                np.testing.assert_array_equal(column, np.floor(column))
                self.assertTrue(np.all(column >= 0))
                self.assertTrue(np.all(column <= heavy_atoms))
        self.assertTrue(np.all(values[:, descriptors.index("TPSA")] >= 0))
        self.assertTrue(np.all(values[:, descriptors.index("MW")] > 0))
        self.assertEqual(self.metadata["descriptor_definitions"]["ALogP"]["rdkit_function"],
                         "rdkit.Chem.Descriptors.MolLogP")

    def test_current_DATA_tests_have_correct_MWU_direction_and_BH_adjustment(self):
        tests = self.payload["statistical_tests"]
        self.assertEqual(len(tests), 7)
        raw = np.asarray([test["p_raw"] for test in tests])
        adjusted = np.asarray([test["p_BH_adjusted"] for test in tests])
        np.testing.assert_allclose(adjusted, false_discovery_control(raw, method="bh"), rtol=1e-15, atol=0)
        for test in tests:
            descriptor = test["descriptor"]
            positive = np.asarray([row[descriptor] for row in self.payload["records"] if row["label"] == 1])
            reference = np.asarray([row[descriptor] for row in self.payload["records"] if row["label"] == 0])
            reversed_test = mannwhitneyu(reference, positive, alternative="two-sided",
                                         method="asymptotic", use_continuity=True)
            with self.subTest(descriptor=descriptor):
                self.assertEqual(test["positive_n"], 842)
                self.assertEqual(test["reference_negative_n"], 831)
                self.assertEqual(reversed_test.pvalue, test["p_raw"])
                self.assertEqual(reversed_test.statistic, test["U_reference_negative"])
                self.assertEqual(test["U_positive"] + reversed_test.statistic, 842 * 831)
                reversed_effect = 2 * reversed_test.statistic / (842 * 831) - 1
                self.assertAlmostEqual(test["rank_biserial_positive_minus_reference"], -reversed_effect, places=15)
                self.assertEqual(test["significance_BH"], figure1.significance(test["p_BH_adjusted"]))


if __name__ == "__main__":
    unittest.main()
