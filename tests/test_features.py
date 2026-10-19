"""Feature semantics and failure tests using synthetic molecules only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from nephrotox import features


CONFIG = Path(__file__).resolve().parents[1] / "configs/rdkit_descriptors_166.txt"


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"compound_id": "SYN001", "standardized_smiles": "[CH3][C@H](O)CC", "PubChem_CID": 1, "label": 0},
            {"compound_id": "SYN002", "standardized_smiles": "[CH3][C@@H](O)CC", "PubChem_CID": 2, "label": 1},
        ]

    def test_bit_order_and_leading_zeroes(self):
        expected = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
        text = features.encode_bits(expected)
        self.assertEqual(text, "00101")
        np.testing.assert_array_equal(features.decode_bits(text, 5), expected)
        for invalid, width in [(101, 3), ("001", 4), ("001x", 4)]:
            with self.assertRaises(ValueError):
                features.decode_bits(invalid, width)
        with self.assertRaises(ValueError):
            features.encode_bits(np.array([0, 0.5, 1]))

    def test_frozen_descriptor_specification(self):
        names = features.load_descriptor_names(CONFIG)
        self.assertEqual(len(names), 166)
        self.assertEqual(len(set(names)), 166)
        self.assertEqual(names[:4], ["MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex"])
        self.assertEqual(names[-1], "fr_barbitur")

    def test_chirality_and_maccs_bit_zero_removal(self):
        molecules = [Chem.MolFromSmiles(row["standardized_smiles"]) for row in self.rows]
        morgan = features.fingerprint_matrix(molecules, "morgan2048")
        self.assertTrue(np.any(morgan[0] != morgan[1]))
        maccs = features.fingerprint_matrix(molecules, "maccs166")
        np.testing.assert_array_equal(maccs[0], maccs[1])
        reference = np.zeros(167, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecules[0]), reference)
        np.testing.assert_array_equal(maccs[0], reference[1:])
        for name, width in features.FINGERPRINT_WIDTHS.items():
            matrix = features.fingerprint_matrix(molecules, name)
            self.assertEqual(matrix.shape, (2, width))
            self.assertEqual(matrix.dtype, np.uint8)

    def test_real_DATA_stereo_diagnostics_do_not_change_feature_input(self):
        # Regression fixture copied from the current DATA row NTX0004,
        # Aristophyllide B, PubChem CID 44559001. The nonlegacy stereo detector
        # can replace atom 8's cached R with S even though the graph is unchanged.
        # A diagnostic snapshot must not propagate that property to features.
        smiles = "C[C@H](CC1=CCC[C@]([C@H]1C(=C)C=O)(C)C=C)OC(=O)C2=CC3=C(C4=C5C=CC=C(C5=CC(=C24)[N+](=O)[O-])OC)OCO3"
        molecule = Chem.MolFromSmiles(smiles)
        expected = {name: features.fingerprint_matrix([Chem.MolFromSmiles(smiles)], name)
                    for name in features.FINGERPRINT_WIDTHS}
        cip_before = [(atom.GetIdx(), atom.GetProp("_CIPCode")) for atom in molecule.GetAtoms() if atom.HasProp("_CIPCode")]
        features._snapshot(molecule)
        cip_after = [(atom.GetIdx(), atom.GetProp("_CIPCode")) for atom in molecule.GetAtoms() if atom.HasProp("_CIPCode")]
        self.assertEqual(cip_before, cip_after)
        for name in features.FINGERPRINT_WIDTHS:
            np.testing.assert_array_equal(features.fingerprint_matrix([molecule], name), expected[name])

    def test_compute_real_DATA_fingerprints_equal_direct_SMILES_primitives(self):
        smiles = "C[C@H](CC1=CCC[C@]([C@H]1C(=C)C=O)(C)C=C)OC(=O)C2=CC3=C(C4=C5C=CC=C(C5=CC(=C24)[N+](=O)[O-])OC)OCO3"
        source = [{"compound_id": "DATA_REGRESSION_NTX0004", "PubChem_CID": 44559001,
                   "standardized_smiles": smiles}]
        expected = {name: features.fingerprint_matrix([Chem.MolFromSmiles(smiles)], name)
                    for name in features.FINGERPRINT_WIDTHS}
        with tempfile.TemporaryDirectory(prefix="nephrotox-DATA-regression-") as folder:
            features.compute(source, folder, CONFIG, {"kind": "current_DATA_regression_fixture"})
            for name in features.FINGERPRINT_WIDTHS:
                with np.load(Path(folder) / f"{name}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["X"], expected[name])

    def test_compute_preserves_input_and_roundtrips_every_bit(self):
        original = copy.deepcopy(self.rows)
        source_info = {"kind": "synthetic_unittest", "sha256": "synthetic"}
        with tempfile.TemporaryDirectory(prefix="nephrotox-synthetic-features-") as folder:
            out = Path(folder)
            manifest = features.compute(self.rows, out, CONFIG, source_info)
            self.assertEqual(self.rows, original)
            self.assertEqual(manifest["source"], source_info)
            self.assertTrue(manifest["molecular_graph_stereo_unchanged"])
            self.assertEqual(manifest["nonfinite_descriptor_values"], 0)
            payload = json.loads((out / "features_payload.json").read_text(encoding="utf8"))
            for name, width in features.FINGERPRINT_WIDTHS.items():
                with np.load(out / f"{name}.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["compound_id"].tolist(), [r["compound_id"] for r in original])
                    self.assertEqual(saved["standardized_smiles"].tolist(), [r["standardized_smiles"] for r in original])
                    self.assertEqual(str(saved["descriptor_file_sha256"]), manifest["descriptor_file_sha256"])
                    for i, row in enumerate(payload["fingerprints"]):
                        self.assertEqual(len(row[name]), width)
                        np.testing.assert_array_equal(features.decode_bits(row[name], width), saved["X"][i])
            with np.load(out / "rdkit166.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["X"].shape, (2, 166))
                self.assertEqual(saved["X"].dtype, np.float64)
                self.assertTrue(np.isfinite(saved["X"]).all())
                self.assertEqual(saved["descriptor_names"].tolist(), features.load_descriptor_names(CONFIG))
            self.assertFalse(list(out.glob(".features-stage-*")))

    def test_nonfinite_descriptor_aborts_matrix_publication(self):
        failure = {"compound_id": "SYN001", "descriptor": "MolWt", "reason": "test_nonfinite"}
        with tempfile.TemporaryDirectory(prefix="nephrotox-synthetic-failure-") as folder:
            out = Path(folder)
            with patch.object(features, "compute_descriptors", return_value=(np.full((2, 166), np.nan), [failure])):
                with self.assertRaisesRegex(ValueError, "Nonfinite descriptors"):
                    features.compute(self.rows, out, CONFIG)
            self.assertTrue((out / "descriptor_failures.json").exists())
            self.assertFalse(list(out.glob("*.npz")))
            self.assertFalse((out / "feature_manifest.json").exists())

    def test_synthetic_gadolinium_reports_actual_descriptor_failure(self):
        # This ion is a synthetic negative probe, never a DATA record.
        rows = [{"compound_id": "SYN_GD", "standardized_smiles": "[Gd+3]"}]
        with tempfile.TemporaryDirectory(prefix="nephrotox-synthetic-gadolinium-") as folder:
            out = Path(folder)
            with self.assertRaisesRegex(ValueError, "Nonfinite descriptors"):
                features.compute(rows, out, CONFIG)
            report = json.loads((out / "descriptor_failures.json").read_text(encoding="utf8"))
            self.assertGreater(len(report["failures"]), 0)
            self.assertTrue(all(row["compound_id"] == "SYN_GD" for row in report["failures"]))
            self.assertFalse(list(out.glob("*.npz")))
            self.assertFalse((out / "features_payload.json").exists())

    def test_invalid_and_duplicate_source_rows_rejected(self):
        for rows in [[], self.rows + [self.rows[0]], [{"compound_id": "SYNBAD", "standardized_smiles": "invalid_SMILES"}]]:
            with tempfile.TemporaryDirectory(prefix="nephrotox-synthetic-invalid-") as folder:
                with self.assertRaises(ValueError):
                    features.compute(rows, folder, CONFIG)


if __name__ == "__main__":
    unittest.main()
