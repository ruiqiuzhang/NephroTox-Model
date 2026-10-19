"""Explanation semantics and original-layout rendering, synthetic inputs only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from nephrotox import modeling_explain as explain
from nephrotox.modeling_classical import make_estimator

HAS_SHAP = importlib.util.find_spec("shap") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
ROOT = Path(__file__).resolve().parents[1]


class LinearBitModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        p = X.sum(axis=1) / 167.0
        return np.column_stack((1 - p, p))


class ExplanationSemanticsTests(unittest.TestCase):
    def test_case_selection_is_oof_only_and_ties_use_id(self):
        rows = [{"compound_id": identifier, "label": label} for identifier, label in
                [("B", 1), ("A", 1), ("C", 0), ("D", 0), ("E", 1), ("TEST", 1)]]
        selected, missing = explain.select_oof_cases(rows, [0, 1, 2, 3, 4], [.9, .9, .1, .8, .2])
        self.assertEqual([r["compound_id"] for r in selected], ["A", "C", "D", "E"])
        self.assertEqual(missing, [])
        self.assertTrue(all(r["compound_id"] != "TEST" for r in selected))
        selected, missing = explain.select_oof_cases(rows, [0, 2], [.9, .1])
        self.assertEqual(missing, ["FP", "FN"])

    def test_disagreement_selection_is_per_label_and_excludes_main(self):
        rows = [{"compound_id": f"S{i}", "label": i % 2} for i in range(6)]
        p = np.array([[.1, .1, .1, .2, .3, .4], [.9, .9, .5, .6, .5, .5], [.2, .3, .4, .5, .4, .5]])
        selected = explain.select_disagreement_cases(rows, np.arange(6), p, excluded_ids=["S0", "S1"])
        self.assertEqual([r["compound_id"] for r in selected], ["S3", "S2"])

    def test_maccs_column_mapping_uses_keys_1_to_166(self):
        definitions = explain.maccs_key_definitions()
        self.assertEqual([r["MACCS_key"] for r in definitions], list(range(1, 167)))
        self.assertEqual([r["feature_column_zero_based"] for r in definitions], list(range(166)))
        self.assertEqual(definitions[0]["feature_name"], "MACCS_001")
        self.assertIn("algorithmic", definitions[124]["definition_note"])

    def test_bit_off_delta_and_atom_projection_are_explicit_input_perturbations(self):
        smiles = "CCOc1ccccc1"
        molecule = Chem.MolFromSmiles(smiles)
        bits = np.zeros(167, np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), bits)
        result = explain.local_bit_off(LinearBitModel(), bits[1:], smiles)
        for record in result["records"]:
            self.assertEqual(record["MACCS_key"], record["feature_column_zero_based"] + 1)
            self.assertAlmostEqual(record["delta_probability"], 1 / 167)
            self.assertIn("input-key", record["interpretation"])
        mapped_total = sum(r["delta_probability"] for r in result["records"] if r["mapped_atom_count"])
        self.assertAlmostEqual(float(result["atom_sensitivity"].sum()), mapped_total)
        changed = bits[1:].copy(); changed[0] = 1
        with self.assertRaises(ValueError):
            explain.local_bit_off(LinearBitModel(), changed, smiles)

    def test_algorithmic_aromatic_ring_key_maps_actual_ring_atoms(self):
        molecule = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
        atoms, matches = explain.maccs_key_atoms(molecule, 125)
        self.assertEqual(matches, 2)
        self.assertEqual(len(atoms), 12)
        self.assertEqual(explain.maccs_key_atoms(Chem.MolFromSmiles("c1ccccc1"), 125)[0], [])

    def test_abs_95_clipping_is_display_only_and_handles_zero_maps(self):
        values = np.array([-2., 0., .1, .3, .5])
        before = values.copy()
        limits = explain._display_limit(values)
        self.assertAlmostEqual(limits["quantile_value"], float(np.quantile(abs(values), .95)))
        np.testing.assert_array_equal(values, before)
        self.assertEqual(explain._display_limit([0, 0])["symmetric_color_limit"], 1e-9)

    def test_missing_lock_is_pending_without_predictions_or_test_access(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(explain, "load_locked_predictions", side_effect=AssertionError("must not access")):
                result = explain.explain_models([], [], Path(folder) / "models")
            self.assertEqual(result["status"], "pending")
            files = list((Path(folder) / "models/explain").iterdir())
            self.assertEqual([f.name for f in files], ["explain_manifest.json"])

    @unittest.skipUnless(HAS_SHAP, "Install requirements-modeling.txt for SHAP tests")
    def test_actual_tree_shap_positive_output_reconstructs_probability(self):
        rng = np.random.default_rng(13)
        X = rng.integers(0, 2, size=(80, 166), dtype=np.uint8)
        y = X[:, 3]
        model = make_estimator("rf", {"n_estimators": 12, "max_depth": 4}, "maccs166")
        model.fit(X, y, np.array([f"g{i}" for i in range(len(y))]))
        result = explain.forest_tree_shap(model, X[:11])
        self.assertEqual(result["values"].shape, (11, 166))
        self.assertLess(result["additivity_max_absolute_error"], 1e-6)
        np.testing.assert_allclose(result["base_value"] + result["values"].sum(axis=1),
                                   model.predict_proba(X[:11])[:, 1], atol=1e-6)

    def test_permutation_auc_drop_preserves_input_and_finds_predictive_feature(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(100, 3)); y = (X[:, 0] > 0).astype(int)
        model = make_estimator("rf", {"n_estimators": 16, "max_depth": 4}, "rdkit166")
        model.fit(X[:70], y[:70], np.array([f"g{i}" for i in range(70)]))
        original = X[70:].copy()
        result = explain.permutation_importance_auc(model, X[70:], y[70:], repeats=4)
        np.testing.assert_array_equal(original, X[70:])
        self.assertEqual(result["repeated_auc_drop"].shape, (3, 4))
        self.assertGreater(result["mean"][0], .2)


@unittest.skipUnless(HAS_MATPLOTLIB, "Install requirements-figures.txt for rendering tests")
class ExplanationRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import draw_explain_figures as draw
        try:
            draw.apply_font()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        cls.draw = draw

    @staticmethod
    def synthetic_source():
        rng = np.random.default_rng(19)
        smiles = ["CCOc1ccccc1", "CC(O)CC", "CCN(CC)CC", "CC(=O)Oc1ccccc1C(=O)O", "CCCO", "CCCN"]
        cases = []
        for i, structure in enumerate(smiles):
            cases.append({"compound_id": f"SYN{i:03}", "name": "Synthetic layout validation molecule",
                "PubChem_CID": i + 1, "standardized_smiles": structure, "label": int(i in (0, 3, 4)),
                "row_index": i, "cv_validation_fold": i + 1, "selection_probability": [.95, .05, .9, .1, .6, .3][i],
                "category": explain.CASE_CATEGORIES[i] if i < 4 else f"disagreement_label_{i - 4}",
                "role": "main" if i < 4 else "supplementary"})
        source = {"cases": cases, "graph_bond_occlusion": [], "graph_case_predictions": [],
                  "graph_occlusion_correlations": [], "local_maccs_atom_sensitivity": [],
                  "shap_feature_summary": [], "shap_fold_importance": [], "permutation_importance": [],
                  "shap": {}, "manifest": {"graph_bond_color_scale": {"symmetric_color_limit": .15},
                      "MACCS_atom_color_scale": {"symmetric_color_limit": .1}, "shap_checks": {}}}
        for case in cases:
            molecule = Chem.MolFromSmiles(case["standardized_smiles"])
            source["local_maccs_atom_sensitivity"].extend({"compound_id": case["compound_id"], "atom_index": a,
                "atom_sensitivity": float(rng.normal(0, .05))} for a in range(molecule.GetNumAtoms()))
            for model in explain.GRAPH_MODELS:
                source["graph_case_predictions"].append({"model_id": model, "compound_id": case["compound_id"],
                                                          "probability": float(rng.uniform(.05, .95))})
                source["graph_bond_occlusion"].extend({"model_id": model, "compound_id": case["compound_id"],
                    "bond_index": b, "delta_probability": float(rng.normal(0, .1))} for b in range(molecule.GetNumBonds()))
        source["graph_occlusion_correlations"] = [{"compound_id": "all_cases_pooled_bonds", "model_a": a,
            "model_b": b, "spearman_rho": 1. if a == b else .5} for a in explain.GRAPH_MODELS for b in explain.GRAPH_MODELS]
        for model in explain.RF_MODELS:
            names = [f"MACCS_{i+1:03}" if model == explain.RF_MODELS[0] else f"Descriptor_{i}" for i in range(166)]
            source["shap"][model] = {"X": rng.normal(size=(20, 166)), "shap": rng.normal(0, .01, size=(20, 166))}
            source["manifest"]["shap_checks"][model] = {"SHAP_to_test_permutation_spearman": .42}
            for i, name in enumerate(names):
                source["shap_feature_summary"].append({"model_id": model, "feature_name": name,
                    "feature_column_zero_based": i, "OOF_mean_absolute_shap": float(.03 / (i + 1)),
                    "fold_sd_absolute_shap": .001})
                source["shap_fold_importance"].extend({"model_id": model, "feature_name": name, "fold": f,
                    "mean_absolute_shap": float(.03 / (i + 1) * rng.uniform(.8, 1.2))} for f in range(1, 11))
                source["permutation_importance"].append({"model_id": model, "feature_name": name,
                    "mean_auc_drop": float(.02 / (i + 1)), "sd_auc_drop": .001})
        return source

    def test_all_four_original_layouts_render_inside_Arial_canvas(self):
        source = self.synthetic_source()
        with tempfile.TemporaryDirectory() as folder:
            for method in (self.draw.draw_figure6, self.draw.draw_figure7, self.draw.draw_s6):
                with self.subTest(layout=method.__name__):
                    fig = method(source, Path(folder))
                    self.draw.validate_figure(fig)
                    self.draw.plt.close(fig)
            fig = self.draw.draw_s7(source)
            self.draw.validate_figure(fig)
            self.draw.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
