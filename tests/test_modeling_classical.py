"""Regression checks for preprocessing isolation and grouped model subfits."""
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np

from nephrotox.modeling_classical import CANDIDATES, fit_with_groups, group_splits, make_estimator


class ModelingEstimatorTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(31)
        self.X = generator.normal(size=(60, 8))
        self.y = np.tile([0, 1], 30)
        self.groups = np.repeat([f"g{i}" for i in range(30)], 2)
        self.ids = np.array([f"S{i:03}" for i in range(60)])

    def test_original_s2a_candidates_and_code_ids(self):
        import json
        # Deliberately independent of the parameter-loader implementation and
        # of the unpublished source workbook, so a GitHub checkout can run it.
        expected = {
            "rf": [
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
                {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
                {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "max_features": "log2"},
            ],
            "svm": [{"C": 1.0, "gamma": "scale"}, {"C": 10.0, "gamma": "scale"},
                    {"C": 100.0, "gamma": "scale"}],
            "mlp": [
                {"hidden_layer_sizes": [64], "alpha": 0.0001, "learning_rate_init": 0.001},
                {"hidden_layer_sizes": [128], "alpha": 0.0001, "learning_rate_init": 0.0005},
                {"hidden_layer_sizes": [128, 64], "alpha": 0.001, "learning_rate_init": 0.001},
            ],
            "xgb": [
                {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
                {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            ],
            "dnn": [
                {"hidden_layers": [256, 128], "batch_norm": True, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 64},
                {"hidden_layers": [512, 256], "batch_norm": True, "dropout": 0.4, "learning_rate": 0.0005, "weight_decay": 0.0001, "batch_size": 64},
                {"hidden_layers": [128, 64], "batch_norm": True, "dropout": 0.25, "learning_rate": 0.0005, "weight_decay": 0.00001, "batch_size": 64},
            ],
        }
        self.assertEqual(CANDIDATES, expected)
        for model, configs in CANDIDATES.items():
            with self.subTest(model=model):
                self.assertEqual(list(range(len(configs))), [0, 1, 2])
                self.assertEqual(len({json.dumps(p, sort_keys=True) for p in configs}), 3)

    def test_default_seeds_follow_original_configuration(self):
        for name in CANDIDATES:
            with self.subTest(model=name):
                estimator = make_estimator(name, CANDIDATES[name][0], "rdkit166")
                self.assertEqual(estimator.seed, 20260810)

    def test_group_subsplits_do_not_separate_structure_groups(self):
        for a, b in group_splits(self.y, self.groups, 3, 31):
            self.assertFalse(set(self.groups[a]) & set(self.groups[b]))
            self.assertEqual(set(self.y[a]), {0, 1})
            self.assertEqual(set(self.y[b]), {0, 1})

    def test_grouped_svm_calibration_refits_training_only_preprocessing(self):
        x = self.X.copy()
        x[0, 0] = np.nan
        estimator = make_estimator("svm", {"C": 1.0, "gamma": "scale"}, "rdkit166", seed=31, n_jobs=1)
        fit_with_groups(estimator, x, self.y, self.groups, self.ids)
        calibrated = estimator.estimator_.calibrated_classifiers_[0]
        preprocessing = calibrated.estimator.named_steps["preprocessing"]
        np.testing.assert_allclose(preprocessing.named_steps["imputer"].statistics_, np.nanmedian(x, axis=0))
        before = preprocessing.named_steps["scaler"].mean_.copy()
        probability = estimator.predict_proba(np.full((4, 8), 1e9))
        np.testing.assert_array_equal(before, preprocessing.named_steps["scaler"].mean_)
        self.assertEqual(probability.shape, (4, 2))
        self.assertEqual(len(estimator.fit_metadata_["calibration_folds"]), 3)
        for split in estimator.fit_metadata_["calibration_folds"]:
            a, b = np.isin(self.ids, split["training_ids"]), np.isin(self.ids, split["validation_ids"])
            self.assertFalse(set(self.groups[a]) & set(self.groups[b]))

    def test_mlp_has_no_random_validation_or_early_stopping(self):
        params = {"hidden_layer_sizes": [4], "alpha": 0.001, "learning_rate_init": 0.001, "max_iter": 3}
        estimator = fit_with_groups(make_estimator("mlp", params, "rdkit166", seed=31, n_jobs=1),
                                    self.X, self.y, self.groups, self.ids)
        self.assertFalse(estimator.estimator_.named_steps["model"].early_stopping)
        self.assertEqual(estimator.fit_metadata_["n_iter"], 3)
        self.assertEqual(len(estimator.fit_metadata_["loss_curve"]), 3)
        self.assertTrue(estimator.fit_metadata_["convergence_warning"])

    def test_missing_groups_and_nonbinary_labels_fail(self):
        estimator = make_estimator("rf", {"n_estimators": 3}, "rdkit166", n_jobs=1)
        with self.assertRaises(ValueError):
            estimator.fit(self.X, self.y)
        labels = self.y.astype(float)
        labels[0] = 0.5
        with self.assertRaises(ValueError):
            estimator.fit(self.X, labels, self.groups)

    def test_rf_xgb_probability_and_serialization(self):
        import json
        for name, params in [("rf", {"n_estimators": 5, "max_depth": 2}),
                             ("xgb", {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1})]:
            estimator = fit_with_groups(make_estimator(name, params, "rdkit166", n_jobs=1),
                                        self.X, self.y, self.groups, self.ids)
            before = estimator.predict_proba(self.X[:5])
            # The complete runner uses strict JSON for trial/final metadata.
            json.dumps(estimator.fit_metadata_, allow_nan=False)
            if name == "xgb":
                self.assertEqual(estimator.fit_metadata_["effective_estimator_parameters"]["missing"],
                                 "NaN (XGBoost missing-value sentinel)")
                self.assertTrue(np.isnan(estimator.estimator_.named_steps["model"].missing))
                effective = estimator.fit_metadata_["effective_estimator_parameters"]
                self.assertEqual(effective["reg_lambda"], 1.0)
                self.assertEqual(effective["min_child_weight"], 1)
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / "model.joblib"
                joblib.dump(estimator, filename)
                restored = joblib.load(filename)
                np.testing.assert_allclose(before, restored.predict_proba(self.X[:5]))

    def test_unexpected_nonfinite_parameter_is_rejected_without_sanitizing(self):
        from nephrotox.modeling_classical import _documented_estimator_parameters
        class UnexpectedParameters:
            def get_params(self, deep=False):
                return {"learning_rate": float("nan")}
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _documented_estimator_parameters("xgb", UnexpectedParameters())

    def test_dnn_internal_group_epoch_selection_and_same_seed_aggregation(self):
        params = {"hidden_layers": [8, 4], "batch_norm": True, "dropout": 0.2,
                  "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
                  "max_epochs": 3, "patience": 2}
        estimator = fit_with_groups(make_estimator("dnn", params, "rdkit166", seed=31, n_jobs=1),
                                    self.X, self.y, self.groups, self.ids)
        internal = estimator.fit_metadata_["epoch_selection_split"]
        a, b = np.isin(self.ids, internal["training_ids"]), np.isin(self.ids, internal["validation_ids"])
        self.assertFalse(set(self.groups[a]) & set(self.groups[b]))
        np.testing.assert_allclose(estimator.preprocessing_.named_steps["scaler"].mean_, self.X.mean(axis=0))
        members = estimator.predict_seed_probabilities(self.X[:7])
        self.assertEqual(members.shape, (3, 7))
        np.testing.assert_allclose(estimator.predict_proba(self.X[:7])[:, 1], members.mean(axis=0))
        self.assertTrue(all(1 <= r["selected_epoch"] <= 3 for r in estimator.seed_fit_metadata_))
        self.assertTrue(all(len(r["refit_history"]) == r["selected_epoch"] for r in estimator.seed_fit_metadata_))
        for record in estimator.seed_fit_metadata_:
            selected_rates = [row["learning_rate"] for row in record["epoch_selection_history"][:record["selected_epoch"]]]
            refit_rates = [row["learning_rate"] for row in record["refit_history"]]
            self.assertEqual(selected_rates, refit_rates)

    def test_dnn_replays_actual_internal_learning_rates_during_refit(self):
        from unittest.mock import patch

        params = {"hidden_layers": [8, 4], "batch_norm": True, "dropout": 0.2,
                  "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16,
                  "max_epochs": 5, "patience": 10, "lr_patience": 0, "lr_factor": 0.5}
        # Control only the training-internal validation score to induce actual
        # optimizer LR reductions, followed by a later winning epoch.
        with patch("nephrotox.modeling_neural.roc_auc_score", side_effect=[0.5, 0.5, 0.6, 0.6, 0.7]):
            estimator = fit_with_groups(
                make_estimator("dnn", params, "rdkit166", seed=31, seeds=[31], n_jobs=1),
                self.X, self.y, self.groups, self.ids)
        record = estimator.seed_fit_metadata_[0]
        self.assertEqual(record["selected_epoch"], 5)
        expected = [0.001, 0.001, 0.0005, 0.0005, 0.00025]
        self.assertEqual(record["refit_learning_rate_sequence"], expected)
        self.assertEqual([row["learning_rate"] for row in record["epoch_selection_history"]], expected)
        self.assertEqual([row["learning_rate"] for row in record["refit_history"]], expected)
        self.assertEqual(estimator.fit_metadata_["scheduler_parameters"]["factor"], 0.5)
        self.assertEqual(estimator.fit_metadata_["scheduler_parameters"]["patience"], 0)


if __name__ == "__main__":
    unittest.main()
