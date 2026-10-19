"""Numerical and holdout-lock regression checks for current-model reporting."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nephrotox.data import ROOT, rows_fingerprint
from nephrotox.modeling_reporting import (calibration_bins, holm_adjust, load_locked_predictions,
    metric_values, paired_auc_comparison, report_predictions, stratified_bootstrap_ci, verify_model_lock,
    structure_group_bootstrap_ci, _bootstrap_distributions, _stratified_indices,
    _group_bootstrap_weights, _weighted_bootstrap_distributions)


class MetricTests(unittest.TestCase):
    def test_metrics_agree_with_independent_sklearn_including_score_ties(self):
        rng = np.random.default_rng(13)
        y = rng.integers(0, 2, 100)
        p = np.round(rng.random(100), 1)
        result, predicted = metric_values(y, p), p >= 0.5
        reference = {"ACC": metrics.accuracy_score(y, predicted), "SE": metrics.recall_score(y, predicted),
            "SP": metrics.recall_score(y, predicted, pos_label=0), "Precision": metrics.precision_score(y, predicted),
            "F1": metrics.f1_score(y, predicted), "MCC": metrics.matthews_corrcoef(y, predicted),
            "BA": metrics.balanced_accuracy_score(y, predicted), "ROC_AUC": metrics.roc_auc_score(y, p),
            "AP": metrics.average_precision_score(y, p), "Brier": metrics.brier_score_loss(y, p),
            "LogLoss": metrics.log_loss(y, p)}
        for name, value in reference.items():
            self.assertAlmostEqual(result[name], value, places=12, msg=name)
        np.testing.assert_array_equal(metrics.confusion_matrix(y, predicted).ravel(),
                                      [result[k] for k in ("TN", "FP", "FN", "TP")])

    def test_metric_validation_and_single_class(self):
        self.assertIsNone(metric_values([0, 0], [0.1, 0.2])["ROC_AUC"])
        self.assertEqual(metric_values([0, 0], [0.1, 0.2])["MCC"], 0)
        for y, p in (([0, 2], [0.1, 0.2]), ([0, 1], [0.1, float("nan")]),
                     ([0], [0.1, 0.2]), ([0, 1], [-0.1, 0.2])):
            with self.assertRaises(ValueError):
                metric_values(y, p)
        with self.assertRaises(ValueError):
            metric_values([0, 1], [0.1, 0.8], threshold=2)

    def test_bootstrap_is_reproducible_and_perfect_classifier_has_fixed_auc(self):
        y, p = [0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        first = stratified_bootstrap_ci(y, p, n_bootstrap=30, seed=19)
        second = stratified_bootstrap_ci(y, p, n_bootstrap=30, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["ROC_AUC"]["lower"], 1)
        self.assertEqual(first["ROC_AUC"]["upper"], 1)
        with self.assertRaises(ValueError):
            stratified_bootstrap_ci([0, 0], [0.1, 0.2])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
        np.testing.assert_array_equal(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            holm_adjust([-0.1])

    def test_vectorized_bootstrap_metrics_match_scalar_with_ties(self):
        y = np.tile([0, 1], 31)
        p = np.round(np.random.default_rng(41).random(len(y)), 1)
        indices = _stratified_indices(y, 43, 24)
        actual = _bootstrap_distributions(y, p, indices, 0.5)
        for i, index in enumerate(indices):
            expected = metric_values(y[index], p[index])
            for name in actual:
                self.assertAlmostEqual(actual[name][i], expected[name], places=12, msg=name)

    def test_group_bootstrap_keeps_all_members_and_matches_expanded_rows(self):
        y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        p = np.array([0.2, 0.5, 0.2, 0.6, 0.9, 0.5, 0.9, 0.1])
        groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
        weights, names, draws = _group_bootstrap_weights(groups, 35, 41)
        self.assertEqual(draws.shape, (35, 4))
        self.assertTrue(np.all(draws.sum(axis=1) == 4))
        for group in names:
            indices = np.flatnonzero(groups == group)
            np.testing.assert_array_equal(weights[:, indices[0]], weights[:, indices[1]])
        valid = (weights[:, y == 0].sum(axis=1) > 0) & (weights[:, y == 1].sum(axis=1) > 0)
        actual = _weighted_bootstrap_distributions(y, p, weights[valid])
        for i, multiplicity in enumerate(weights[valid]):
            expanded = np.repeat(np.arange(len(y)), multiplicity)
            expected = metric_values(y[expanded], p[expanded])
            for name in actual:
                self.assertAlmostEqual(actual[name][i], expected[name], places=12, msg=name)

    def test_group_bootstrap_reports_discarded_single_class_draws(self):
        y, p, groups = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["A", "A", "B", "B"]
        result = structure_group_bootstrap_ci(y, p, groups, n_bootstrap=200, seed=53)
        interval = result["ROC_AUC"]
        self.assertEqual(interval["n_bootstrap_requested"], 200)
        self.assertGreater(interval["n_bootstrap_single_class_discarded"], 0)
        self.assertEqual(interval["n_bootstrap_valid"] + interval["n_bootstrap_single_class_discarded"], 200)
        self.assertEqual(interval["lower"], 1)
        self.assertEqual(interval["upper"], 1)

    def test_paired_auc_preserves_pairing_and_null_comparison(self):
        y = np.tile([0, 1], 20)
        perfect = 0.1 + y * 0.8
        result = paired_auc_comparison(y, {"A": perfect, "B": perfect.copy(), "C": 1 - perfect},
                                       pairs=[("A", "B"), ("A", "C")], n_bootstrap=99, seed=31)
        self.assertEqual(result[0]["difference"], 0)
        self.assertEqual(result[0]["lower"], 0)
        self.assertEqual(result[0]["upper"], 0)
        self.assertEqual(result[0]["p_bootstrap"], 1)
        self.assertEqual(result[1]["difference"], 1)
        self.assertLess(result[1]["p_holm"], 0.05)

    def test_calibration_boundaries_empty_bins_and_total(self):
        bins = calibration_bins([0, 0, 1, 1], [0, 0.5, 0.9, 1], n_bins=10)
        self.assertEqual(sum(b["n"] for b in bins), 4)
        self.assertEqual(bins[-1]["n"], 2)
        self.assertIsNone(bins[1]["mean_probability"])


class LockTests(unittest.TestCase):
    def _fixture(self, folder):
        rows = [{"compound_id": f"C{i:03}", "label": i % 2, "name": f"Compound {i}", "PubChem_CID": i + 1} for i in range(24)]
        assignments = [{"compound_id": row["compound_id"], "dataset_partition": "development" if i < 20 else "test",
                        "cv_validation_fold": i // 2 + 1 if i < 20 else None, "structure_group_id": f"G{i:03}"}
                       for i, row in enumerate(rows)]
        model = folder / "descriptor_rf"
        model.mkdir()
        (folder / "protocol.json").write_text(json.dumps({"schema_version": 1, "test_fixture": True}), encoding="utf-8")
        protocol_sha = hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest()
        meta = {"algorithm": "RF", "representation": "rdkit166", "role": "main", "complete": True,
                "protocol_sha256": protocol_sha, "selection_metric": "ROC_AUC", "model_file": "model.joblib"}
        (model / "model.joblib").write_bytes(b"test fixture model artifact")
        meta["model_sha256"] = hashlib.sha256((model / "model.joblib").read_bytes()).hexdigest()
        (model / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez_compressed(model / "predictions.npz", compound_id=np.asarray([r["compound_id"] for r in rows]),
                            development_index=np.arange(20), test_index=np.arange(20, 24),
                            oof_probability=np.asarray([0.2, 0.8] * 10), test_probability=np.asarray([0.2, 0.8] * 2))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        source = json.loads((ROOT / "data/source_manifest.json").read_text(encoding="utf-8"))
        lock = {"schema_version": 1, "test_evaluation_authorized": True, "locked_at_utc": "2026-10-19T00:00:00+00:00",
                "source_rows_fingerprint": rows_fingerprint(rows), "source_excel_sha256": source["sha256"],
                "protocol_sha256": protocol_sha, "expected_model_ids": ["descriptor_rf"],
                "models": [{"model_id": "descriptor_rf", "metadata_sha256": digest(model / "metadata.json"),
                            "predictions_sha256": digest(model / "predictions.npz"), "model_sha256": meta["model_sha256"]}]}
        (folder / "model_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        return rows, assignments

    def test_missing_global_lock_rejects_before_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "forbidden before"):
                report_predictions(directory, [], [])

    def test_complete_payload_and_index_correspondence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, assignments = self._fixture(folder)
            bundle = load_locked_predictions(folder, rows, assignments, expected_model_ids=["descriptor_rf"])
            self.assertEqual(len(bundle["predictions"]["descriptor_rf"]["test"]), 4)
            with self.assertRaisesRegex(ValueError, "prespecified model set"):
                verify_model_lock(folder, rows, expected_model_ids=["other"])
            assignments[20]["structure_group_id"] = assignments[0]["structure_group_id"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_locked_predictions(folder, rows, assignments)

    def test_modified_prediction_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, assignments = self._fixture(folder)
            with (folder / "descriptor_rf/predictions.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_locked_predictions(folder, rows, assignments)

    def test_modified_trained_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, assignments = self._fixture(folder)
            (folder / "descriptor_rf/model.joblib").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_locked_predictions(folder, rows, assignments)

    def test_modified_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, assignments = self._fixture(folder)
            (folder / "protocol.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                load_locked_predictions(folder, rows, assignments)

    def test_report_selects_primary_from_development_and_keeps_fold_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, assignments = self._fixture(folder)
            result = report_predictions(folder, rows, assignments, n_bootstrap=10, seed=12,
                                        expected_model_ids=["descriptor_rf"])
            self.assertEqual(result["primary_model_id"], "descriptor_rf")
            self.assertTrue((folder / "reports/oof_predictions.csv").is_file())
            self.assertTrue((folder / "reports/test_predictions.csv").is_file())
            selection = json.loads((folder / "reports/primary_selection.json").read_text(encoding="utf-8"))
            self.assertFalse(selection["test_used_for_selection"])
            with (folder / "reports/perfold_metrics.csv").open(encoding="utf-8-sig") as stream:
                import csv
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 10)
            self.assertEqual({r["n_structure_groups"] for r in records}, {"2"})


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "Install figure requirements for layout checks")
class TrainingDiagnosticSearchTests(unittest.TestCase):
    """Graph search scores must come from all declared inner-fold predictions."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import draw_training_diagnostics as draw
        cls.draw = draw

    def test_graph_random_search_verifies_all_twenty_candidates_and_three_inner_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model_id = "graph__DMPNN"
            candidates = [{"hidden_dim": 128, "trial_fixture": i} for i in range(20)]
            selected = {"protocol_sha256": "frozen", "selected_params": candidates[0],
                        "selected_candidate": 1, "trials": []}
            search = folder / model_id / "full_development_search"
            search.mkdir(parents=True)
            path = search / "selection.json"
            rows = [{"compound_id": f"C{i}", "label": i % 2} for i in range(12)]
            ids = [r["compound_id"] for r in rows]
            bundle = {"groups": [f"G{i}" for i in range(12)], "development_index": np.arange(12),
                      "lock": {"expected_model_ids": [model_id], "protocol_sha256": "frozen"},
                      "metadata": {model_id: {"algorithm": "DMPNN", "representation": "graph", "params": candidates[0]}}}
            protocol = {"registry": [{"model_id": model_id, "estimator_key": "DMPNN"}],
                        "graph_candidate_parameters": {"DMPNN": candidates}}
            for candidate, params in enumerate(candidates, 1):
                selected["trials"].append({"candidate": candidate, "params": params,
                                           "inner_auc": [1.0] * 3, "mean_inner_auc": 1.0})
                for inner in range(1, 4):
                    inner_folder = search / f"candidate_{candidate:02d}/inner_{inner}"
                    inner_folder.mkdir(parents=True)
                    validation_ids = ids[(inner - 1) * 4:inner * 4]
                    probability = inner_folder / "validation_predictions.npz"
                    np.savez_compressed(probability, compound_id=np.asarray(validation_ids), probability=[0.2, 0.8] * 2)
                    trial = {"params": params, "protocol_sha256": "frozen", "roc_auc": 1.0,
                             "training_ids": [value for value in ids if value not in validation_ids],
                             "validation_ids": validation_ids,
                             "prediction_sha256": hashlib.sha256(probability.read_bytes()).hexdigest()}
                    (inner_folder / "trial.json").write_text(json.dumps(trial), encoding="utf-8")
            path.write_text(json.dumps(selected), encoding="utf-8")
            records, sources = self.draw.search_source(folder, bundle, rows, protocol)
            self.assertEqual(len(records), 20)
            self.assertEqual(len(sources), 121)
            self.assertEqual(sum(record["selected"] for record in records), 1)
            for field, incorrect in (("trials", []), ("selected_candidate", 2),
                                     ("selected_params", {"hidden_dim": 512})):
                changed = {**selected, field: incorrect}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.draw.search_source(folder, bundle, rows, protocol)

    def test_search_plot_separates_three_conventional_and_twenty_graph_candidates(self):
        records = [{"model_id": model, "model_label": model, "candidate": candidate,
                    "representation": "morgan2048", "mean_inner_ROC_AUC": 0.5 + candidate / 20, "selected": candidate == 3}
                   for model in ("morgan2048__rf", "maccs166__rf") for candidate in (1, 2, 3)]
        records += [{"model_id": "graph__DMPNN", "model_label": "D-MPNN", "candidate": candidate,
                     "representation": "graph", "mean_inner_ROC_AUC": 0.5 + candidate / 100, "selected": candidate == 20}
                    for candidate in range(1, 21)]
        figure = self.draw.draw_hpo(records, ["morgan2048__rf", "graph__DMPNN", "maccs166__rf"])
        try:
            panels = [ax for ax in figure.axes if ax.images]
            self.assertEqual([ax.images[0].get_array().shape for ax in panels], [(2, 3), (1, 20)])
            self.assertEqual([tick.get_text() for tick in panels[0].get_yticklabels()],
                             ["morgan2048__rf", "maccs166__rf"])
            self.assertEqual([tick.get_text() for tick in panels[1].get_yticklabels()], ["D-MPNN"])
        finally:
            self.draw.plt.close(figure)


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "Install figure requirements for layout checks")
class ModelFigureLayoutTests(unittest.TestCase):
    """Synthetic in-memory layouts, never published as scientific results."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import draw_model_figures as draw
        try:
            draw.apply_font()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        cls.draw = draw
        generator = np.random.default_rng(47)
        cls.labels = np.tile([0, 1], 50)
        metadata, predictions = {}, {}
        for representation in draw.REPRESENTATIONS:
            for algorithm in ("RF", "SVM", "MLP", "XGB", "DNN"):
                name = representation + "__" + algorithm.lower()
                metadata[name] = {"representation": representation, "algorithm": algorithm, "role": "main"}
        for algorithm in draw.GRAPH_ORDER:
            metadata["graph__" + algorithm] = {"representation": "graph", "algorithm": algorithm, "role": "main"}
        cls.summaries = {}
        for model_id in metadata:
            predictions[model_id] = {"development": generator.random(80), "test": generator.random(20)}
            y, probability = cls.labels[80:], predictions[model_id]["test"]
            record = metric_values(y, probability)
            intervals = stratified_bootstrap_ci(y, probability, n_bootstrap=25, seed=41)
            for metric, interval in intervals.items():
                record[metric + "_lower"], record[metric + "_upper"] = interval["lower"], interval["upper"]
            cls.summaries[model_id] = record
        cls.bundle = {"metadata": metadata, "predictions": predictions,
                      "development_index": np.arange(80), "test_index": np.arange(80, 100)}
        cls.lookup = draw._model_lookup(metadata)
        graphs = [cls.lookup[("graph", algorithm)] for algorithm in draw.GRAPH_ORDER]
        cls.paired = paired_auc_comparison(cls.labels[80:], {n: predictions[n]["test"] for n in graphs},
                                           pairs=[(graphs[0], graphs[1]), (graphs[0], graphs[2]), (graphs[1], graphs[2])],
                                           n_bootstrap=25, seed=41)

    def _check_in_memory(self, figure, folder, name):
        from matplotlib.text import Text
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        width, height = figure.canvas.get_width_height()
        for text in figure.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            self.assertEqual(text.get_fontfamily(), ["Arial"], msg=text.get_text())
            bounds = text.get_window_extent(renderer)
            self.assertGreaterEqual(bounds.x0, -1, msg=(name, text.get_text()))
            self.assertGreaterEqual(bounds.y0, -1, msg=(name, text.get_text()))
            self.assertLessEqual(bounds.x1, width + 1, msg=(name, text.get_text()))
            self.assertLessEqual(bounds.y1, height + 1, msg=(name, text.get_text()))
        self.draw.plt.close(figure)
        return []

    def test_main_figure_layouts_keep_all_visible_text_in_arial_and_on_canvas(self):
        # Export is replaced with an in-memory geometry check. No fixture image
        # can be mistaken for, or copied into, the final manuscript figures.
        with patch.object(self.draw, "_exports", side_effect=self._check_in_memory):
            self.draw.draw_figure3(self.bundle, self.labels, self.lookup, Path("unused"))
            self.draw.draw_figure4(self.bundle, self.summaries, self.lookup, Path("unused"))
            self.draw.draw_figure5(self.bundle, self.labels, self.summaries, self.paired, self.lookup, Path("unused"))


if __name__ == "__main__":
    unittest.main()
