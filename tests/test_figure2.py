"""Independent, current-DATA-only checks for the Figure 2 analysis.

The complete analysis is recomputed once in a new temporary directory. A file
audit checks that no other numerical dataset is opened, while an independently
implemented binary-set Tanimoto calculation checks every matrix element.
Synthetic fixtures check the optimisation and automatic K-selection rules.
The release analysis uses the user's explicit K=4 display choice while retaining
the independently computed automatic recommendation for transparency.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("prepare_figure2_data", ROOT / "scripts/prepare_figure2_data.py")
figure2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure2)


class Figure2MethodTests(unittest.TestCase):
    def test_kmedoids_matches_exhaustive_optimum_for_separated_toy_clusters(self):
        # Both groups have a unique central medoid; no squared-Euclidean
        # objective is substituted for the required sum of distances.
        points = np.asarray([0., 1., 2., 10., 11., 12.])
        distances = np.abs(points[:, None] - points[None, :])
        candidates = [(float(np.min(distances[:, pair], axis=1).sum()), pair)
                      for pair in itertools.combinations(range(6), 2)]
        optimum, expected_medoids = min(candidates)
        self.assertEqual(expected_medoids, (1, 4))
        for seed in range(10):
            with self.subTest(seed=seed):
                fit = figure2.alternating_kmedoids(distances, 2, seed)
                self.assertTrue(fit["converged"])
                self.assertEqual(fit["objective"], optimum)
                np.testing.assert_array_equal(fit["medoids"], expected_medoids)
                np.testing.assert_array_equal(fit["labels"], [0, 0, 0, 1, 1, 1])

    def test_collision_initialisation_is_distinct_and_assignment_keeps_medoid_ownership(self):
        # Identical bit sets do not make the source compounds the same record.
        distances = np.zeros((5, 5), dtype=np.float64)
        first = figure2.initialise_medoids(distances, 3, 42)
        second = figure2.initialise_medoids(distances, 3, 42)
        self.assertEqual(len(set(first.tolist())), 3)
        np.testing.assert_array_equal(first, second)
        labels = figure2.assign(distances, first)
        np.testing.assert_array_equal(labels[first], np.arange(3))

    def test_best_run_uses_objective_before_lexicographic_numerical_tie(self):
        fit = lambda objective, medoids: {"objective": objective, "medoids": np.asarray(medoids)}
        better = fit(1., [2, 4])
        worse = fit(1.1, [0, 1])
        numerical_tie = fit(1. + 5e-11, [1, 5])
        self.assertIs(figure2.best_run([worse, better]), better)
        self.assertIs(figure2.best_run([better, numerical_tie, worse]), numerical_tie)

    def test_k_choice_excludes_ineligible_and_prefers_smallest_within_tolerance(self):
        rows = [{"k": 2, "eligible": True, "silhouette": .201},
                {"k": 3, "eligible": True, "silhouette": .209},
                {"k": 4, "eligible": False, "silhouette": .8},
                {"k": 5, "eligible": True, "silhouette": .213}]
        self.assertEqual(figure2.select_k(rows)["k"], 3)
        self.assertEqual(figure2.select_k(list(reversed(rows)))["k"], 3)
        with self.assertRaisesRegex(ValueError, "No candidate partition"):
            figure2.select_k([{"k": 2, "eligible": False, "silhouette": .7}])

    def test_pair_summary_preserves_high_similarity_and_uses_linear_quantiles(self):
        values = np.asarray([0., .1, .2, .7, 1.])
        summary = figure2.distribution(values)
        self.assertEqual(summary["n_pairs"], 5)
        self.assertEqual(summary["q1"], .1)
        self.assertEqual(summary["median"], .2)
        self.assertEqual(summary["q3"], .7)
        self.assertAlmostEqual(summary["IQR"], .6)
        self.assertEqual(summary["mean"], .4)
        self.assertEqual(summary["max"], 1.)

    def test_frozen_protocol_preserves_chirality_and_does_not_use_tsne_to_cluster(self):
        fixed = figure2.protocol(842)
        self.assertEqual(fixed["clustering"]["candidate_k"], list(range(2, 11)))
        self.assertEqual(fixed["clustering"]["minimum_cluster_size"], 17)
        self.assertEqual(fixed["fingerprint"]["radius"], 2)
        self.assertEqual(fixed["fingerprint"]["fpSize"], 2048)
        self.assertTrue(fixed["fingerprint"]["includeChirality"])
        self.assertEqual(fixed["tsne"]["metric"], "precomputed")
        self.assertIn("do not use", fixed["tsne"]["role"])
        self.assertTrue(fixed["stability"]["validation_only_not_used_to_retune_k"])
        self.assertEqual(fixed["display_mode"], "automatic_candidate_rule")
        self.assertIsNone(fixed["user_requested_fixed_k"])
        self.assertFalse(fixed["tsne3d"]["enabled"])

    def test_fixed_four_protocol_keeps_user_choice_distinct_from_automatic_selection(self):
        fixed = figure2.protocol(842, fixed_k=4)
        self.assertEqual(fixed["user_requested_fixed_k"], 4)
        self.assertEqual(fixed["display_mode"], "user_requested_fixed_k")
        self.assertTrue(fixed["tsne3d"]["enabled"])
        self.assertEqual(fixed["tsne3d"]["n_components"], 3)
        self.assertEqual(fixed["tsne3d"]["metric"], "precomputed")
        self.assertIn("does not use either embedding", fixed["tsne3d"]["role"])
        self.assertEqual(fixed["clustering"]["candidate_k"], list(range(2, 11)))
        for invalid in (0, 1, 11):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                figure2.protocol(842, fixed_k=invalid)


class Figure2CurrentDATATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_rows, cls.info = figure2.load_data()
        cls.before = copy.deepcopy(cls.all_rows)
        cls.rows = sorted((row for row in cls.all_rows if row["label"] == 1),
                          key=lambda row: row["compound_id"])
        cls.temporary = tempfile.TemporaryDirectory(prefix="nephrotox-current-DATA-figure2-")
        cls.output = Path(cls.temporary.name).resolve()
        cls.workbook = ROOT / cls.info["source_file"]
        cls.workbook_before = hashlib.sha256(cls.workbook.read_bytes()).hexdigest()
        cls.reads = []
        active = [True]

        def audit(event, args):
            if not active[0] or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            mode = args[1]
            if isinstance(mode, str) and ("r" in mode or "+" in mode):
                cls.reads.append(Path(os.fsdecode(args[0])).resolve())

        # Passive audit only. This hook never denies an operation or changes
        # numerical behaviour, and is disabled as soon as this recompute ends.
        sys.addaudithook(audit)
        try:
            with patch.object(figure2, "load_data", return_value=(cls.all_rows, cls.info)):
                with contextlib.redirect_stdout(io.StringIO()):
                    cls.payload = figure2.prepare(cls.output, fixed_k=4)
        finally:
            active[0] = False
        cls.metadata = cls.payload["metadata"]
        cls.records = cls.payload["records"]
        with np.load(cls.output / "figure2_pairwise_matrices.npz", allow_pickle=False) as saved:
            cls.ids = saved["compound_id"].tolist()
            cls.similarity = saved["tanimoto_similarity"].copy()
            cls.distance = saved["tanimoto_distance"].copy()
            cls.saved_source_sha = str(saved["source_sha256"])
            cls.saved_protocol_sha = str(saved["analysis_protocol_sha256"])
        cls.labels = np.asarray([row["cluster"] for row in cls.records], dtype=np.int64)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_842_positive_identities_and_original_stereo_are_preserved(self):
        self.assertEqual(len(self.all_rows), 1673)
        self.assertEqual(len(self.rows), 842)
        self.assertEqual(len(self.records), 842)
        self.assertEqual(self.all_rows, self.before)
        self.assertEqual(hashlib.sha256(self.workbook.read_bytes()).hexdigest(), self.workbook_before)
        self.assertEqual(self.ids, [row["compound_id"] for row in self.rows])
        self.assertEqual(self.metadata["excluded_positive_records"], [])
        self.assertTrue(self.metadata["source_smiles_preserved_verbatim"])
        self.assertTrue(self.metadata["graph_and_stereochemistry_unchanged"])
        for original, record in zip(self.rows, self.records):
            for field in figure2.IDENTITY:
                with self.subTest(compound_id=original["compound_id"], field=field):
                    self.assertEqual(record[field], original[field])
        self.assertNotIn("NTX0788", self.ids)  # Current DATA marks Vinorelbine 0.
        self.assertTrue(np.isfinite([[row["tsne_1"], row["tsne_2"]] for row in self.records]).all())

    def test_user_requested_four_clusters_have_shared_2D_and_3D_record_assignments(self):
        self.assertEqual(self.metadata["selected_k"], 4)
        self.assertEqual(self.metadata["user_requested_fixed_k"], 4)
        self.assertEqual(self.metadata["display_mode"], "user_requested_fixed_k")
        self.assertEqual(set(self.labels.tolist()), {1, 2, 3, 4})
        self.assertEqual(len(self.payload["representatives"]), 4)
        self.assertTrue(self.metadata["tsne3d"]["enabled"])
        self.assertEqual(self.metadata["tsne3d"]["n_components"], 3)
        self.assertEqual(self.metadata["tsne3d"]["metric"], "precomputed")
        self.assertEqual(self.metadata["tsne"]["metric"], "precomputed")
        self.assertIn("does not use either embedding", self.metadata["tsne3d"]["role"])
        coordinates = np.asarray([[row[f"tsne_3d_{axis}"] for axis in (1, 2, 3)] for row in self.records])
        self.assertEqual(coordinates.shape, (842, 3))
        self.assertTrue(np.isfinite(coordinates).all())
        # Both coordinate systems live on each same identified row, with one
        # authoritative cluster column. No separate projected-space labels exist.
        for row in self.records:
            with self.subTest(compound_id=row["compound_id"]):
                self.assertEqual([key for key in row if "cluster" in key], ["cluster"])
                self.assertIn("tsne_1", row)
                self.assertIn("tsne_2", row)
                self.assertEqual(row["cluster"], self.labels[self.ids.index(row["compound_id"])])

    def test_fresh_recompute_reads_no_external_numerical_inputs_and_locks_source_hash(self):
        # The authenticated DATA rows were supplied directly by the current
        # loader above. During analysis only newly written outputs may be read.
        numerical_suffixes = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".npz", ".npy",
                              ".pkl", ".pickle", ".parquet", ".feather", ".h5", ".hdf5"}
        external_numerical = [path for path in self.reads if path.suffix.lower() in numerical_suffixes
                              and not path.is_relative_to(self.output)]
        self.assertEqual(external_numerical, [])
        self.assertEqual(self.saved_source_sha, self.info["sha256"])
        self.assertEqual(self.metadata["source"]["sha256"], self.workbook_before)
        protocol_sha = hashlib.sha256((self.output / "analysis_protocol.json").read_bytes()).hexdigest()
        self.assertEqual(protocol_sha, self.saved_protocol_sha)
        self.assertEqual(protocol_sha, self.metadata["analysis_protocol_sha256"])
        self.assertTrue(self.metadata["no_historical_numerical_input"])

    def test_every_tanimoto_value_matches_independent_bitset_arithmetic(self):
        # Integer dot products are an independent implementation of set
        # intersection/union, rather than a second call to BulkTanimotoSimilarity.
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        bits = np.zeros((842, 2048), dtype=np.int32)
        for index, row in enumerate(self.rows):
            fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(row["standardized_smiles"]))
            DataStructs.ConvertToNumpyArray(fingerprint, bits[index])
        intersection = bits @ bits.T
        cardinalities = bits.sum(axis=1)
        union = cardinalities[:, None] + cardinalities[None, :] - intersection
        self.assertTrue(np.all(union > 0))
        expected = intersection.astype(np.float64) / union
        np.testing.assert_array_equal(self.similarity, expected)
        np.testing.assert_array_equal(self.distance, 1. - expected)
        np.testing.assert_array_equal(self.similarity, self.similarity.T)
        np.testing.assert_array_equal(np.diag(self.similarity), np.ones(842))
        np.testing.assert_array_equal(np.diag(self.distance), np.zeros(842))
        self.assertTrue(np.isfinite(self.similarity).all())
        self.assertTrue(np.all((self.similarity >= 0) & (self.similarity <= 1)))
        # Morgan bit sets can collide for chemically distinct long chains.
        collisions = np.count_nonzero(np.triu(self.similarity == 1., k=1))
        self.assertEqual(collisions, self.metadata["fingerprint_collision_pairs_distinct_records"])

    def test_representatives_maximise_all_other_member_similarity_with_stable_ties(self):
        representatives = self.payload["representatives"]
        self.assertEqual(len(representatives), self.metadata["selected_k"])
        self.assertEqual([row["compound_id"] for row in representatives],
                         sorted(row["compound_id"] for row in representatives))
        for representative in representatives:
            cluster = representative["cluster"]
            members = np.flatnonzero(self.labels == cluster)
            others = [[other for other in members if other != member] for member in members]
            means = np.asarray([np.mean(self.similarity[member, other]) for member, other in zip(members, others)])
            maximum = float(np.max(means))
            best = members[np.flatnonzero(np.isclose(means, maximum, rtol=0, atol=1e-12))[0]]
            with self.subTest(cluster=cluster):
                self.assertEqual(representative["compound_id"], self.ids[best])
                self.assertTrue(representative["same_as_optimisation_medoid"])
                self.assertAlmostEqual(representative["mean_within_similarity_excluding_self"], maximum, places=14)
                self.assertEqual(representative["n_cluster"], len(members))
                for local, index in enumerate(members):
                    self.assertAlmostEqual(self.records[index]["mean_within_similarity_excluding_self"], means[local], places=14)
                    self.assertEqual(self.records[index]["is_representative"], index == best)

    def test_pair_arrays_and_summary_include_each_unordered_distinct_pair_once(self):
        n = len(self.records)
        total_within = 0
        with np.load(self.output / "figure2_intra_cluster_similarities.npz", allow_pickle=False) as saved:
            self.assertEqual(len(saved.files), self.metadata["selected_k"])
            for summary in self.payload["cluster_summaries"]:
                cluster = summary["cluster"]
                members = np.flatnonzero(self.labels == cluster)
                # Independent explicit enumeration, excluding all self pairs.
                expected = np.asarray([self.similarity[i, j] for i, j in itertools.combinations(members, 2)])
                observed = saved[f"cluster_{cluster}"]
                with self.subTest(cluster=cluster):
                    np.testing.assert_array_equal(observed, expected)
                    count = len(members) * (len(members) - 1) // 2
                    self.assertEqual(summary["n_pairs"], count)
                    self.assertEqual(summary["n_compounds"], len(members))
                    self.assertEqual(summary["fraction_of_842"], len(members) / n)
                    np.testing.assert_allclose([summary["q1"], summary["median"], summary["q3"]],
                                               np.percentile(expected, [25, 50, 75], method="linear"), rtol=0, atol=0)
                    self.assertEqual(summary["mean"], float(expected.mean()))
                    self.assertEqual(summary["min"], float(expected.min()))
                    self.assertEqual(summary["max"], float(expected.max()))
                    total_within += count
        total_between = sum(int(np.count_nonzero(self.labels == a)) * int(np.count_nonzero(self.labels == b))
                            for a, b in itertools.combinations(range(1, self.metadata["selected_k"] + 1), 2))
        self.assertEqual(total_within + total_between, n * (n - 1) // 2)
        self.assertEqual(n * (n - 1) // 2, 354061)

    def test_candidate_selection_and_selected_silhouette_use_original_distances(self):
        evaluations = self.payload["k_evaluation"]
        self.assertEqual([row["k"] for row in evaluations], list(range(2, 11)))
        eligible = []
        for evaluation in evaluations:
            sizes = [int(value) for value in evaluation["cluster_sizes"].split(";")]
            self.assertEqual(sum(sizes), 842)
            self.assertEqual(len(sizes), evaluation["k"])
            self.assertEqual(evaluation["eligible"], min(sizes) >= 17 and evaluation["all_10_runs_converged"])
            if evaluation["eligible"]:
                eligible.append(evaluation)
        maximum = max(row["silhouette"] for row in eligible)
        independently_selected = min(row["k"] for row in eligible if row["silhouette"] >= maximum - .01)
        self.assertEqual(self.metadata["auto_selected_k"], independently_selected)
        self.assertEqual(independently_selected, 7)
        automatic_rows = [row for row in evaluations if row["automatically_recommended"]]
        self.assertEqual(len(automatic_rows), 1)
        self.assertEqual(automatic_rows[0]["k"], independently_selected)
        self.assertEqual(figure2.select_k(evaluations)["k"], independently_selected)
        self.assertEqual(self.metadata["selected_k"], 4)
        selected_rows = [row for row in evaluations if row["selected"]]
        self.assertEqual(len(selected_rows), 1)
        selected = selected_rows[0]
        self.assertEqual(selected["k"], self.metadata["user_requested_fixed_k"])
        self.assertEqual(selected["k"], 4)
        self.assertEqual(self.metadata["k_selection"]["automatically_recommended"]["k"], independently_selected)
        self.assertEqual(silhouette_score(self.distance, self.labels, metric="precomputed"), selected["silhouette"])
        medoid_indices = [self.ids.index(row["compound_id"]) for row in self.payload["representatives"]]
        objective = sum(self.distance[index, medoid_indices[cluster - 1]] for index, cluster in enumerate(self.labels))
        self.assertAlmostEqual(objective, selected["objective"], places=10)

    def test_stability_has_independent_initialisations_and_without_replacement_subsamples(self):
        starts = self.payload["multistart_stability"]
        self.assertEqual(len(starts), 90)
        self.assertEqual(len({row["seed"] for row in starts}), 90)
        for k in range(2, 11):
            self.assertEqual(sorted(row["start"] for row in starts if row["k"] == k), list(range(10)))
        subsamples = self.payload["subsample_stability"]
        self.assertEqual(len(subsamples), 20)
        self.assertEqual(len({row["sampling_seed"] for row in subsamples}), 20)
        original_starts = {row["seed"] for row in starts}
        for row in subsamples:
            with self.subTest(replicate=row["replicate"]):
                self.assertEqual(row["n_sample"], 673)
                sample = np.sort(np.random.default_rng(row["sampling_seed"]).choice(842, size=673, replace=False))
                self.assertEqual(len(set(sample.tolist())), 673)
                self.assertNotIn(row["best_fit_seed"], original_starts)
                self.assertTrue(row["converged"])
                for name in ("ARI_on_shared_compounds", "ARI_full_data_reassignment"):
                    self.assertTrue(-1. <= row[name] <= 1.)
        aris = np.asarray([row["ARI_on_shared_compounds"] for row in subsamples])
        self.assertEqual(self.metadata["stability_summary"]["ARI_shared_mean"], float(aris.mean()))
        self.assertEqual(self.metadata["stability_summary"]["ARI_shared_median"], float(np.median(aris)))

    def test_heatmap_order_is_only_a_complete_within_cluster_permutation_and_manifest_hashes_match(self):
        order = self.payload["heatmap_order"]
        self.assertEqual([row["heatmap_position"] for row in order], list(range(842)))
        self.assertEqual(sorted(row["matrix_index"] for row in order), list(range(842)))
        self.assertEqual(sorted(row["compound_id"] for row in order), self.ids)
        self.assertEqual([row["cluster"] for row in order], sorted(row["cluster"] for row in order))
        for row in order:
            self.assertEqual(row["compound_id"], self.ids[row["matrix_index"]])
            self.assertEqual(row["cluster"], self.records[row["matrix_index"]]["cluster"])
        manifest = json.loads((self.output / "figure2_analysis_manifest.json").read_text(encoding="utf-8"))
        for filename, expected in manifest["files_sha256"].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((self.output / filename).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
