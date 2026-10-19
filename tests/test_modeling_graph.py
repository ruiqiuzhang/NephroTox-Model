"""Graph semantics, isolation and checkpoint tests using synthetic molecules."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from rdkit import Chem

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from nephrotox import modeling_graph as graph


@unittest.skipIf(torch is None, "Install requirements-modeling.txt for graph tests")
class GraphSemanticsTests(unittest.TestCase):
    def test_original_reference_is_candidate_zero_without_selecting_a_winner(self):
        self.assertEqual(graph.GRAPH_SELECTION_MODE, "seeded_random_search")
        self.assertEqual(graph.GRAPH_REFERENCE_CONFIGS["D-MPNN"], {
            "aggregation": "sum", "batch_size": 32, "ffn_dropout": 0.25,
            "ffn_hidden_dim": 300, "ffn_num_layers": 3,
            "learning_rate": 0.0002875575337885439, "message_depth": 5,
            "message_dropout": 0, "message_hidden_dim": 256,
            "weight_decay": 0.00009791096446029439,
        })
        self.assertEqual(graph.GRAPH_REFERENCE_CONFIGS["GINE"], {
            "batch_size": 32, "dropout": 0.2, "hidden_dim": 256,
            "learning_rate": 0.0003, "mlp_num_layers": 2,
            "normalization": "batch_norm", "num_layers": 4,
            "pooling": "gated_attention", "train_epsilon": True,
            "weight_decay": 0.0001,
        })
        self.assertEqual(graph.GRAPH_REFERENCE_CONFIGS["AttentiveFP"], {
            "batch_size": 32, "dropout": 0.2, "hidden_channels": 256,
            "learning_rate": 0.0003, "num_layers": 3, "num_timesteps": 2,
            "weight_decay": 0.0001,
        })
        for name in graph.GRAPH_MODEL_NAMES:
            self.assertEqual(len(graph.GRAPH_CANDIDATES[name]), 20)
            estimator = graph.make_graph_estimator(name, graph.GRAPH_CANDIDATES[name][0])
            for key, value in graph.GRAPH_REFERENCE_CONFIGS[name].items():
                self.assertEqual(estimator.params[key], value)
        self.assertEqual((graph.ATOM_FDIM, graph.BOND_FDIM), (221, 33))

    def test_sampled_candidates_are_repeatable_unique_and_executable(self):
        import json
        for name in graph.GRAPH_MODEL_NAMES:
            options = graph.GRAPH_CANDIDATES[name]
            expected = [dict(graph._TRAINING_DEFAULTS, **params)
                        for params in graph.graph_candidate_configurations(name)]
            self.assertEqual(options, expected)
            self.assertEqual(len({json.dumps(params, sort_keys=True) for params in options}), 20)
            for candidate, params in enumerate(options):
                with self.subTest(architecture=name, candidate=candidate):
                    estimator = graph.make_graph_estimator(name, params, seed=20260810)
                    self.assertEqual(estimator.params, params)
                    network = graph._new_network(name, estimator.params)
                    if name == "D-MPNN":
                        self.assertEqual(network.input.out_features, params["message_hidden_dim"])
                        self.assertEqual(network.depth, params["message_depth"])
                        self.assertEqual(network.aggregation, params["aggregation"])
                        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in network.predictor),
                                         params["ffn_num_layers"] + 1)
                        self.assertEqual(network.dropout, params["message_dropout"])
                    elif name == "GINE":
                        self.assertEqual(network.input.out_features, params["hidden_dim"])
                        self.assertEqual(len(network.layers), params["num_layers"])
                        self.assertEqual(network.pooling, params["pooling"])
                        self.assertEqual(sum(isinstance(layer, nn.Linear)
                                             for layer in network.layers[0].update), params["mlp_num_layers"])
                    else:
                        self.assertEqual(network.input.out_features, params["hidden_channels"])
                        self.assertEqual(len(network.attention_layers), params["num_layers"])
                        self.assertEqual(network.readout_steps, params["num_timesteps"])
                    del network

    def test_original_network_parameters_change_the_actual_architecture(self):
        dmpnn = graph._new_network("D-MPNN", graph.GRAPH_CANDIDATES["D-MPNN"][0])
        self.assertEqual(dmpnn.input.out_features, 256)
        self.assertEqual(dmpnn.depth, 5)
        self.assertEqual(dmpnn.dropout, 0.0)
        self.assertEqual(dmpnn.aggregation, "sum")
        linear_layers = [layer for layer in dmpnn.predictor if isinstance(layer, nn.Linear)]
        self.assertEqual([(x.in_features, x.out_features) for x in linear_layers],
                         [(256, 300), (300, 300), (300, 300), (300, 1)])
        self.assertEqual([layer.p for layer in dmpnn.predictor if isinstance(layer, nn.Dropout)],
                         [0.25, 0.25, 0.25])
        gine = graph._new_network("GINE", graph.GRAPH_CANDIDATES["GINE"][0])
        self.assertEqual((len(gine.layers), gine.input.out_features, gine.pooling),
                         (4, 256, "gated_attention"))
        self.assertTrue(hasattr(gine, "pool_gate"))
        for layer in gine.layers:
            self.assertIsInstance(layer.normalization, nn.BatchNorm1d)
            self.assertIsInstance(layer.epsilon, nn.Parameter)
            self.assertEqual(sum(isinstance(x, nn.Linear) for x in layer.update), 2)
        attentive = graph._new_network("AttentiveFP", graph.GRAPH_CANDIDATES["AttentiveFP"][0])
        self.assertEqual((attentive.input.out_features, len(attentive.attention_layers),
                          attentive.readout_steps), (256, 3, 2))

    def test_unused_or_conflicting_parameters_are_rejected(self):
        for name in graph.GRAPH_MODEL_NAMES:
            with self.subTest(name=name), self.assertRaises(ValueError):
                graph.make_graph_estimator(name, {"attention_heads": 2})
        with self.assertRaises(ValueError):
            graph.make_graph_estimator("D-MPNN", {"hidden_dim": 8, "message_hidden_dim": 16})
        with self.assertRaises(ValueError):
            graph.make_graph_estimator("GINE", {"normalization": "unsupported"})
        with self.assertRaises(ValueError):
            graph.make_graph_estimator("AttentiveFP", {"learning_rate_schedule": "cosine"})

    def test_single_atom_batchnorm_and_fixed_epsilon_are_supported(self):
        norm = graph.SafeBatchNorm1d(4).train()
        value = norm(torch.ones((1, 4), requires_grad=True))
        value.sum().backward()
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(int(norm.num_batches_tracked), 0)
        layer = graph.GINELayer(4, train_epsilon=False)
        self.assertNotIsInstance(layer.epsilon, nn.Parameter)
        self.assertIn("epsilon", dict(layer.named_buffers()))

    def test_atom_and_double_bond_stereo_are_retained_without_completion(self):
        left = graph.molecular_graph("C[C@H](O)CC")
        right = graph.molecular_graph("C[C@@H](O)CC")
        unknown = graph.molecular_graph("CC(O)CC")
        np.testing.assert_array_equal(left.edge_index, right.edge_index)
        self.assertTrue(np.any(left.x != right.x))
        r_col, s_col = (graph.ATOM_FEATURE_NAMES.index(f"CIP_code={s}") for s in ("R", "S"))
        self.assertEqual(int(left.x[:, [r_col, s_col]].sum()), 1)
        self.assertEqual(int(right.x[:, [r_col, s_col]].sum()), 1)
        self.assertEqual(float(unknown.x[:, [r_col, s_col]].sum()), 0.0)
        cis, trans = graph.molecular_graph("F/C=C\\F"), graph.molecular_graph("F/C=C/F")
        self.assertTrue(np.any(cis.edge_attr != trans.edge_attr))
        self.assertEqual(cis.edge_attr.shape[1], graph.BOND_FDIM)

    def test_nonbacktracking_excludes_reverse_edge_not_other_neighbors(self):
        molecule = graph.molecular_graph("CCC")
        batch = graph.collate_graphs([molecule])
        edge_states = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
        actual = graph.nonbacktracking_messages(edge_states, batch.edge_index,
                                                 batch.reverse_edge, 3)
        # Edges are 0->1, 1->0, 1->2, 2->1. Only 2->1 may feed 1->0;
        # only 0->1 may feed 1->2. The other two sources are terminal atoms.
        np.testing.assert_array_equal(actual.numpy().ravel(), [0, 8, 1, 0])

    def test_gine_aggregates_relu_neighbor_plus_edge_and_epsilon(self):
        molecule = graph.collate_graphs([graph.molecular_graph("CCC")])
        layer = graph.GINELayer(1)
        layer.update, layer.normalization = nn.Identity(), nn.Identity()
        with torch.no_grad():
            layer.edge_projection.weight.zero_()
            layer.edge_projection.bias.zero_()
            layer.epsilon.fill_(0.5)
        output = layer(torch.tensor([[1.0], [2.0], [3.0]]), molecule)
        np.testing.assert_allclose(output.detach().numpy().ravel(), [3.5, 7.0, 6.5])

    def test_occlusion_masks_both_directions_keeps_atoms_and_original_features(self):
        original = graph.molecular_graph("c1ccccc1O")
        masked = graph.molecular_graph("c1ccccc1O", [0, 2])
        np.testing.assert_array_equal(masked.x, original.x)
        self.assertEqual(masked.edge_index.shape[1], original.edge_index.shape[1] - 4)
        self.assertFalse(set(masked.bond_index) & {0, 2})
        for i, reverse in enumerate(masked.reverse_edge):
            np.testing.assert_array_equal(masked.edge_index[:, i], masked.edge_index[::-1, reverse])
        with self.assertRaises(ValueError):
            graph.molecular_graph("CC", [1])
        with self.assertRaises(ValueError):
            graph.molecular_graph("CC", [0.5])

    def test_batch_isolation_and_achiral_atom_order_invariance(self):
        torch.set_num_threads(2)
        molecules = [graph.molecular_graph(s) for s in ("CCO", "OCC", "c1ccncc1", "[Na+]")]
        batch = graph.collate_graphs(molecules)
        np.testing.assert_array_equal(batch.batch[batch.edge_index[0]].numpy(),
                                      batch.batch[batch.edge_index[1]].numpy())
        for model_type in (graph.DMPNN, graph.GINE, graph.AttentiveFP):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(42)
                model = model_type(hidden_dim=8, depth=2, dropout=0).eval()
                with torch.inference_mode():
                    batched = model(batch).numpy()
                    separate = np.array([float(model(graph.collate_graphs([m]))) for m in molecules])
                np.testing.assert_allclose(batched, separate, atol=2e-7, rtol=1e-5)
                self.assertAlmostEqual(batched[0], batched[1], places=6)

    def test_equivalent_stereo_SMILES_traversals_give_equivalent_graph_predictions(self):
        # Reordering changes @/@@ and slash directions while preserving the
        # stereoisomer. Features must therefore be canonicalized and mapped
        # back, rather than encoding raw traversal-relative stereo flags.
        torch.set_num_threads(2)
        for smiles in ("C[C@H](O)CC", "N[C@@H](CC1=CC=CC=C1)C(=O)O",
                       "F/C=C/F", "C[C@H](O)/C=C/C", "Cl[Pt@SP1](Br)(N)O"):
            mol = Chem.MolFromSmiles(smiles)
            variants = list(dict.fromkeys(Chem.MolToRandomSmilesVect(mol, 30, randomSeed=42)))
            molecular_batch = graph.collate_graphs([graph.molecular_graph(s) for s in variants])
            for constructor in (graph.DMPNN, graph.GINE, graph.AttentiveFP):
                with self.subTest(smiles=smiles, model=constructor.__name__):
                    torch.manual_seed(42)
                    model = constructor(hidden_dim=16, depth=3, dropout=0).eval()
                    with torch.inference_mode():
                        actual = torch.sigmoid(model(molecular_batch)).numpy()
                    self.assertLessEqual(float(np.ptp(actual)), 1e-6)

    def test_occlusion_indices_still_refer_to_exact_original_SMILES_bonds(self):
        smiles = "O[C@@H](C)C/C=C/C"
        original = Chem.MolFromSmiles(smiles)
        for bond in original.GetBonds():
            masked = graph.molecular_graph(smiles, [bond.GetIdx()])
            removed_pair = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            self.assertFalse(any(set(masked.edge_index[:, j]) == removed_pair
                                 for j in range(masked.edge_index.shape[1])))
            self.assertNotIn(bond.GetIdx(), masked.bond_index)

    def test_invalid_or_mixture_input_is_rejected(self):
        for smiles in ("", "not a SMILES", "CC.O", "[*]C"):
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                graph.molecular_graph(smiles)

    def test_segment_softmax_is_per_destination_and_handles_empty_edges(self):
        values = torch.tensor([1000.0, 1001.0, -1000.0], requires_grad=True)
        membership = torch.tensor([0, 0, 2])
        probabilities = graph.segment_softmax(values, membership, 4)
        np.testing.assert_allclose(probabilities.detach().numpy(),
                                   [1 / (1 + np.e), np.e / (1 + np.e), 1], atol=1e-7)
        probabilities[0].backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(graph.segment_softmax(torch.tensor([]), torch.tensor([], dtype=torch.long), 1).numel(), 0)


@unittest.skipIf(torch is None, "Install requirements-modeling.txt for graph tests")
class GraphTrainingTests(unittest.TestCase):
    @staticmethod
    def synthetic():
        smiles = ["CCO", "CCCO", "CCCCO", "CC(C)O", "CC(=O)O", "CCC(=O)O", "CCOC", "CCOCC",
                  "CCN", "CCCN", "CCCCN", "CC(C)N", "CC(=O)N", "CCC(=O)N", "CCNC", "CCNCC"]
        return smiles, np.array([0] * 8 + [1] * 8), [f"group_{i}" for i in range(16)]

    def test_all_architectures_fit_only_given_rows_and_refit_fixed_epoch(self):
        smiles, y, groups = self.synthetic()
        ids = [f"SYN{i:03}" for i in range(len(y))]
        for name in graph.GRAPH_MODEL_NAMES:
            with self.subTest(architecture=name), tempfile.TemporaryDirectory() as folder:
                model = graph.make_graph_estimator(name, dict(hidden_dim=8, max_epochs=2,
                                                              batch_size=8, patience=1), seed=42)
                model.fit(smiles, y, groups, sample_ids=ids, log_dir=folder)
                probabilities = model.predict_proba(smiles)
                self.assertEqual(probabilities.shape, (len(y), 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                member = model.fit_metadata_["members"][0]
                train_ids = set(member["epoch_selection_training_ids"])
                validation_ids = set(member["epoch_selection_validation_ids"])
                self.assertFalse(train_ids & validation_ids)
                self.assertEqual(train_ids | validation_ids, set(ids))
                self.assertFalse(set(member["epoch_selection_training_groups"])
                                 & set(member["epoch_selection_validation_groups"]))
                self.assertEqual(len(member["refit_curve"]), member["selected_epoch"])
                self.assertEqual(set(member["refit_training_ids"]), set(ids))
                self.assertTrue((Path(folder) / "seed_42_state.pt").exists())
                self.assertTrue((Path(folder) / "fit_metadata.json").exists())
                unchanged = model.predict_with_removed_bonds(smiles[0], [])
                np.testing.assert_allclose(unchanged[0], probabilities[0], atol=2e-7)
                masked = model.predict_with_removed_bonds(smiles[:2], [[0], [1]])
                self.assertEqual(masked.shape, (2, 2))

    def test_multi_seed_checkpoint_and_joblib_roundtrip(self):
        smiles, y, groups = self.synthetic()
        model = graph.make_graph_estimator("GINE", dict(hidden_dim=8, max_epochs=2,
                                                        batch_size=8), seeds=[3, 7]).fit(smiles, y, groups)
        expected = model.predict_proba(smiles)
        seed_probabilities = model.predict_seed_probabilities(smiles)
        self.assertEqual(seed_probabilities.shape, (2, len(smiles)))
        np.testing.assert_array_equal(expected[:, 1], seed_probabilities.mean(axis=0))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = model.save_checkpoint(Path(folder) / "model.pt")
            restored = graph.load_graph_estimator(checkpoint)
            np.testing.assert_array_equal(expected, restored.predict_proba(smiles))
            file = Path(folder) / "model.joblib"
            joblib.dump(model, file)
            np.testing.assert_array_equal(expected, joblib.load(file).predict_proba(smiles))
        member_probabilities = [torch.sigmoid(torch.from_numpy(graph._predict_network(
            member, [graph.molecular_graph(s) for s in smiles], 8))).numpy()
            for member in model.models_]
        np.testing.assert_array_equal(expected[:, 1], np.mean(member_probabilities, axis=0))

    def test_repeat_fit_seed_reproduces_predictions(self):
        smiles, y, groups = self.synthetic()
        params = dict(hidden_dim=8, max_epochs=2, batch_size=8)
        first = graph.make_graph_estimator("D-MPNN", params, seed=17).fit(smiles, y, groups)
        second = graph.make_graph_estimator("D-MPNN", params, seed=17).fit(smiles, y, groups)
        np.testing.assert_array_equal(first.predict_proba(smiles), second.predict_proba(smiles))

    def test_no_ungrouped_fallback_when_grouped_holdout_is_impossible(self):
        with self.assertRaises(ValueError):
            graph.grouped_epoch_split(np.array([0, 0, 1, 1]), ["a", "a", "b", "b"], 42)


if __name__ == "__main__":
    unittest.main()
