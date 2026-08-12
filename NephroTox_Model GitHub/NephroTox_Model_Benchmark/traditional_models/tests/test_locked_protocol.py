from __future__ import annotations

import pandas as pd
from torch import nn

from nephrotox.config import load_config, project_path
from nephrotox.dnn import FeedForwardDNN
from nephrotox.locked_pipeline import MODELS, REPRESENTATIONS


def test_complete_five_by_five_model_matrix() -> None:
    config = load_config()
    assert len(REPRESENTATIONS) == 5
    assert len(MODELS) == 5
    assert set(config["models"]) == set(MODELS)
    assert all(len(config["models"][model]) == 3 for model in MODELS)


def test_test_labels_are_separated_from_tuning_inputs() -> None:
    config = load_config()
    development = pd.read_csv(project_path(config, config["run"]["development_file"]))
    test_features = pd.read_csv(project_path(config, config["run"]["test_features_file"]))
    sealed_labels = pd.read_csv(
        project_path(config, config["run"]["sealed_test_labels_file"])
    )
    assert "label" in development.columns
    assert "label" not in test_features.columns
    assert set(sealed_labels.columns) == {"row_index", "compound_id", "label"}
    assert len(development) == 1482
    assert len(test_features) == len(sealed_labels) == 371


def test_dnn_matches_two_hidden_layer_batchnorm_design() -> None:
    config = load_config()
    for candidate in config["models"]["dnn"]:
        assert len(candidate["hidden_layers"]) == 2
        assert candidate["batch_norm"] is True
        model = FeedForwardDNN(
            input_dim=166,
            hidden_layers=candidate["hidden_layers"],
            dropout=candidate["dropout"],
            batch_norm=candidate["batch_norm"],
        )
        assert sum(isinstance(layer, nn.BatchNorm1d) for layer in model.network) == 2
