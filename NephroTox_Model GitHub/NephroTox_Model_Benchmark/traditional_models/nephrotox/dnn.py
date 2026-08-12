from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class FeedForwardDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


@dataclass
class DNNFitResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_auc: float
    history: list[dict[str, float | int]]


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, X: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(X, dtype=torch.float32)
    probabilities: list[np.ndarray] = []
    for start in range(0, len(tensor), batch_size):
        logits = model(tensor[start : start + batch_size])
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def fit_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    params: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
) -> DNNFitResult:
    seed_everything(seed)
    model = FeedForwardDNN(
        input_dim=X_train.shape[1],
        hidden_layers=list(params["hidden_layers"]),
        dropout=float(params["dropout"]),
        batch_norm=bool(params.get("batch_norm", False)),
    )
    n_positive = int(np.sum(y_train == 1))
    n_negative = int(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_config.get("lr_factor", 0.5)),
        patience=int(training_config.get("lr_patience", 4)),
    )
    loader = _loader(
        X_train,
        y_train,
        int(params["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["patience"])
    min_delta = float(training_config["min_delta"])
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_examples += len(labels)
        validation_probability = predict_probabilities(model, X_validation)
        validation_auc = float(roc_auc_score(y_validation, validation_probability))
        scheduler.step(validation_auc)
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / total_examples,
                "validation_roc_auc": validation_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_auc > best_auc + min_delta:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("DNN training did not produce a valid checkpoint.")
    return DNNFitResult(best_state, best_epoch, best_auc, history)


def fit_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    epochs: int,
    seed: int,
) -> FeedForwardDNN:
    """Refit on development data only; no holdout labels enter this function."""
    seed_everything(seed)
    model = FeedForwardDNN(
        input_dim=X.shape[1],
        hidden_layers=list(params["hidden_layers"]),
        dropout=float(params["dropout"]),
        batch_norm=bool(params.get("batch_norm", False)),
    )
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_negative / n_positive], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loader = _loader(
        X,
        y,
        int(params["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    for _ in range(int(epochs)):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    return model
