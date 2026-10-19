"""CPU DNN ensemble with training-internal grouped epoch selection.

For every outer/inner fit, epoch selection uses a *further* training-only
grouped split. A fresh model and fresh preprocessing are then fitted on all
rows of that fit subset. Outer validation/test labels never select epochs.
The same three-seed arithmetic probability mean is used for OOF and final
prediction. Training-internal ReduceLROnPlateau retains the original scheduler
parameters. Refitting replays its selected epoch prefix without holdout labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from .modeling_classical import (DEFAULT_SEED, DEFAULT_SEEDS, PARAMETER_DESIGN,
                                group_splits, make_preprocessor, validate_fit_inputs)


DNN_TRAINING_DEFAULTS = PARAMETER_DESIGN["implementation_training_defaults"]["DNN"]


def seed_everything(seed: int, n_jobs: int = 2):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(int(n_jobs))
    torch.use_deterministic_algorithms(True)


class FeedForwardDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, batch_norm=True):
        super().__init__()
        layers = []
        previous = int(input_dim)
        for width in hidden_layers:
            layers.append(nn.Linear(previous, int(width)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(int(width)))
            layers.extend([nn.ReLU(), nn.Dropout(float(dropout))])
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(1)


@torch.no_grad()
def network_probability(model, X, batch_size=512):
    model.eval()
    tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return np.concatenate([torch.sigmoid(model(tensor[i:i + batch_size])).cpu().numpy()
                           for i in range(0, len(tensor), batch_size)]).astype(float)


def _epoch_batches(n, batch_size, generator):
    """Keep every row while avoiding singleton BatchNorm minibatches."""
    indices = torch.randperm(n, generator=generator)
    batches = list(indices.split(int(batch_size)))
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        batches.pop()
    return batches


def _train_network(X, y, params, seed, n_jobs, epochs, validation=None, patience=10,
                   min_delta=0.0001, lr_patience=4, lr_factor=0.5, learning_rates=None):
    seed_everything(seed, n_jobs)
    if len(X) < 2:
        raise ValueError("DNN BatchNorm fitting requires at least two training rows")
    model = FeedForwardDNN(X.shape[1], params["hidden_layers"], params["dropout"],
                           params.get("batch_norm", True))
    tensor_x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    tensor_y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    weight = float(np.sum(y == 0) / np.sum(y == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                                  weight_decay=float(params["weight_decay"]))
    if validation is not None and learning_rates is not None:
        raise ValueError("A replay learning-rate sequence cannot also use validation")
    if learning_rates is not None:
        learning_rates = [float(value) for value in learning_rates]
        if len(learning_rates) != int(epochs) or not all(np.isfinite(v) and v > 0 for v in learning_rates):
            raise ValueError("Learning-rate replay needs one finite positive value per epoch")
    scheduler = None
    if validation is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=float(lr_factor), patience=int(lr_patience),
            threshold=0.0001, threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-8)
    generator = torch.Generator().manual_seed(int(seed))
    history = []
    best_auc, best_epoch, stalled = -np.inf, 0, 0
    for epoch in range(1, int(epochs) + 1):
        if learning_rates is not None:
            for optimizer_group in optimizer.param_groups:
                optimizer_group["lr"] = learning_rates[epoch - 1]
        # Record the rate actually used for optimization, before a scheduler
        # update can change the rate for the following epoch.
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        total_loss = 0.0
        for batch in _epoch_batches(len(y), params["batch_size"], generator):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tensor_x[batch]), tensor_y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite DNN training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        row = {"epoch": epoch, "training_loss": total_loss / len(y),
               "learning_rate": learning_rate}
        if validation is not None:
            xv, yv = validation
            auc = float(roc_auc_score(yv, network_probability(model, xv)))
            row["validation_roc_auc"] = auc
            scheduler.step(auc)
            row["next_learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if auc > best_auc + float(min_delta):
                best_auc, best_epoch, stalled = auc, epoch, 0
            else:
                stalled += 1
        history.append(row)
        if validation is not None and stalled >= int(patience):
            break
    if validation is not None and best_epoch < 1:
        raise RuntimeError("No DNN epoch produced a finite validation AUC")
    return model, history, best_epoch if validation is not None else int(epochs), best_auc


class GroupAwareDNN(ClassifierMixin, BaseEstimator):
    def __init__(self, params, representation, seed=DEFAULT_SEED, seeds=None, n_jobs=2):
        self.params, self.representation = params, representation
        self.seed, self.seeds, self.n_jobs = seed, seeds, n_jobs

    def fit(self, X, y, groups=None, sample_ids=None, log_dir=None):
        if groups is None:
            raise ValueError("DNN fit requires structural groups")
        X, y, groups, ids = validate_fit_inputs(X, y, groups, sample_ids)
        if self.representation != "rdkit166" and not np.isfinite(X).all():
            raise ValueError("Fingerprint/count features must be finite")
        if "count" in self.representation.lower() and (X < 0).any():
            raise ValueError("Count fingerprints must be nonnegative")
        params = dict(self.params)
        max_epochs = int(params.pop("max_epochs", DNN_TRAINING_DEFAULTS["max_epochs"]))
        patience = int(params.pop("patience", DNN_TRAINING_DEFAULTS["patience"]))
        min_delta = float(params.pop("min_delta", DNN_TRAINING_DEFAULTS["min_delta"]))
        lr_patience = int(params.pop("lr_patience", DNN_TRAINING_DEFAULTS["lr_patience"]))
        lr_factor = float(params.pop("lr_factor", DNN_TRAINING_DEFAULTS["lr_factor"]))
        # Explicit alternative seeds remain supported for callers and tests.
        # Default ensemble seeds are taken directly from the shared design.
        default_seed_values = [int(self.seed) + value - DEFAULT_SEED for value in DEFAULT_SEEDS]
        seed_values = [int(v) for v in (self.seeds if self.seeds is not None else
                                      default_seed_values)]
        if not seed_values or len(seed_values) != len(set(seed_values)):
            raise ValueError("DNN seed list must be nonempty and unique")
        # Same group split for every member, separating split and init variation.
        epoch_train, epoch_validation = group_splits(y, groups, 5, int(self.seed))[0]
        selection_transform = make_preprocessor(self.representation, "dnn")
        xt = np.asarray(selection_transform.fit_transform(X[epoch_train]), dtype=np.float32)
        xv = np.asarray(selection_transform.transform(X[epoch_validation]), dtype=np.float32)
        self.preprocessing_ = make_preprocessor(self.representation, "dnn")
        full_x = np.asarray(self.preprocessing_.fit_transform(X), dtype=np.float32)
        if not np.isfinite(xt).all() or not np.isfinite(xv).all() or not np.isfinite(full_x).all():
            raise ValueError("Nonfinite DNN inputs after training-only preprocessing")
        self.models_, self.seed_fit_metadata_ = [], []
        started = time.perf_counter()
        for seed_value in seed_values:
            member_start = time.perf_counter()
            _, selection_history, best_epoch, best_auc = _train_network(
                xt, y[epoch_train], params, seed_value, self.n_jobs, max_epochs,
                validation=(xv, y[epoch_validation]), patience=patience, min_delta=min_delta,
                lr_patience=lr_patience, lr_factor=lr_factor)
            selected_learning_rates = [row["learning_rate"] for row in selection_history[:best_epoch]]
            model, refit_history, _, _ = _train_network(
                full_x, y, params, seed_value, self.n_jobs, best_epoch,
                learning_rates=selected_learning_rates)
            self.models_.append(model)
            record = {"seed": seed_value, "selected_epoch": best_epoch,
                      "best_training_internal_validation_auc": best_auc,
                      "refit_learning_rate_sequence": selected_learning_rates,
                      "epoch_selection_history": selection_history, "refit_history": refit_history,
                      "fit_seconds": time.perf_counter() - member_start}
            self.seed_fit_metadata_.append(record)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.fit_metadata_ = {
            "model": "dnn", "representation": self.representation, "params": dict(self.params),
            "seeds": seed_values, "ensemble_rule": "arithmetic mean of member positive probabilities",
            "same_ensemble_rule_for_oof_and_test": True, "n_training": len(y),
            "training_ids": ids.tolist(), "n_training_groups": len(set(groups)),
            "fit_seconds": time.perf_counter() - started, "optimizer": "AdamW",
            "effective_architecture": {"input_features": X.shape[1],
                                       "hidden_layers": params["hidden_layers"],
                                       "batch_norm": bool(params.get("batch_norm", True)),
                                       "activation": "ReLU", "dropout": float(params["dropout"]),
                                       "output": "single logit; sigmoid at prediction"},
            "loss": "BCEWithLogitsLoss; pos_weight=training negatives/training positives",
            "optimizer_parameters": {"learning_rate": float(params["learning_rate"]),
                                     "weight_decay": float(params["weight_decay"]),
                                     "betas": [0.9, 0.999], "eps": 1e-8},
            "learning_rate_schedule": "ReduceLROnPlateau on training-internal validation; refit replays selected epoch prefix",
            "scheduler_parameters": {"mode": "max", "factor": lr_factor, "patience": lr_patience,
                                     "threshold": 0.0001, "threshold_mode": "rel", "cooldown": 0,
                                     "min_lr": 0, "eps": 1e-8},
            "max_epochs": max_epochs, "patience": patience, "min_delta": min_delta,
            "early_stopping_monitor": "training-internal grouped validation ROC-AUC",
            "epoch_selection_split": {"splitter": "StratifiedGroupKFold(5), first fold",
                                      "random_state": int(self.seed),
                                      "training_ids": ids[epoch_train].tolist(),
                                      "validation_ids": ids[epoch_validation].tolist(),
                                      "group_overlap": 0},
            "preprocessing_fit_scope": "epoch selector: internal training only; refit: entire fit subset only",
            "seeds_detail": self.seed_fit_metadata_,
        }
        if log_dir is not None:
            out = Path(log_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "fit_metadata.json").write_text(json.dumps(self.fit_metadata_, indent=2, ensure_ascii=False), encoding="utf-8")
        return self

    def predict_seed_probabilities(self, X):
        transformed = np.asarray(self.preprocessing_.transform(np.asarray(X)), dtype=np.float32)
        return np.asarray([network_probability(model, transformed) for model in self.models_])

    def predict_proba(self, X):
        probability = self.predict_seed_probabilities(X).mean(axis=0)
        if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
            raise ValueError("Invalid DNN probabilities")
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
