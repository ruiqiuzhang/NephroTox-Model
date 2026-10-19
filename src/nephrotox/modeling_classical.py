"""Finite-candidate conventional classifiers with group-isolated fitting.

Only the *fit* subset enters this module. Validation/test rows must be passed
to ``predict_proba`` afterwards. Candidate selection belongs to the nested-CV
runner. Each model compares the three prespecified configurations in the
original supplementary Table S2A, retaining candidate IDs 0, 1 and 2.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


PARAMETER_DESIGN_PATH = Path(__file__).resolve().parents[2] / "configs" / "original_parameter_design.json"
PARAMETER_DESIGN = json.loads(PARAMETER_DESIGN_PATH.read_text(encoding="utf-8"))
CANDIDATES: dict[str, list[dict[str, Any]]] = PARAMETER_DESIGN["conventional_candidates"]
DEFAULT_SEEDS = tuple(int(value) for value in PARAMETER_DESIGN["seeds"])
DEFAULT_SEED = DEFAULT_SEEDS[0]
if set(CANDIDATES) != {"rf", "svm", "mlp", "xgb", "dnn"} or any(len(values) != 3 for values in CANDIDATES.values()):
    raise ValueError("Original S2A design requires five algorithms with candidate IDs 0..2")
if len(DEFAULT_SEEDS) != 3 or len(set(DEFAULT_SEEDS)) != 3:
    raise ValueError("The neural ensemble requires exactly three distinct configured seeds")


def group_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Create explicit splits, failing on class loss or structural overlap."""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or len(np.unique(groups)) < n_splits:
        raise ValueError("Insufficient groups or mismatched labels for group CV")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for a, b in splits:
        if set(groups[a]) & set(groups[b]):
            raise ValueError("Structural group crossed an internal split")
        if set(y[a]) != {0, 1} or set(y[b]) != {0, 1}:
            raise ValueError("Internal grouped split lacks one class")
    return splits


def _is_count(representation: str) -> bool:
    return "count" in representation.lower()


def make_preprocessor(representation: str, model_name: str) -> Pipeline:
    """Return an unfitted transform; tree models do not require scaling."""
    steps = []
    if representation == "rdkit166":
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if _is_count(representation) and model_name in {"svm", "mlp", "dnn"}:
        steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))
    if model_name in {"svm", "mlp", "dnn"} and (representation == "rdkit166" or _is_count(representation)):
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps or [("identity", FunctionTransformer(validate=False))])


def validate_fit_inputs(X, y, groups, sample_ids=None):
    X, original_y, original_groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if not np.issubdtype(X.dtype, np.number) or not np.isin(original_y, [0, 1]).all():
        raise ValueError("Fit requires numeric features and exactly binary labels")
    if any(v is None or str(v).strip().lower() in {"", "nan", "none"} for v in original_groups):
        raise ValueError("Structural groups must be present for every fit row")
    y, groups = np.asarray(original_y, dtype=int), np.asarray(original_groups, dtype=str)
    ids = np.asarray(sample_ids if sample_ids is not None else [str(i) for i in range(len(y))], dtype=str)
    if X.ndim != 2 or X.shape[0] != len(y) or len(y) != len(groups) or len(ids) != len(y):
        raise ValueError("Fit feature/label/group/ID dimensions disagree")
    if set(y) != {0, 1} or np.isinf(X).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Fit requires binary classes, finite non-infinite features and unique IDs")
    return X, y, groups, ids


def _split_metadata(splits, groups, ids):
    return [{"fold": i + 1, "training_ids": ids[a].tolist(), "validation_ids": ids[b].tolist(),
             "n_training_groups": len(set(groups[a])), "n_validation_groups": len(set(groups[b])),
             "group_overlap": 0} for i, (a, b) in enumerate(splits)]


def _documented_estimator_parameters(model_name, model):
    """Document XGBoost's NaN input sentinel without emitting invalid JSON.

    This changes metadata only. The fitted estimator keeps its actual NaN
    missing-value sentinel; predictions/metrics are never sanitized here.
    """
    parameters = model.get_params(deep=False)
    for name, value in parameters.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            if model_name == "xgb" and name == "missing" and np.isnan(value):
                parameters[name] = "NaN (XGBoost missing-value sentinel)"
            else:
                raise ValueError(f"Unexpected nonfinite estimator parameter: {name}")
    return parameters


class GroupAwareClassifier(ClassifierMixin, BaseEstimator):
    """Estimator retaining group-aware probability-calibration evidence."""
    def __init__(self, model_name, params, representation, seed=DEFAULT_SEED, n_jobs=2):
        self.model_name, self.params, self.representation = model_name, params, representation
        self.seed, self.n_jobs = seed, n_jobs

    def fit(self, X, y, groups=None, sample_ids=None, log_dir=None):
        if groups is None:
            raise ValueError("Structural groups are required for every model fit")
        X, y, groups, ids = validate_fit_inputs(X, y, groups, sample_ids)
        if self.representation != "rdkit166" and not np.isfinite(X).all():
            raise ValueError("Fingerprint/count features must be finite")
        if _is_count(self.representation) and (X < 0).any():
            raise ValueError("Count fingerprints must be nonnegative")
        params = dict(self.params)
        if self.model_name == "rf":
            model = RandomForestClassifier(**params, class_weight="balanced", random_state=int(self.seed), n_jobs=int(self.n_jobs))
        elif self.model_name == "svm":
            model = SVC(**params, kernel="rbf", probability=False, class_weight="balanced", random_state=int(self.seed), cache_size=512)
        elif self.model_name == "mlp":
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
            max_iter = int(params.pop("max_iter", 200))
            model = MLPClassifier(**params, activation="relu", solver="adam", batch_size=128,
                                  max_iter=max_iter, early_stopping=False, n_iter_no_change=max_iter + 1,
                                  tol=0.0, random_state=int(self.seed))
        elif self.model_name == "xgb":
            from xgboost import XGBClassifier
            # These are fixed defaults, not additional searched parameters.
            params.setdefault("reg_lambda", 1.0)
            params.setdefault("min_child_weight", 1)
            model = XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss",
                                  scale_pos_weight=float(np.sum(y == 0) / np.sum(y == 1)),
                                  tree_method="hist", random_state=int(self.seed), n_jobs=int(self.n_jobs))
        else:
            raise ValueError(f"Unknown classical classifier: {self.model_name}")
        pipeline = Pipeline([("preprocessing", make_preprocessor(self.representation, self.model_name)), ("model", model)])
        calibration = []
        if self.model_name == "svm":
            splits = group_splits(y, groups, 3, int(self.seed))
            calibration = _split_metadata(splits, groups, ids)
            estimator = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=splits,
                                               ensemble=False, n_jobs=1)
        else:
            estimator = pipeline
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=int(self.n_jobs)):
            warnings.simplefilter("always", category=ConvergenceWarning)
            estimator.fit(X, y)
        self.estimator_ = estimator
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.fit_metadata_ = {"model": self.model_name, "representation": self.representation,
                              "params": dict(self.params), "seed": int(self.seed), "n_training": len(y),
                              "effective_estimator_parameters": _documented_estimator_parameters(self.model_name, model),
                              "training_ids": ids.tolist(), "n_training_groups": len(set(groups)),
                              "fit_seconds": time.perf_counter() - start,
                              "preprocessing_fit_scope": "each fit subset only; calibration subfits refit preprocessing",
                              "calibration": "grouped 3-fold out-of-fold sigmoid" if calibration else None,
                              "calibration_folds": calibration,
                              "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in recorded]}
        if self.model_name == "mlp":
            fitted = self.estimator_.named_steps["model"]
            self.fit_metadata_.update({"n_iter": int(fitted.n_iter_), "early_stopping": False,
                                      "loss_curve": [float(v) for v in fitted.loss_curve_],
                                      "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in recorded)})
        if log_dir is not None:
            out = Path(log_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "fit_metadata.json").write_text(json.dumps(self.fit_metadata_, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return self

    def predict_proba(self, X):
        result = np.asarray(self.estimator_.predict_proba(np.asarray(X)), dtype=float)
        if not np.isfinite(result).all() or result.shape != (len(X), 2):
            raise ValueError("Invalid classifier probabilities")
        return result

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_estimator(model_name: str, params: dict, representation: str, seed: int = DEFAULT_SEED,
                   seeds=None, n_jobs: int = 2):
    if model_name == "dnn":
        from .modeling_neural import GroupAwareDNN
        return GroupAwareDNN(params=params, representation=representation, seed=seed,
                             seeds=seeds, n_jobs=n_jobs)
    return GroupAwareClassifier(model_name=model_name, params=params, representation=representation,
                                 seed=seed, n_jobs=n_jobs)


def fit_with_groups(estimator, X, y, groups, sample_ids=None, log_dir=None):
    """Uniform fit entry point; never accepts a scoring/holdout feature array."""
    return estimator.fit(X, y, groups=groups, sample_ids=sample_ids, log_dir=log_dir)


def predict_proba(estimator, X):
    return estimator.predict_proba(X)
