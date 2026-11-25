"""Post-lock, cross-fitted explanations from current DATA and newly fitted models.

The RF comparators and all selection/perturbation rules are prespecified here.
No historical predictions, explanations, importance values or structures enter
the analysis. Explanations describe fitted-model behavior, not mechanisms.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .features import load_descriptor_names
from .modeling_common import raw_matrix, registry, training_context
from .modeling_reporting import load_locked_predictions


RF_MODELS = ("maccs166__rf", "rdkit166__rf")
GRAPH_MODELS = ("graph__DMPNN", "graph__GINE", "graph__AttentiveFP")
CASE_CATEGORIES = ("TP", "TN", "FP", "FN")
PERMUTATION_REPEATS = 10
PERMUTATION_SEED = 20261019
DISPLAY_QUANTILE = 0.95
REFERENCES = {
    "TreeSHAP_output_and_additivity": "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html",
    "MACCS_implementation": "https://www.rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html",
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path, records, fields=None):
    records = list(records)
    columns = list(fields or dict.fromkeys(key for row in records for key in row))
    if not columns:
        columns = ["status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def select_oof_cases(rows, development_index, probability, threshold=0.5):
    """Choose one extreme per MACCS-RF OOF confusion category; ID breaks ties.

    TP and FP choose largest positive probability; TN and FN choose smallest.
    Missing categories remain explicitly missing rather than being fabricated.
    No held-out observation or outcome is used for this case selection.
    """
    development_index = np.asarray(development_index, int)
    probability = np.asarray(probability, float)
    if probability.shape != (len(development_index),) or not np.isfinite(probability).all():
        raise ValueError("Case selection requires aligned finite OOF probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Case selection probabilities must be in [0,1]")
    chosen, missing = [], []
    labels = np.asarray([rows[i]["label"] for i in development_index], int)
    predicted = probability >= threshold
    mask = {"TP": (labels == 1) & predicted, "TN": (labels == 0) & ~predicted,
            "FP": (labels == 0) & predicted, "FN": (labels == 1) & ~predicted}
    for category in CASE_CATEGORIES:
        eligible = np.flatnonzero(mask[category])
        if not len(eligible):
            missing.append(category)
            continue
        local = min(eligible, key=lambda i: (
            -probability[i] if category in ("TP", "FP") else probability[i],
            rows[development_index[i]]["compound_id"]))
        chosen.append({"category": category, "row_index": int(development_index[local]),
                       "compound_id": str(rows[development_index[local]]["compound_id"]),
                       "selection_probability": float(probability[local]),
                       "selection_model": RF_MODELS[0], "role": "main",
                       "selection_population": "development OOF only"})
    return chosen, missing


def select_disagreement_cases(rows, development_index, graph_probabilities, excluded_ids=()):
    """Two additional OOF cases: widest graph probability spread per label."""
    development_index = np.asarray(development_index, int)
    probabilities = np.asarray(graph_probabilities, float)
    if probabilities.shape != (3, len(development_index)) or not np.isfinite(probabilities).all():
        raise ValueError("Graph disagreement requires three aligned finite OOF probability rows")
    spread = np.ptp(probabilities, axis=0)
    excluded_ids = set(excluded_ids)
    records = []
    for label in (1, 0):
        eligible = [i for i, g in enumerate(development_index)
                    if rows[g]["label"] == label and rows[g]["compound_id"] not in excluded_ids]
        if not eligible:
            continue
        local = min(eligible, key=lambda i: (-spread[i], rows[development_index[i]]["compound_id"]))
        index = int(development_index[local])
        records.append({"category": f"disagreement_label_{label}", "row_index": index,
                        "compound_id": str(rows[index]["compound_id"]), "role": "supplementary",
                        "selection_probability_spread": float(spread[local]),
                        "selection_model": "three graph model OOF probability range",
                        "selection_population": "development OOF only"})
    return records


def maccs_key_definitions():
    """Column zero corresponds to public RDKit key 1, not key 0."""
    special = {1: "RDKit isotope key is undefined and remains unset",
               125: "more than one aromatic ring; algorithmic key",
               166: "more than one molecular fragment; algorithmic key"}
    return [{"feature_column_zero_based": key - 1, "MACCS_key": key,
             "feature_name": f"MACCS_{key:03d}", "SMARTS": MACCSkeys.smartsPatts[key][0],
             "match_count_must_exceed": int(MACCSkeys.smartsPatts[key][1]),
             "definition_note": special.get(key, "RDKit public SMARTS/count definition")}
            for key in range(1, 167)]


def maccs_key_atoms(molecule, key):
    """Map an active key to all matching atoms, not to a unique toxic fragment."""
    if isinstance(key, bool) or int(key) != key or not 1 <= key <= 166:
        raise ValueError("MACCS key must be in 1..166")
    key = int(key)
    if key == 125:
        rings = [ring for ring in molecule.GetRingInfo().BondRings()
                 if all(molecule.GetBondWithIdx(i).GetIsAromatic() for i in ring)]
        atoms = sorted({a for ring in rings for i in ring for a in
                        (molecule.GetBondWithIdx(i).GetBeginAtomIdx(),
                         molecule.GetBondWithIdx(i).GetEndAtomIdx())})
        return atoms if len(rings) > 1 else [], len(rings)
    if key in (1, 166):
        return [], 0
    pattern, minimum = MACCSkeys.smartsPatts[key]
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"Invalid RDKit public SMARTS definition for MACCS key {key}")
    matches = molecule.GetSubstructMatches(query, uniquify=True, maxMatches=100000)
    if len(matches) <= minimum:
        return [], len(matches)
    return sorted({i for match in matches for i in match}), len(matches)


def forest_tree_shap(estimator, X):
    """Exact positive-probability TreeSHAP with an explicit additivity check.

    SHAP 0.52 sklearn RF output is (samples, features, classes). Here ``raw``
    is RF leaf probability, unlike the raw log-odds of an XGBoost classifier.
    Training-path counts provide the background without any test background.
    """
    import shap
    pipeline = getattr(estimator, "estimator_", None)
    if pipeline is None or set(pipeline.named_steps) != {"preprocessing", "model"}:
        raise ValueError("Expected the newly fitted GroupAwareClassifier RF pipeline")
    forest = pipeline.named_steps["model"]
    if not isinstance(forest, RandomForestClassifier) or list(forest.classes_) != [0, 1]:
        raise ValueError("TreeSHAP comparators must be a binary RandomForestClassifier")
    transformed = np.asarray(pipeline.named_steps["preprocessing"].transform(X))
    explainer = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent", model_output="raw")
    values = np.asarray(explainer.shap_values(transformed, check_additivity=True, approximate=False))
    if values.shape != (len(transformed), transformed.shape[1], 2):
        raise ValueError(f"Unexpected SHAP RF output dimensions: {values.shape}")
    expected = np.asarray(explainer.expected_value, float)
    if expected.shape != (2,):
        raise ValueError("Expected two RF class baselines")
    positive = values[:, :, 1]
    probability = estimator.predict_proba(X)[:, 1]
    error = np.abs(expected[1] + positive.sum(axis=1) - probability)
    if not np.isfinite(positive).all() or not np.allclose(error, 0, atol=1e-6, rtol=0):
        raise ValueError(f"Positive-class TreeSHAP additivity failed (max error {error.max()})")
    return {"values": positive, "base_value": float(expected[1]),
            "probability": probability, "transformed_X": transformed,
            "additivity_max_absolute_error": float(error.max()),
            "shap_version": str(shap.__version__),
            "output": "positive-class RF probability", "feature_perturbation": "tree_path_dependent"}


def local_bit_off(estimator, binary_row, smiles):
    """Ablate active MACCS inputs; distribute each delta equally over matched atoms.

    This perturbs the input vector. It does not remove chemical substructures,
    create a realizable counterfactual molecule or establish causality.
    """
    vector = np.asarray(binary_row)
    if vector.shape != (166,) or not np.isin(vector, [0, 1]).all():
        raise ValueError("Local MACCS perturbation requires all 166 modeled binary columns")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid local case SMILES")
    reference = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), reference)
    if not np.array_equal(reference[1:], vector):
        raise ValueError("Local MACCS input disagrees with the exact case structure")
    active = np.flatnonzero(vector)
    original = float(estimator.predict_proba(vector[None, :])[0, 1])
    altered = np.repeat(vector[None, :], len(active), axis=0)
    altered[np.arange(len(active)), active] = 0
    probabilities = estimator.predict_proba(altered)[:, 1] if len(active) else np.array([])
    atoms = np.zeros(molecule.GetNumAtoms(), dtype=float)
    records = []
    for column, probability in zip(active, probabilities):
        key = int(column + 1)
        matched_atoms, n_matches = maccs_key_atoms(molecule, key)
        delta = float(original - probability)
        if matched_atoms:
            atoms[matched_atoms] += delta / len(matched_atoms)
        records.append({"feature_column_zero_based": int(column), "MACCS_key": key,
                        "original_probability": original, "bit_off_probability": float(probability),
                        "delta_probability": delta, "matched_atom_indices": json.dumps(matched_atoms),
                        "match_count": n_matches, "mapped_atom_count": len(matched_atoms),
                        "interpretation": "input-key perturbation; not removal of a chemical fragment"})
    return {"records": records, "atom_sensitivity": atoms, "probability": original}


def permutation_importance_auc(estimator, X, y, repeats=PERMUTATION_REPEATS, seed=PERMUTATION_SEED):
    """Held-out AUC drop from a fixed final model, with feature-wise permutations."""
    X, y = np.asarray(X), np.asarray(y, int)
    if X.ndim != 2 or len(X) != len(y) or set(y.tolist()) != {0, 1} or repeats < 2:
        raise ValueError("Permutation importance needs aligned binary two-class data and >=2 repeats")
    baseline = float(roc_auc_score(y, estimator.predict_proba(X)[:, 1]))
    rng = np.random.default_rng(seed)
    values = np.empty((X.shape[1], repeats), float)
    for column in range(X.shape[1]):
        perturbed = X.copy()
        for repeat in range(repeats):
            perturbed[:, column] = X[rng.permutation(len(X)), column]
            score = float(roc_auc_score(y, estimator.predict_proba(perturbed)[:, 1]))
            values[column, repeat] = baseline - score
    return {"baseline_auc": baseline, "repeated_auc_drop": values,
            "mean": values.mean(axis=1), "sd": values.std(axis=1, ddof=1),
            "repeats": int(repeats), "seed": int(seed)}


def _verified_outer_model(model_root, model_id, fold, bundle, rows):
    folder = Path(model_root) / model_id / f"outer_{fold:02d}"
    complete_path = folder / "complete.json"
    if not complete_path.is_file():
        raise ValueError(f"Missing OOF fitted model: {model_id} fold {fold}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (complete.get("protocol_sha256") != bundle["lock"]["protocol_sha256"]
            or complete.get("model_sha256") != _digest(folder / "model.joblib")
            or complete.get("predictions_sha256") != _digest(folder / "outer_predictions.npz")):
        raise ValueError("OOF explanation model/prediction integrity failure")
    dev, folds = bundle["development_index"], bundle["folds"]
    query, training = dev[folds == fold], dev[folds != fold]
    expected_query_ids = np.array([rows[i]["compound_id"] for i in query], dtype=str)
    with np.load(folder / "outer_predictions.npz", allow_pickle=False) as values:
        if not np.array_equal(values["compound_id"], expected_query_ids):
            raise ValueError("OOF explanation prediction IDs differ from assigned validation fold")
        if not np.allclose(values["probability"], bundle["predictions"][model_id]["development"][folds == fold],
                           atol=1e-10, rtol=0):
            raise ValueError("OOF fold/global explanation probability mismatch")
    estimator = joblib.load(folder / "model.joblib")
    meta = estimator.fit_metadata_
    actual_ids = meta.get("training_ids", meta.get("fit_sample_ids"))
    if actual_ids != [rows[i]["compound_id"] for i in training]:
        raise ValueError("OOF explanation training IDs differ or include evaluation rows")
    return estimator, query, complete


def _write_shap_wide(path, ids, folds, baseline, probabilities, values, feature_names):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["compound_id", "cv_validation_fold", "base_probability", "predicted_probability", *feature_names])
        for identifier, fold, base, probability, phi in zip(ids, folds, baseline, probabilities, values):
            writer.writerow([identifier, int(fold), float(base), float(probability), *phi.tolist()])


def _display_limit(values):
    values = np.asarray(values, float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Display clipping requires nonempty finite attributions")
    actual = float(np.quantile(np.abs(values), DISPLAY_QUANTILE, method="linear"))
    return {"absolute_value_percentile": 95, "quantile_method": "linear", "quantile_value": actual,
            "symmetric_color_limit": max(actual, 1e-9), "zero_only_display_floor": 1e-9}


def _explanation_scope(model_root, scope):
    """Return one explicit lock/population; never fall back to another scope."""
    jobs = registry()
    all_ids = [job["model_id"] for job in jobs]
    if scope == "full":
        return {"scope": scope, "lock_path": model_root / "model_lock.json",
                "model_ids": all_ids, "graph_models": GRAPH_MODELS,
                "required_models": RF_MODELS + GRAPH_MODELS, "lock_kind": "global",
                "output": model_root / "explain"}
    if scope != "conventional40":
        raise ValueError("Explanation scope must be full or conventional40")
    model_ids = [job["model_id"] for job in jobs if job["representation"] != "graph"]
    if len(model_ids) != 40 or len(set(model_ids)) != 40 or set(model_ids) & set(GRAPH_MODELS):
        raise ValueError("conventional40 explanations require the exact registered 40 non-graph models")
    return {"scope": scope, "lock_path": model_root / "conventional40_lock.json",
            "model_ids": model_ids, "graph_models": (), "required_models": RF_MODELS,
            "lock_kind": "conventional40", "output": model_root / "explain_conventional40"}


def _verify_conventional_explanation_lock(lock, expected_ids):
    """Check subset authorization before any labels/probabilities are consumed."""
    all_ids = [job["model_id"] for job in registry()]
    graph_ids = [job["model_id"] for job in registry() if job["representation"] == "graph"]
    if (lock.get("schema_version") != 1 or lock.get("test_evaluation_authorized") is not True
            or lock.get("evaluation_scope") != "conventional40" or lock.get("lock_kind") != "conventional40"
            or lock.get("coverage_scope") != "conventional40" or lock.get("covered_model_count") != 40
            or lock.get("full_model_count") != len(all_ids)
            or lock.get("full_registry_completion_asserted") is not False
            or len(lock.get("expected_model_ids", [])) != 40
            or set(lock.get("expected_model_ids", [])) != set(expected_ids)
            or len(lock.get("full_expected_model_ids", [])) != len(all_ids)
            or set(lock.get("full_expected_model_ids", [])) != set(all_ids)
            or len(lock.get("excluded_model_ids", [])) != len(graph_ids)
            or set(lock.get("excluded_model_ids", [])) != set(graph_ids)):
        raise ValueError("A matching conventional40-only lock is required; graph/full completion is not asserted")


def _rf_selection_provenance(model_root, plan, lock):
    """Bind the RF primary claim to this scope's development-only report.

    Read only the selection audit and report manifest, never test predictions
    or held-out metrics. The RF analyses remain model-specific comparators;
    they are not promoted to explain a different selected primary model.
    """
    report_root = model_root / ("reports_conventional40" if plan["scope"] == "conventional40" else "reports")
    selection_path = report_root / "primary_selection.json"
    report_path = report_root / "report_manifest.json"
    if not selection_path.is_file() or not report_path.is_file():
        raise ValueError(f"{plan['scope']} explanations require the current development primary-selection report")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = [job["model_id"] for job in registry()
                if job["role"] == "main" and job["representation"] != "graph"]
    candidates = selection.get("candidate_model_ids", [])
    scores = selection.get("mean_outer_auc", {})
    lock_sha = _digest(plan["lock_path"])
    selection_sha = _digest(selection_path)
    if (report.get("status") != "complete" or report.get("evaluation_scope") != plan["scope"]
            or report.get("lock_kind") != plan["lock_kind"]
            or (report.get("scoped_lock_sha256") != lock_sha if plan["scope"] == "conventional40"
                else report.get("global_lock_sha256") != lock_sha)
            or report.get("evaluation_lock_sha256") != lock_sha
            or report.get("source_rows_fingerprint") != lock.get("source_rows_fingerprint")
            or report.get("protocol_sha256") != lock.get("protocol_sha256")
            or len(report.get("model_ids", [])) != len(plan["model_ids"])
            or set(report.get("model_ids", [])) != set(plan["model_ids"])
            or report.get("files", {}).get(selection_path.name) != selection_sha
            or selection.get("selection_population") != "development only"
            or selection.get("selection_metric") != "mean of the 10 outer-validation ROC-AUC values"
            or selection.get("tie_breaker") != "model_id ascending"
            or selection.get("test_used_for_selection") is not False
            or len(candidates) != len(expected) or set(candidates) != set(expected)
            or set(scores) != set(expected)):
        raise ValueError(f"Primary selection must match the current locked {plan['scope']} development report")
    values = np.asarray([scores[name] for name in expected], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Primary selection requires finite development fold-mean ROC-AUC values in [0,1]")
    primary = min(expected, key=lambda name: (-scores[name], name))
    if selection.get("primary_model_id") != primary or report.get("primary_model_id") != primary:
        raise ValueError("Saved primary selection disagrees with development ROC-AUC and its tie-breaker")
    descriptor_candidates = [name for name in expected if name.startswith("rdkit166__")]
    descriptor_best = min(descriptor_candidates, key=lambda name: (-scores[name], name))
    return {"primary_model_id": primary, "conventional_primary_validated": True,
            "MACCS_RF_is_selected_primary": primary == RF_MODELS[0],
            "RF_analyses_explain_selected_primary": primary in RF_MODELS,
            "primary_role": "selected primary among 25 main conventional models by development cross-validation",
            "descriptor_comparator_model_id": RF_MODELS[1],
            "descriptor_comparator_role": "descriptor RF comparator; not the selected primary",
            "descriptor_family_best_model_id": descriptor_best,
            "descriptor_RF_is_family_best": descriptor_best == RF_MODELS[1],
            "selection_population": selection["selection_population"],
            "selection_metric": selection["selection_metric"], "tie_breaker": selection["tie_breaker"],
            "test_used_for_selection": False, "candidate_model_ids": candidates,
            "mean_outer_auc": scores,
            "primary_selection_file": str(selection_path.relative_to(model_root)),
            "primary_selection_sha256": selection_sha,
            "report_manifest_file": str(report_path.relative_to(model_root)),
            "report_manifest_sha256": _digest(report_path), "evaluation_scope": plan["scope"],
            "evaluation_lock_sha256": lock_sha,
            "scoped_lock_sha256": lock_sha if plan["scope"] == "conventional40" else None}


def _graph_interpretation_selection(rows, bundle):
    """Choose the graph interpretation emphasis from ten development folds."""
    dev, folds = bundle["development_index"], bundle["folds"]
    labels = np.asarray([rows[i]["label"] for i in dev], dtype=int)
    if set(np.asarray(folds).tolist()) != set(range(1, 11)):
        raise ValueError("Graph interpretation selection requires all ten development folds")
    means, fold_auc = {}, {}
    for model_id in GRAPH_MODELS:
        probabilities = bundle["predictions"][model_id]["development"]
        values = [float(roc_auc_score(labels[folds == fold], probabilities[folds == fold]))
                  for fold in range(1, 11)]
        fold_auc[model_id] = values
        means[model_id] = float(np.mean(values))
    selected = min(GRAPH_MODELS, key=lambda name: (-means[name], name))
    return {"selected_graph_model_id": selected,
            "role": "development-selected emphasis for graph interpretation; other graph models are comparators",
            "selection_population": "development only",
            "selection_metric": "mean of the 10 outer-validation ROC-AUC values",
            "tie_breaker": "model_id ascending", "candidate_model_ids": list(GRAPH_MODELS),
            "mean_outer_auc": means, "per_fold_roc_auc": fold_auc, "test_used_for_selection": False,
            "training_or_parameters_changed": False}


def _reuse_rf_explanations(verified, model_root, output, selection):
    """Copy only the17 verified same-run RF files; never cases or graph sources."""
    from .modeling_explain_cache import RF_FILES, numerical_signature
    if verified.get("eligible_for_full_reuse") is not True:
        raise ValueError("Full RF reuse requires genuine full43 eligibility")
    eligible = verified.get("eligible_files", {})
    if set(eligible) != set(RF_FILES) or len(eligible) != 17:
        raise ValueError("Full RF reuse requires exactly the17 declared RF files")
    cache_dir = Path(model_root).resolve() / "explain_conventional40"
    output = Path(output)
    for name, item in eligible.items():
        source = Path(item["source"])
        if (Path(name).name != name or source.resolve() != cache_dir / name
                or _digest(source) != item["sha256"]):
            raise ValueError(f"RF cache source changed before copying: {name}")
    definitions = output / "maccs_key_definitions.csv"
    if definitions.is_file() and _digest(definitions) != eligible[definitions.name]["sha256"]:
        raise ValueError("Cached MACCS definitions differ from current definitions")
    for name, item in eligible.items():
        target = output / name
        temporary = target.with_name(target.name + ".rf-copy.tmp")
        shutil.copyfile(item["source"], temporary)
        if _digest(temporary) != item["sha256"]:
            temporary.unlink()
            raise ValueError(f"RF cache source changed while copying: {name}")
        temporary.replace(target)
    def records(name):
        with (output / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    aggregates = {name: records(name + ".csv") for name in (
        "shap_feature_summary", "shap_fold_importance", "shap_fold_stability",
        "permutation_importance", "local_maccs_bit_off", "local_maccs_atom_sensitivity")}
    for record in aggregates["local_maccs_atom_sensitivity"]:
        record["atom_sensitivity"] = float(record["atom_sensitivity"])
    checks = {name: dict(verified["shap_checks"][name]) for name in RF_MODELS}
    for name in RF_MODELS:
        checks[name]["role"] = (selection["primary_role"] if name == selection["primary_model_id"]
                                else "prespecified RF interpretation comparator; not the selected primary")
    provenance = {"used": True, "same_run_cache_only": True, "producer_scope": "conventional40",
        "producer_directory": "explain_conventional40", "copied_file_count": 17,
        "copied_files_sha256": {name: item["sha256"] for name, item in eligible.items()},
        "producer_manifest_sha256": verified["producer_manifest_sha256"],
        "rf_input_binding_sha256": verified["rf_input_binding_sha256"],
        "scoped_producer_lock_sha256": verified["scoped_producer_lock_sha256"],
        "current_full_evaluation_lock_sha256": verified["current_evaluation_lock_sha256"],
        "numerical_signature": numerical_signature(), "cases_reused": False,
        "case_selection_reused": False, "graph_sources_reused": False,
        "historical_results_used": False}
    return aggregates, checks, {name: dict(value) for name, value in verified["permutation"].items()}, provenance


def explain_models(rows, assignments, model_results_root, output_dir=None, *, scope="full"):
    """Generate explicitly locked explanation sources without fitting models.

    The full scope analyzes the random forests and graph models. The
    conventional40 scope uses the 40-model lock for random-forest explanations.

    A missing lock creates a pending manifest. An inconsistent lock raises
    an error.
    """
    model_root = Path(model_results_root)
    plan = _explanation_scope(model_root, scope)
    output = Path(output_dir) if output_dir else plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "explain_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("evaluation_scope", "full") != scope:
            raise ValueError("Different explanation scopes must use separate output directories")
    if scope == "conventional40" and any(output.glob("graph_*.csv")):
        raise ValueError("conventional40 explanation output cannot contain full graph-analysis artifacts")
    bound_producer = scope == "conventional40" and (output / "rf_input_binding.json").exists()
    canonical_producer = output.resolve() == model_root.resolve() / "explain_conventional40"
    if bound_producer and not canonical_producer:
        raise ValueError("A custom scoped output cannot carry a canonical same-run RF binding")
    if not plan["lock_path"].is_file():
        if bound_producer:
            raise ValueError("A bound RF producer cannot resume without its original conventional40 lock")
        manifest = {"schema_version": 1, "status": "pending",
                    "reason": "complete global model lock absent" if scope == "full" else "conventional40 lock absent",
                    "evaluation_scope": scope, "lock_kind": plan["lock_kind"],
                    "full_registry_completion_asserted": False,
                    "required_models": list(plan["required_models"]),
                    "graph_models": list(plan["graph_models"]), "historical_results_used": False}
        _json(manifest_path, manifest)
        return manifest
    if scope == "conventional40":
        scoped_lock = json.loads(plan["lock_path"].read_text(encoding="utf-8"))
        _verify_conventional_explanation_lock(scoped_lock, plan["model_ids"])
        selection_provenance = _rf_selection_provenance(model_root, plan, scoped_lock)
    else:
        selection_provenance = None
    bundle = load_locked_predictions(model_root, rows, assignments,
                                      expected_model_ids=plan["model_ids"],
                                      lock_path=plan["lock_path"] if scope == "conventional40" else None)
    for name in plan["required_models"]:
        if name not in bundle["predictions"]:
            raise ValueError(f"Prespecified explanation comparator is absent: {name}")
    if scope == "full":
        selection_provenance = _rf_selection_provenance(model_root, plan, bundle["lock"])
        graph_selection = _graph_interpretation_selection(rows, bundle)
        graph_selection_path = output / "graph_interpretation_selection.json"
        _json(graph_selection_path, graph_selection)
        selection_provenance.update({"selected_graph_model_id": graph_selection["selected_graph_model_id"],
                                     "graph_selection_file": graph_selection_path.name,
                                     "graph_selection_sha256": _digest(graph_selection_path)})
    if isinstance(assignments, dict):
        assignments = assignments["assignments"]
    context = training_context(model_root.parent)
    if context["ids"].tolist() != [r["compound_id"] for r in rows]:
        raise ValueError("Explanation DATA IDs differ from feature context")
    cache_enabled = context.get("parameter_mode", "search") == "fixed_original"
    bound_producer = bound_producer and cache_enabled
    dev, test, folds = bundle["development_index"], bundle["test_index"], bundle["folds"]
    ids = np.array([r["compound_id"] for r in rows], dtype=str)
    labels = np.array([r["label"] for r in rows], int)  # explicit evaluation lock has been verified
    definitions = maccs_key_definitions()
    main_cases, missing = select_oof_cases(rows, dev, bundle["predictions"][RF_MODELS[0]]["development"])
    supplementary = (select_disagreement_cases(
        rows, dev, [bundle["predictions"][m]["development"] for m in plan["graph_models"]],
        excluded_ids=[c["compound_id"] for c in main_cases]) if plan["graph_models"] else [])
    cases = main_cases + supplementary
    for case in cases:
        i = case["row_index"]
        case.update({"name": rows[i]["name"], "PubChem_CID": int(rows[i]["PubChem_CID"]),
                     "standardized_smiles": rows[i]["standardized_smiles"], "label": int(rows[i]["label"]),
                     "cv_validation_fold": int(assignments[i]["cv_validation_fold"])})
    matrices, feature_names = {}, {}
    for model_id in RF_MODELS:
        representation = model_id.split("__")[0]
        matrices[representation] = raw_matrix(representation, context)
        feature_names[representation] = ([d["feature_name"] for d in definitions] if representation == "maccs166"
                                         else load_descriptor_names(ROOT / "configs/rdkit_descriptors_166.txt"))
    if bound_producer:
        from .modeling_explain_cache import validate_rf_cache
        checked = validate_rf_cache(model_root, rows, bundle, matrices, feature_names, main_cases,
                                    cache_dir=output, allow_scoped_preflight=True)
        if checked.get("eligible_for_full_reuse") is not False:
            raise ValueError("Scoped RF resume cannot assert full43 reuse eligibility")
        return checked["producer_manifest"]
    _csv(output / "maccs_key_definitions.csv", definitions)
    _csv(output / "cases.csv", cases)
    _json(output / "case_selection.json", {
        "threshold": 0.5, "population": "development OOF only; no test outcome selection",
        "main_rule": "MACCS-RF: TP/FP maximum p, TN/FN minimum p, compound ID ascending on ties",
        "supplementary_rule": ("one additional case per label with maximum graph OOF probability range, excluding main cases; ID ascending on ties"
                               if plan["graph_models"] else "not performed in conventional40 RF-only scope"),
        "evaluation_scope": scope,
        "interpretation_model_selection": selection_provenance,
        "main_cases": [c["compound_id"] for c in main_cases], "missing_main_categories": missing,
        "supplementary_cases": [c["compound_id"] for c in supplementary],
    })
    summaries, fold_records, permutations, stability, bit_records, atom_records = [], [], [], [], [], []
    shap_checks, permutation_details = {}, {}
    rf_cache_reuse = {"used": False, "enabled_for_parameter_mode": cache_enabled,
                      "historical_results_used": False}
    verified_cache = None
    cache_dir = model_root / "explain_conventional40"
    if cache_enabled and scope == "full" and cache_dir.exists():
        from .modeling_explain_cache import validate_rf_cache
        # Default validation requires the real physical43 lock. A partial or
        # invalid present cache raises; it never falls back to heavy recomputation.
        verified_cache = validate_rf_cache(model_root, rows, bundle, matrices, feature_names,
                                           main_cases, cache_dir=cache_dir)
    if verified_cache is not None:
        aggregates, shap_checks, permutation_details, rf_cache_reuse = _reuse_rf_explanations(
            verified_cache, model_root, output, selection_provenance)
        summaries = aggregates["shap_feature_summary"]
        fold_records = aggregates["shap_fold_importance"]
        stability = aggregates["shap_fold_stability"]
        permutations = aggregates["permutation_importance"]
        bit_records = aggregates["local_maccs_bit_off"]
        atom_records = aggregates["local_maccs_atom_sensitivity"]
        print("RF explanations reused:17 strictly verified same-run files; current full cases retained", flush=True)
    else:
        for model_id in RF_MODELS:
            representation = model_id.split("__")[0]
            X = matrices[representation]
            names = feature_names[representation]
            phi, bases = np.full((len(dev), len(names)), np.nan), np.full(len(dev), np.nan)
            fold_importance = []
            maximum_error, shap_version = 0.0, None
            for fold in range(1, 11):
                model, query, complete = _verified_outer_model(model_root, model_id, fold, bundle, rows)
                values = forest_tree_shap(model, X[query])
                if not np.allclose(values["probability"], bundle["predictions"][model_id]["development"][folds == fold], atol=1e-8, rtol=0):
                    raise ValueError("OOF TreeSHAP fitted-model predictions differ from locked probabilities")
                phi[folds == fold], bases[folds == fold] = values["values"], values["base_value"]
                importance = np.abs(values["values"]).mean(axis=0)
                fold_importance.append(importance)
                for j, name in enumerate(names):
                    fold_records.append({"model_id": model_id, "fold": fold, "feature_column_zero_based": j,
                                         "feature_name": name, "mean_absolute_shap": float(importance[j]),
                                         "validation_n": len(query)})
                maximum_error = max(maximum_error, values["additivity_max_absolute_error"])
                shap_version = values["shap_version"]
                if model_id == RF_MODELS[0]:
                    for case in [c for c in main_cases if c["cv_validation_fold"] == fold]:
                        local = local_bit_off(model, X[case["row_index"]], case["standardized_smiles"])
                        if not np.isclose(local["probability"], case["selection_probability"], atol=1e-8, rtol=0):
                            raise ValueError("Local MACCS case was not predicted by its held-out outer model")
                        bit_records.extend({"compound_id": case["compound_id"], "fold": fold, **r} for r in local["records"])
                        atom_records.extend({"compound_id": case["compound_id"], "atom_index": a,
                                             "atom_sensitivity": float(v), "fold": fold}
                                            for a, v in enumerate(local["atom_sensitivity"]))
                del model
            if not np.isfinite(phi).all() or not np.isfinite(bases).all():
                raise ValueError("Incomplete OOF SHAP coverage")
            final_model = joblib.load(model_root / model_id / bundle["metadata"][model_id]["model_file"])
            test_values = forest_tree_shap(final_model, X[test])
            if not np.allclose(test_values["probability"], bundle["predictions"][model_id]["test"], atol=1e-8, rtol=0):
                raise ValueError("Final TreeSHAP predictions differ from locked test probabilities")
            fold_importance = np.asarray(fold_importance)
            global_importance = np.abs(phi).mean(axis=0)
            perm = permutation_importance_auc(final_model, X[test], labels[test])
            permutation_details[model_id] = {"baseline_auc": perm["baseline_auc"], "repeats": perm["repeats"],
                                             "seed": perm["seed"], "selection_population": "no model selection; post-lock held-out interpretation"}
            for j, name in enumerate(names):
                summaries.append({"model_id": model_id, "feature_column_zero_based": j, "feature_name": name,
                                  "OOF_mean_absolute_shap": float(global_importance[j]),
                                  "fold_mean_absolute_shap": float(fold_importance[:, j].mean()),
                                  "fold_sd_absolute_shap": float(fold_importance[:, j].std(ddof=1)),
                                  "test_mean_absolute_shap": float(np.abs(test_values["values"][:, j]).mean())})
                permutations.append({"model_id": model_id, "feature_column_zero_based": j, "feature_name": name,
                                      "mean_auc_drop": float(perm["mean"][j]), "sd_auc_drop": float(perm["sd"][j]),
                                      "baseline_auc": perm["baseline_auc"], "repeats": perm["repeats"], "seed": perm["seed"]})
            for a in range(10):
                for b in range(10):
                    stability.append({"model_id": model_id, "fold_a": a + 1, "fold_b": b + 1,
                                      "spearman_rho": _correlation(fold_importance[a], fold_importance[b])})
            np.savez_compressed(output / f"{model_id}_shap_oof.npz", compound_id=ids[dev], row_index=dev,
                                fold=folds, feature_names=np.array(names), X=X[dev], shap=phi,
                                base_value=bases, probability=bundle["predictions"][model_id]["development"])
            np.savez_compressed(output / f"{model_id}_shap_test.npz", compound_id=ids[test], row_index=test,
                                feature_names=np.array(names), X=X[test], shap=test_values["values"],
                                base_value=np.full(len(test), test_values["base_value"]),
                                probability=bundle["predictions"][model_id]["test"])
            np.savez_compressed(output / f"{model_id}_permutation.npz", feature_names=np.array(names),
                                repeated_auc_drop=perm["repeated_auc_drop"], seed=perm["seed"])
            _write_shap_wide(output / f"{model_id}_shap_oof.csv", ids[dev], folds, bases,
                             bundle["predictions"][model_id]["development"], phi, names)
            _write_shap_wide(output / f"{model_id}_shap_test.csv", ids[test], np.zeros(len(test), int),
                             np.full(len(test), test_values["base_value"]), test_values["probability"], test_values["values"], names)
            role = (selection_provenance["primary_role"]
                    if selection_provenance and model_id == selection_provenance["primary_model_id"]
                    else "prespecified RF interpretation comparator; not the selected primary")
            shap_checks[model_id] = {"shap_version": shap_version, "OOF_max_additivity_error": maximum_error,
                                     "test_max_additivity_error": test_values["additivity_max_absolute_error"],
                                     "SHAP_to_test_permutation_spearman": _correlation(global_importance, perm["mean"]),
                                     "output": "positive-class RF probability", "perturbation": "tree_path_dependent",
                                     "role": role}
            print(f"Explanation complete: {model_id}", flush=True)
        _csv(output / "shap_feature_summary.csv", summaries)
        _csv(output / "shap_fold_importance.csv", fold_records)
        _csv(output / "shap_fold_stability.csv", stability)
        _csv(output / "permutation_importance.csv", permutations)
        _csv(output / "local_maccs_bit_off.csv", bit_records)
        _csv(output / "local_maccs_atom_sensitivity.csv", atom_records)

    graph_records, graph_predictions, correlations = [], [], []
    for model_id in plan["graph_models"]:
        for fold in sorted({c["cv_validation_fold"] for c in cases}):
            model, query, complete = _verified_outer_model(model_root, model_id, fold, bundle, rows)
            if len(model.models_) != 3 or len(model.seeds) != 3:
                raise ValueError("OOF graph explanations require the same three formal seeds")
            for case in [c for c in cases if c["cv_validation_fold"] == fold]:
                smiles = case["standardized_smiles"]
                molecule = Chem.MolFromSmiles(smiles)
                probability = float(model.predict_proba([smiles])[0, 1])
                local = int(np.flatnonzero(dev == case["row_index"])[0])
                if not np.isclose(probability, bundle["predictions"][model_id]["development"][local], atol=1e-7, rtol=0):
                    raise ValueError("Graph explanation model does not reproduce its locked OOF probability")
                seeds = model.predict_seed_probabilities([smiles])[:, 0]
                graph_predictions.append({"model_id": model_id, "compound_id": case["compound_id"],
                    "fold": fold, "probability": probability, "label": case["label"],
                    "predicted_label": int(probability >= 0.5), "seeds": json.dumps(list(model.seeds)),
                    "member_probabilities": json.dumps(seeds.tolist()), "members": 3,
                    "population": "development OOF"})
                n_bonds = molecule.GetNumBonds()
                masked = model.predict_with_removed_bonds([smiles] * n_bonds, [[i] for i in range(n_bonds)])[:, 1] if n_bonds else []
                for bond, changed in zip(molecule.GetBonds(), masked):
                    graph_records.append({"model_id": model_id, "compound_id": case["compound_id"], "fold": fold,
                        "bond_index": bond.GetIdx(), "begin_atom_index": bond.GetBeginAtomIdx(),
                        "end_atom_index": bond.GetEndAtomIdx(), "original_probability": probability,
                        "occluded_probability": float(changed), "delta_probability": float(probability - changed),
                        "directed_edges_removed": 2, "atom_features_recomputed": False, "ensemble_members": 3,
                        "population": "development OOF"})
            del model
    for identifier in [c["compound_id"] for c in cases] + ["all_cases_pooled_bonds"]:
        subset = graph_records if identifier == "all_cases_pooled_bonds" else [r for r in graph_records if r["compound_id"] == identifier]
        lookup = {(r["model_id"], r["compound_id"], r["bond_index"]): r["delta_probability"] for r in subset}
        keys = sorted({(r["compound_id"], r["bond_index"]) for r in subset})
        for a in plan["graph_models"]:
            for b in plan["graph_models"]:
                correlations.append({"compound_id": identifier, "model_a": a, "model_b": b, "n_bonds": len(keys),
                                     "spearman_rho": _correlation([lookup[(a, *key)] for key in keys],
                                                                   [lookup[(b, *key)] for key in keys])})
    if plan["graph_models"]:
        _csv(output / "graph_bond_occlusion.csv", graph_records)
        _csv(output / "graph_case_predictions.csv", graph_predictions)
        _csv(output / "graph_occlusion_correlations.csv", correlations)
    graph_limits = _display_limit([r["delta_probability"] for r in graph_records]) if graph_records else None
    atom_limits = _display_limit([r["atom_sensitivity"] for r in atom_records]) if atom_records else None
    if verified_cache is not None and atom_limits != verified_cache.get("MACCS_atom_color_scale"):
        raise ValueError("Reused MACCS atom display limits differ from the validated producer")
    evaluation_lock_sha = _digest(plan["lock_path"])
    manifest = {"schema_version": 1, "status": "complete",
        "evaluation_scope": scope, "lock_kind": plan["lock_kind"],
        "split_mode": bundle.get("split_mode", "structure_group"),
        "structural_overlap": bundle.get("structural_overlap"),
        "evaluation_lock_sha256": evaluation_lock_sha,
        "global_lock_sha256": evaluation_lock_sha if scope == "full" else None,
        "scoped_lock_sha256": evaluation_lock_sha if scope == "conventional40" else None,
        "full_registry_completion_asserted": scope == "full",
        "covered_model_ids": plan["model_ids"], "required_models": list(plan["required_models"]),
        "graph_analysis_performed": bool(plan["graph_models"]),
        "source_rows_fingerprint": bundle["lock"]["source_rows_fingerprint"],
        "protocol_sha256": bundle["lock"]["protocol_sha256"], "analysis_source_sha256": _digest(__file__),
        "RF_comparators": list(RF_MODELS), "graph_models": list(plan["graph_models"]),
        "interpretation_model_selection": selection_provenance,
        "shap_checks": shap_checks, "permutation": permutation_details, "RF_cache_reuse": rf_cache_reuse,
        "main_cases": [c["compound_id"] for c in main_cases], "supplementary_cases": [c["compound_id"] for c in supplementary],
        "missing_main_categories": missing, "graph_bond_color_scale": graph_limits,
        "MACCS_atom_color_scale": atom_limits, "clipping_shared_across_main_and_supplement": True,
        "clipping_changes_numerical_source": False,
        "interpretation_limit": "Fitted-model feature/edge sensitivities and associations only; no causal toxic mechanism or realizable chemical counterfactual is established",
        "historical_results_used": False, "references": REFERENCES,
        "files": {p.name: _digest(p) for p in output.iterdir()
                  if p.is_file() and p.name != manifest_path.name and p.suffix in (".csv", ".json", ".npz")}}
    _json(manifest_path, manifest)
    if scope == "conventional40" and cache_enabled and canonical_producer:
        from .modeling_explain_cache import freeze_rf_inputs
        # Independent binding is recorded only after the original producer
        # manifest is final; that original manifest is never rewritten for it.
        freeze_rf_inputs(model_root, output)
    return manifest
