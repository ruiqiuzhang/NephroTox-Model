"""Strict same-run RF explanation reuse; never load estimators or graph outputs.

The binding is independent evidence, including when it is recorded after the RF
producer has completed. It never modifies that producer's original manifest.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .input_provenance import source_file_fields

SOURCE_ROOT = Path(__file__).resolve().parents[2]
RF_MODELS = ("maccs166__rf", "rdkit166__rf")
FUNCTIONS = ("forest_tree_shap", "local_bit_off", "maccs_key_atoms", "maccs_key_definitions",
             "permutation_importance_auc", "_correlation", "_display_limit", "select_oof_cases")
CONSTANTS = ("RF_MODELS", "CASE_CATEGORIES", "PERMUTATION_REPEATS", "PERMUTATION_SEED", "DISPLAY_QUANTILE")
FITTING_FILES = ("modeling_common.py", "modeling_pipeline.py", "modeling_classical.py", "modeling_neural.py",
                 "modeling_graph.py", "search_space.py", "fixed_parameters.py")
RF_FILES = tuple([f"{model}_{suffix}" for model in RF_MODELS for suffix in
                  ("shap_oof.npz", "shap_test.npz", "permutation.npz", "shap_oof.csv", "shap_test.csv")]
                 + ["maccs_key_definitions.csv", "shap_feature_summary.csv", "shap_fold_importance.csv",
                    "shap_fold_stability.csv", "permutation_importance.csv", "local_maccs_bit_off.csv",
                    "local_maccs_atom_sensitivity.csv"])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def numerical_signature(source=None):
    """Canonical ASTs ignore locations/comments and documentation-only strings."""
    source = Path(source) if source else SOURCE_ROOT / "src/nephrotox/modeling_explain.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions, constants = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            functions[node.name] = hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANTS:
                    constants[target.id] = ast.literal_eval(node.value)
    _require(set(functions) == set(FUNCTIONS) and set(constants) == set(CONSTANTS), "Missing RF numerical functions/constants")
    return {"algorithm_ast_sha256": functions, "constants": json.loads(json.dumps(constants)),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "rdkit", "shap", "joblib")}}


def _inputs(model_root, cache_dir):
    root, cache = Path(model_root).resolve(), Path(cache_dir).resolve()
    _require(cache == root / "explain_conventional40", "RF cache directory must be the current scoped producer output")
    protocol_path, lock_path = root / "protocol.json", root / "conventional40_lock.json"
    protocol, lock = _read(protocol_path), _read(lock_path)
    _require(protocol.get("split_mode") in {"legacy_random", "structure_group"}
             and protocol.get("parameter_mode") == "fixed_original", "Unsupported RF cache protocol mode: fixed_original is required")
    ids = [job["model_id"] for job in protocol["registry"] if job["representation"] != "graph"]
    _require(len(ids) == 40 and len(set(ids)) == 40 and set(RF_MODELS) <= set(ids), "Expected exact conventional40 registry")
    _require(lock.get("schema_version") == 1 and lock.get("locked_at_utc") and lock.get("test_evaluation_authorized") is True and lock.get("evaluation_scope") == lock.get("lock_kind") == "conventional40"
             and set(lock.get("expected_model_ids", [])) == set(ids) and len(lock["expected_model_ids"]) == 40,
             "Incomplete conventional40 RF cache lock")
    _require(len(lock.get("models", [])) == 40 and {record["model_id"] for record in lock["models"]} == set(ids), "RF scoped lock model entries are incomplete")
    _require(lock.get("protocol_sha256") == _sha(protocol_path), "RF cache lock/protocol mismatch")
    _require(all(_sha(SOURCE_ROOT / "src/nephrotox" / name) == protocol["fitting_source_sha256"][name] for name in FITTING_FILES), "Current fitting source differs from frozen RF protocol")
    _require(protocol.get("fixed_parameter_file") == "configs/original_fixed_parameters.json"
             and _sha(SOURCE_ROOT / protocol["fixed_parameter_file"]) == protocol.get("fixed_parameter_sha256"), "Frozen RF parameter source changed")
    _require(source_file_fields(lock) == source_file_fields(protocol)
             and lock["source_rows_fingerprint"] == protocol["source_rows_fingerprint"], "RF cache DATA identity mismatch")
    split = root.parent / "split_payload.json"
    _require(_sha(split) == protocol["split_payload_sha256"], "RF cache split changed")
    entries = {record["model_id"]: record for record in lock["models"]}
    files = {}
    for model in RF_MODELS:
        folder = root / model
        metadata_path = folder / "metadata.json"
        metadata = _read(metadata_path)
        _require(metadata.get("complete") is True and metadata.get("protocol_sha256") == lock["protocol_sha256"]
                 and _sha(metadata_path) == entries[model]["metadata_sha256"], "RF final metadata differs from scoped lock")
        final = (metadata_path, folder / metadata["model_file"], folder / "predictions.npz")
        _require(final[1].parent == folder and _sha(final[1]) == metadata["model_sha256"] == entries[model]["model_sha256"]
                 and _sha(final[2]) == metadata["predictions_sha256"] == entries[model]["predictions_sha256"], "RF final model/prediction input changed")
        for path in final:
            files[path.relative_to(root).as_posix()] = _sha(path)
        for fold in range(1, 11):
            outer = folder / f"outer_{fold:02d}"
            complete_path = outer / "complete.json"
            complete = _read(complete_path)
            model_path, prediction_path = outer / "model.joblib", outer / "outer_predictions.npz"
            _require(complete.get("protocol_sha256") == lock["protocol_sha256"]
                     and complete.get("model_sha256") == _sha(model_path)
                     and complete.get("predictions_sha256") == _sha(prediction_path), "RF outer input changed")
            for path in (complete_path, model_path, prediction_path):
                files[path.relative_to(root).as_posix()] = _sha(path)
    _require(len(files) == 66, "Expected66 RF final/ten-fold input files")
    return {"model_root": str(root), "cache_directory": str(cache), "source_root": str(SOURCE_ROOT.resolve()),
            "protocol_sha256": _sha(protocol_path), "split_payload_sha256": _sha(split),
            "source_snapshot_sha256": _sha(root.parent / "source_snapshot.json"),
            **source_file_fields(lock), "source_rows_fingerprint": lock["source_rows_fingerprint"],
            "scoped_lock_sha256": _sha(lock_path), "scope_model_ids": ids, "RF_models": list(RF_MODELS), "model_inputs_sha256": files}


def freeze_rf_inputs(model_root, cache_dir=None):
    """Record independent same-run provenance; leave all producer evidence intact."""
    root = Path(model_root).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else root / "explain_conventional40"
    binding = _inputs(root, cache)
    manifest_path = cache / "explain_manifest.json"
    value = {"schema_version": 1, "status": "input_binding_recorded", "same_run_cache_only": True,
             "historical_results_used": False, "binding": binding, "numerical_signature": numerical_signature(),
             "recorded_after_producer_complete": manifest_path.is_file(),
             "producer_manifest_sha256": _sha(manifest_path) if manifest_path.is_file() else None,
             "original_producer_manifest_modified": False}
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        _require(manifest.get("status") == "complete" and manifest.get("evaluation_scope") == "conventional40", "Existing RF producer manifest is incomplete")
        producer_source = SOURCE_ROOT / "src/nephrotox/modeling_explain.py"
        snapshot = cache / "rf_producer_source.py"
        if snapshot.exists():
            _require(_sha(snapshot) == manifest["analysis_source_sha256"], "RF producer source snapshot changed")
            _require(numerical_signature(snapshot) == value["numerical_signature"], "RF producer/current numerical signatures changed")
        else:
            _require(manifest.get("analysis_source_sha256") == _sha(producer_source), "Producer source changed before independent input binding")
            snapshot.write_bytes(producer_source.read_bytes())
    path = cache / "rf_input_binding.json"
    cache.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(_read(path) == value, "Existing RF input binding changed; never silently replace it")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    return {"path": str(path), "sha256": _sha(path), "producer_manifest_sha256": value["producer_manifest_sha256"],
            "model_input_files": len(binding["model_inputs_sha256"]), "recorded_after_producer_complete": value["recorded_after_producer_complete"]}


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _row_fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _main_case_identity(rows, bundle):
    development = np.asarray(bundle["development_index"], int)
    probabilities = np.asarray(bundle["predictions"][RF_MODELS[0]]["development"])
    _require(probabilities.shape == (len(development),), "Wrong current OOF probability shape")
    chosen, missing = [], []
    for category in ("TP", "TN", "FP", "FN"):
        label = 1 if category in ("TP", "FN") else 0
        predicted = category in ("TP", "FP")
        eligible = [local for local, index in enumerate(development)
                    if rows[index]["label"] == label and bool(probabilities[local] >= .5) == predicted]
        if not eligible:
            missing.append(category)
            continue
        local = min(eligible, key=lambda i: (-probabilities[i] if predicted else probabilities[i], rows[development[i]]["compound_id"]))
        index, row = int(development[local]), rows[int(development[local])]
        chosen.append({"category": category, "row_index": index, "compound_id": row["compound_id"],
                       "name": row["name"], "PubChem_CID": int(row["PubChem_CID"]),
                       "standardized_smiles": row["standardized_smiles"], "label": int(row["label"]),
                       "cv_validation_fold": int(bundle["folds"][local]), "selection_probability": float(probabilities[local])})
    return chosen, missing


def _check_cases(current, expected):
    _require(len(current) == len(expected), "RF cached/current main case count differs")
    expected_map = {row["compound_id"]: row for row in expected}
    _require(len(expected_map) == len(expected) and {row["compound_id"] for row in current} == set(expected_map), "RF main case IDs differ")
    integer = {"row_index", "PubChem_CID", "label", "cv_validation_fold"}
    for row in current:
        target = expected_map[row["compound_id"]]
        for key, value in target.items():
            actual = int(row[key]) if key in integer else float(row[key]) if key == "selection_probability" else row[key]
            _require(actual == value, f"RF main case identity changed: {key}")


def validate_rf_cache(model_root, rows, bundle, matrices, feature_names, main_cases, *,
                      cache_dir=None, allow_scoped_preflight=False):
    """Return17 eligible sources only after complete provenance and array checks.

Default use requires the genuine full43 physical lock/current bundle. An explicit
scoped preflight can verify the same RF arrays before graph training finishes,
but returns ``eligible_for_full_reuse=False`` and never asserts full completion.
No file is copied, no estimator is deserialized, and cases.csv is never returned.
"""
    root = Path(model_root).resolve()
    cache = Path(cache_dir).resolve() if cache_dir else root / "explain_conventional40"
    _require(cache == root / "explain_conventional40", "RF cache directory must be the current scoped producer output")
    sidecar_path = cache / "rf_input_binding.json"
    sidecar = _read(sidecar_path)
    _require(sidecar.get("binding", {}).get("model_root") == str(root)
             and sidecar["binding"].get("cache_directory") == str(cache), "RF cached/current run roots differ")
    current = _inputs(root, cache)
    protocol = _read(root / "protocol.json")
    _require(sidecar.get("status") == "input_binding_recorded" and sidecar.get("same_run_cache_only") is True
             and sidecar.get("historical_results_used") is False and sidecar.get("original_producer_manifest_modified") is False,
             "Invalid RF cache input-binding policy")
    _require(sidecar.get("binding") == current, "RF cache run/model/data/split inputs changed")
    signature = numerical_signature()
    _require(sidecar.get("numerical_signature") == signature, "RF numerical algorithm signature/constants/packages changed")
    manifest_path = cache / "explain_manifest.json"
    manifest = _read(manifest_path)
    if sidecar.get("recorded_after_producer_complete"):
        _require(sidecar.get("producer_manifest_sha256") == _sha(manifest_path), "Original RF producer manifest changed")
        snapshot = cache / "rf_producer_source.py"
        _require(_sha(snapshot) == manifest.get("analysis_source_sha256"), "RF producer source snapshot does not match original manifest")
        _require(numerical_signature(snapshot) == signature, "RF producer/current numerical AST signatures differ")
    else:
        _require(manifest.get("files", {}).get(sidecar_path.name) == _sha(sidecar_path), "Producer did not bind the pre-completion RF sidecar")
    _require(manifest.get("status") == "complete" and manifest.get("evaluation_scope") == "conventional40"
             and manifest.get("lock_kind") == "conventional40" and manifest.get("graph_analysis_performed") is False
             and manifest.get("graph_models") == [] and manifest.get("full_registry_completion_asserted") is False
             and set(manifest.get("covered_model_ids", [])) == set(current["scope_model_ids"])
             and len(manifest["covered_model_ids"]) == 40
             and tuple(manifest.get("required_models", [])) == RF_MODELS
             and tuple(manifest.get("RF_comparators", [])) == RF_MODELS,
             "RF cache is not the complete graph-free conventional40 producer")
    for field, expected in (("evaluation_lock_sha256", current["scoped_lock_sha256"]),
                            ("scoped_lock_sha256", current["scoped_lock_sha256"]),
                            ("protocol_sha256", current["protocol_sha256"]),
                            ("source_rows_fingerprint", current["source_rows_fingerprint"])):
        _require(manifest.get(field) == expected, f"RF producer identity changed: {field}")
    _require(manifest.get("global_lock_sha256") is None and manifest.get("split_mode") == protocol["split_mode"]
             and manifest.get("historical_results_used") is False, "RF producer misstates run/scope/history")
    for name, digest in manifest.get("files", {}).items():
        _require(Path(name).name == name and _sha(cache / name) == digest, f"RF cached producer artifact changed: {name}")
    _require(set(RF_FILES) <= set(manifest["files"]), "RF cache does not contain all17 expected reusable files")
    full_ids = {job["model_id"] for job in protocol["registry"]}
    physical_lock = root / ("conventional40_lock.json" if allow_scoped_preflight else "model_lock.json")
    _require(physical_lock.is_file() and _read(physical_lock) == bundle.get("lock"), "Current full/scoped evaluation bundle is not physically locked")
    expected_ids = set(current["scope_model_ids"]) if allow_scoped_preflight else full_ids
    _require(len(expected_ids) == (40 if allow_scoped_preflight else 43)
             and set(bundle["lock"]["expected_model_ids"]) == expected_ids
             and len(bundle["lock"]["expected_model_ids"]) == len(expected_ids)
             and len(bundle["lock"].get("models", [])) == len(expected_ids)
             and {record["model_id"] for record in bundle["lock"]["models"]} == expected_ids
             and set(bundle["metadata"]) == expected_ids
             and bundle["lock"].get("schema_version") == 1 and bundle["lock"].get("locked_at_utc")
             and bundle["lock"].get("test_evaluation_authorized") is True,
             "Full43 RF cache reuse requires the genuine full registered bundle")
    _require(source_file_fields(bundle["lock"]) == source_file_fields(current), "Current evaluation/cache input mismatch")
    for field in ("protocol_sha256", "source_rows_fingerprint"):
        _require(bundle["lock"].get(field) == current[field], f"Current evaluation/cache identity mismatch: {field}")
    _require(_row_fingerprint(rows) == current["source_rows_fingerprint"], "Current caller DATA differs from RF cache")
    development, test, folds = (np.asarray(bundle[key]) for key in ("development_index", "test_index", "folds"))
    ids = np.asarray([row["compound_id"] for row in rows], dtype=str)
    _require(len(set(ids.tolist())) == len(ids) and set(development).isdisjoint(set(test)), "Current DATA row IDs/partition overlap")
    _require(development.ndim == test.ndim == folds.ndim == 1 and len(folds) == len(development)
             and len(set(development.tolist())) == len(development) and len(set(test.tolist())) == len(test)
             and sorted([*development.tolist(), *test.tolist()]) == list(range(len(rows))), "Current RF row-index partition is incomplete/duplicated")
    expected_cases, missing = _main_case_identity(rows, bundle)
    _check_cases(main_cases, expected_cases)
    _check_cases(_csv_rows(cache / "cases.csv"), expected_cases)
    case_selection = _read(cache / "case_selection.json")
    _require(case_selection.get("main_cases") == [row["compound_id"] for row in expected_cases]
             and case_selection.get("missing_main_categories") == missing
             and case_selection.get("supplementary_cases") == [], "RF cached case selection differs from current four main cases")
    input_metadata = {model: _read(root / model / "metadata.json") for model in RF_MODELS}
    checks = {}
    for model in RF_MODELS:
        representation = model.split("__")[0]
        X, names = np.asarray(matrices[representation]), np.asarray(feature_names[representation], dtype=str)
        _require(X.shape == (len(rows), len(names)) and np.isfinite(X).all(), "Current raw RF matrix has invalid shape/values")
        for field in ("model_sha256", "predictions_sha256", "protocol_sha256"):
            _require(bundle["metadata"][model].get(field) == input_metadata[model].get(field), "Current RF bundle metadata changed")
        model_checks = {}
        for partition, index in (("oof", development), ("test", test)):
            with np.load(cache / f"{model}_shap_{partition}.npz", allow_pickle=False) as values:
                _require(np.array_equal(values["compound_id"], ids[index]) and np.array_equal(values["row_index"], index), "RF cached SHAP IDs/row order changed")
                _require(np.array_equal(values["feature_names"], names) and np.array_equal(values["X"], X[index]), "RF cached SHAP raw X/feature order differs")
                if partition == "oof":
                    _require(np.array_equal(values["fold"], folds), "RF cached OOF fold assignment differs")
                p = bundle["predictions"][model]["development" if partition == "oof" else "test"]
                _require(np.array_equal(values["probability"], p), "RF cached SHAP probability differs from current locked predictions")
                phi, base = values["shap"], values["base_value"]
                _require(phi.shape == (len(index), len(names)) and base.shape == (len(index),)
                         and np.isfinite(phi).all() and np.isfinite(base).all(), "RF cached SHAP values invalid")
                errors = np.abs(base + phi.sum(axis=1) - p)
                _require(np.allclose(errors, 0, atol=1e-6, rtol=0), "RF cached SHAP additivity failed")
                model_checks[partition + "_max_additivity_error"] = float(errors.max())
        with np.load(cache / f"{model}_permutation.npz", allow_pickle=False) as values:
            _require(np.array_equal(values["feature_names"], names)
                     and int(values["seed"]) == signature["constants"]["PERMUTATION_SEED"], "RF permutation seed/feature order changed")
            repeated = values["repeated_auc_drop"]
            _require(repeated.shape == (len(names), signature["constants"]["PERMUTATION_REPEATS"])
                     and np.isfinite(repeated).all(), "RF permutation repetitions changed")
        provenance = manifest["permutation"][model]
        _require(provenance.get("repeats") == signature["constants"]["PERMUTATION_REPEATS"]
                 and provenance.get("seed") == signature["constants"]["PERMUTATION_SEED"], "RF permutation provenance changed")
        _require(model in manifest.get("shap_checks", {}), "RF SHAP provenance is absent")
        checks[model] = model_checks
    return {"status": "scoped_preflight_passed_pending_global43_lock" if allow_scoped_preflight else "eligible_same_run_RF_cache",
            "eligible_for_full_reuse": not allow_scoped_preflight, "same_run_cache_only": True,
            "historical_results_used": False, "model_root": str(root),
            "producer_manifest_sha256": _sha(manifest_path), "rf_input_binding_sha256": _sha(sidecar_path),
            "current_evaluation_lock_sha256": _sha(physical_lock), "scoped_producer_lock_sha256": current["scoped_lock_sha256"],
            "protocol_sha256": current["protocol_sha256"], "source_rows_fingerprint": current["source_rows_fingerprint"],
            "eligible_files": {name: {"source": str(cache / name), "sha256": manifest["files"][name]} for name in RF_FILES},
            "files_not_to_copy": ["cases.csv", "case_selection.json", "rf_input_binding.json", "rf_producer_source.py", "explain_manifest.json"],
            "shap_checks": manifest["shap_checks"], "permutation": manifest["permutation"],
            "MACCS_atom_color_scale": manifest.get("MACCS_atom_color_scale"), "independent_additivity_checks": checks,
            "main_case_ids": [row["compound_id"] for row in expected_cases],
            "producer_manifest": manifest}
