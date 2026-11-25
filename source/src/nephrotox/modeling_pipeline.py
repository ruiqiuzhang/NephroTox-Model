"""Resumable fixed/search modeling with explicit splits and test-blind final fits."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time
import traceback
import hashlib

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .input_provenance import source_file_fields
from .modeling_common import (SEEDS, training_context, raw_matrix, registry, prepare_count_features,
                              freeze_protocol, sha256, json_write, utc_now)
from .modeling_classical import CANDIDATES, make_estimator, group_splits
from .fixed_parameters import fixed_parameter_records


def _candidate_list(job):
    if job["representation"] == "graph":
        from .modeling_graph import GRAPH_CANDIDATES
        return GRAPH_CANDIDATES[job["estimator_key"]]
    return CANDIDATES[job["estimator_key"]]


def _execution_seed(job, params, split_mode='structure_group', outer_fold=None):
    """Reproduce original fixed classical seed offsets without parameter search."""
    if (split_mode != 'legacy_random' or outer_fold is None or job['representation'] == 'graph'
            or job['estimator_key'] == 'dnn'):
        return SEEDS[0]
    if not isinstance(outer_fold, int) or isinstance(outer_fold, bool) or not 1 <= outer_fold <= 10:
        raise ValueError('Outer fold must be1..10 for original seed execution')
    record = fixed_parameter_records()[job['model_id']]
    candidate = record.get('original_candidate_id_in_code')
    matches = [i for i, values in enumerate(CANDIDATES[job['estimator_key']]) if values == record['params']]
    if not isinstance(candidate, int) or matches != [candidate] or params != record['params']:
        raise ValueError('Original candidate ID must uniquely match the static fixed parameter record')
    inherited = record.get('count_inheritance')
    if inherited:
        binary = fixed_parameter_records()[inherited]
        if candidate != binary['original_candidate_id_in_code'] or record['params'] != binary['params']:
            raise ValueError('Count/binary fixed parameter and seed provenance differ')
    offset = 1000 if job['estimator_key'] == 'svm' else candidate * 100
    return SEEDS[0] + offset + outer_fold - 1


def _estimator(job, params, evaluation, threads, split_mode='structure_group', outer_fold=None):
    seeds = SEEDS if evaluation else [SEEDS[0]]
    execution_seed = _execution_seed(job, params, split_mode, outer_fold)
    if job["representation"] == "graph":
        from .modeling_graph import make_graph_estimator
        return make_graph_estimator(job["estimator_key"], params, seed=SEEDS[0], seeds=seeds, n_jobs=threads)
    return make_estimator(job["estimator_key"], params, job["representation"],
                          seed=execution_seed, seeds=seeds, n_jobs=threads, split_mode=split_mode)


def _search(job, X, context, indices, output, protocol_hash, threads):
    y, groups, ids = context["y"], context["groups"], context["ids"]
    if (y[indices] < 0).any():
        raise ValueError("Holdout labels entered parameter search")
    if context.get('parameter_mode')=='fixed_original':
        record=fixed_parameter_records()[job['model_id']]
        params=record['params']
        selection={'protocol_sha256':protocol_hash,'selection_method':'fixed_original',
            'selected_params':params,'trials':[],
            'selected_candidate':None,'selected_candidate_id_in_code':None,
            'source_original_candidate_number':record.get('original_candidate_number'),
            'source_original_candidate_id_in_code':record.get('original_candidate_id_in_code'),
            'parameter_source':record,'configuration_fixed_before_current_training':True,
            'training_ids':ids[indices].tolist(),'test_labels_accessed':False}
        path=output/'selection.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=selection:
            raise ValueError('Cached fixed configuration differs from the locked parameter source')
        json_write(path,selection)
        return params
    splits = group_splits(y[indices], groups[indices], 3, SEEDS[0])
    json_write(output / "inner_splits.json", [{"fold": j+1, "training_ids": ids[indices[a]].tolist(),
        "validation_ids": ids[indices[b]].tolist(), "group_overlap": 0} for j, (a, b) in enumerate(splits)])
    trials = []
    for ci, params in enumerate(_candidate_list(job)):
        scores = []
        for fi, (local_train, local_val) in enumerate(splits):
            fit, val = indices[local_train], indices[local_val]
            path = output / f"candidate_{ci+1:02d}/inner_{fi+1}/trial.json"
            if path.exists():
                trial = json.loads(path.read_text(encoding="utf-8"))
                if trial["protocol_sha256"] != protocol_hash or trial["params"] != params or trial["training_ids"] != ids[fit].tolist() or trial["validation_ids"] != ids[val].tolist():
                    raise ValueError("Cached tuning trial provenance differs")
                prediction_file = path.parent / "validation_predictions.npz"
                if sha256(prediction_file) != trial["prediction_sha256"]:
                    raise ValueError("Cached tuning predictions integrity failure")
                with np.load(prediction_file, allow_pickle=False) as cached:
                    if not np.array_equal(cached["compound_id"], ids[val]) or not np.isclose(
                            roc_auc_score(y[val], cached["probability"]), trial["roc_auc"], atol=1e-12, rtol=0):
                        raise ValueError("Cached tuning AUC/ID differs from saved probabilities")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                estimator = (_estimator(job, params, False, threads, 'legacy_random')
                             if context.get('split_mode', 'structure_group') == 'legacy_random'
                             else _estimator(job, params, False, threads))
                estimator.fit(X[fit], y[fit], groups=groups[fit], sample_ids=ids[fit], log_dir=path.parent)
                probability = estimator.predict_proba(X[val])[:, 1]
                trial = {"protocol_sha256": protocol_hash, "params": params, "candidate": ci+1, "inner_fold": fi+1,
                         "candidate_id_in_code": ci,
                         "training_ids": ids[fit].tolist(), "validation_ids": ids[val].tolist(),
                         "roc_auc": float(roc_auc_score(y[val], probability)),
                         "elapsed_seconds": time.perf_counter()-started, "fit_metadata": estimator.fit_metadata_}
                np.savez_compressed(path.parent / "validation_predictions.npz", compound_id=ids[val], probability=probability)
                trial["prediction_sha256"] = sha256(path.parent / "validation_predictions.npz")
                json_write(path, trial)
                del estimator
                gc.collect()
            scores.append(trial["roc_auc"])
        trials.append({"candidate": ci+1, "params": params, "inner_auc": scores, "mean_inner_auc": float(np.mean(scores))})
        print(f'{job["model_id"]} {output.name} candidate {ci+1}: inner AUC {np.mean(scores):.4f}', flush=True)
    best = max(range(len(trials)), key=lambda j: trials[j]["mean_inner_auc"])
    json_write(output / "selection.json", {"protocol_sha256": protocol_hash, "selected_candidate": best+1,
               "selected_candidate_id_in_code": best,
               "selection_method": "seeded_random_search" if job["representation"] == "graph" else "finite_prespecified_candidates",
               "selected_params": trials[best]["params"], "trials": trials, "tie_rule": "first candidate"})
    return trials[best]["params"]


def run_job(model_id, results_dir=None, threads=2):
    from .modeling_graph import GRAPH_CANDIDATES
    context = training_context(results_dir)
    protocol, protocol_hash = freeze_protocol(context, GRAPH_CANDIDATES)
    jobs = {j["model_id"]: j for j in protocol['registry']}
    job = jobs[model_id]
    output = context["results_dir"] / "modeling" / model_id
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete") and metadata["protocol_sha256"] == protocol_hash:
            for name, key in [("predictions.npz", "predictions_sha256"), ("model.joblib", "model_sha256")]:
                if sha256(output / name) != metadata[key]:
                    raise ValueError("Completed model integrity failure")
            print(f"Already complete: {model_id}", flush=True)
            return metadata
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "status.json", {"state": "running", "started_at_utc": utc_now(), "pid": os.getpid(), "protocol_sha256": protocol_hash})
    X = context["smiles"] if job["representation"] == "graph" else raw_matrix(job["representation"], context)
    dev, test = context["development"], context["test"]
    y, groups, ids = context["y"], context["groups"], context["ids"]
    oof = np.full(len(ids), np.nan)
    seed_oof = np.full((3, len(ids)), np.nan)
    start = time.perf_counter()
    outer_params = []
    try:
        for fold in range(1, 11):
            fit = dev[context["fold"][dev] != fold]
            val = dev[context["fold"][dev] == fold]
            part = output / f"outer_{fold:02d}"
            saved = part / "outer_predictions.npz"
            completed = part / "complete.json"
            if completed.exists() and saved.exists():
                done = json.loads(completed.read_text(encoding="utf-8"))
                if done["protocol_sha256"] != protocol_hash or done["predictions_sha256"] != sha256(saved):
                    raise ValueError("Outer-fold cache provenance mismatch")
                selection = json.loads((part / "selection.json").read_text(encoding="utf-8"))
                if sha256(part / "model.joblib") != done["model_sha256"] or selection["selected_params"] != done["params"]:
                    raise ValueError("Outer-fold model/selection integrity failure")
                with np.load(saved, allow_pickle=False) as values:
                    if not np.array_equal(values["compound_id"], ids[val]):
                        raise ValueError("Outer-fold IDs differ")
                    oof[val] = values["probability"]
                    seed_oof[:, val] = values["seed_probability"]
                outer_params.append(done["params"])
                continue
            group_overlap = len(set(groups[fit]) & set(groups[val]))
            if context.get('split_mode', 'structure_group') == 'structure_group' and group_overlap:
                raise ValueError("Outer group overlap")
            params = _search(job, X, context, fit, part, protocol_hash, threads)
            estimator = _estimator(job, params, True, threads, context.get('split_mode', 'structure_group'), outer_fold=fold)
            estimator.fit(X[fit], y[fit], groups=groups[fit], sample_ids=ids[fit], log_dir=part / "refit")
            oof[val] = estimator.predict_proba(X[val])[:, 1]
            member = estimator.predict_seed_probabilities(X[val]) if hasattr(estimator, "predict_seed_probabilities") else np.tile(oof[val], (3,1))
            seed_oof[:, val] = member
            joblib.dump(estimator, part / "model.joblib", compress=3)
            np.savez_compressed(saved, compound_id=ids[val], probability=oof[val], seed_probability=member)
            json_write(completed, {"protocol_sha256": protocol_hash, "params": params, "n_training": len(fit), "n_validation": len(val),
                "group_overlap": group_overlap, "split_mode": context.get('split_mode', 'structure_group'),
                "execution_seed": estimator.fit_metadata_.get('seed'),
                "execution_seeds": estimator.fit_metadata_.get('seeds'),
                "predictions_sha256": sha256(saved), "model_sha256": sha256(part / "model.joblib"),
                "outer_roc_auc": float(roc_auc_score(y[val], oof[val]))})
            outer_params.append(params)
            print(f"{model_id} outer {fold}/10 complete", flush=True)
            del estimator
            gc.collect()
        if not np.isfinite(oof[dev]).all() or np.isfinite(oof[test]).any():
            raise ValueError("OOF coverage/test contamination failure")
        params = _search(job, X, context, dev, output / "full_development_search", protocol_hash, threads)
        # Parameters are locked before this model has access to holdout structures for inference.
        json_write(output / "configuration_lock.json", {"locked_at_utc": utc_now(), "protocol_sha256": protocol_hash,
            "params": params, "training_ids": ids[dev].tolist(), "test_labels_accessed": False})
        estimator = _estimator(job, params, True, threads, context.get('split_mode', 'structure_group'))
        estimator.fit(X[dev], y[dev], groups=groups[dev], sample_ids=ids[dev], log_dir=output / "final_fit")
        model_path = output / "model.joblib"
        joblib.dump(estimator, model_path, compress=3)
        prediction_start = time.perf_counter()
        test_prob = estimator.predict_proba(X[test])[:, 1]
        inference_seconds = time.perf_counter() - prediction_start
        member_test = estimator.predict_seed_probabilities(X[test]) if hasattr(estimator, "predict_seed_probabilities") else np.tile(test_prob, (3,1))
        # Check serialization on inference alone, without reading held-out outcomes.
        restored = joblib.load(model_path)
        if not np.allclose(restored.predict_proba(X[test])[:,1], test_prob, rtol=0, atol=1e-7):
            raise ValueError("Saved-model prediction roundtrip differs")
        np.savez_compressed(output / "predictions.npz", compound_id=ids, development_index=dev, test_index=test,
            oof_probability=oof[dev], test_probability=test_prob,
            oof_seed_probability=seed_oof[:,dev], test_seed_probability=member_test)
        metadata = {**job, "complete": True, "protocol_sha256": protocol_hash, "params": params,
            "outer_params": outer_params, "selection_metric": 'fixed_original' if context.get('parameter_mode')=='fixed_original' else "ROC_AUC",
            "source_rows_fingerprint": context["source"]["rows_fingerprint"],
            "model_file": "model.joblib", "model_sha256": sha256(model_path), "model_bytes": model_path.stat().st_size,
            "predictions_sha256": sha256(output / "predictions.npz"), "total_elapsed_seconds": time.perf_counter()-start,
            "final_fit_seconds": estimator.fit_metadata_["fit_seconds"], "test_inference_seconds": inference_seconds,
            "training_threads": threads, "final_fit_metadata": estimator.fit_metadata_, "test_labels_accessed": False,
            "completed_at_utc": utc_now()}
        json_write(metadata_path, metadata)
        json_write(output / "status.json", {"state": "complete", "completed_at_utc": utc_now()})
        print(f"MODEL COMPLETE: {model_id}", flush=True)
        return metadata
    except Exception as error:
        json_write(output / "status.json", {"state": "failed", "error_type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "failed_at_utc": utc_now()})
        raise


def lock_all(results_dir=None):
    context = training_context(results_dir)
    out = context["results_dir"] / "modeling"
    protocol_path = out / "protocol.json"
    entries = []
    for job in registry():
        folder = out / job["model_id"]
        path = folder / "metadata.json"
        if not path.exists():
            raise ValueError(f"Cannot unlock test evaluation: incomplete {job['model_id']}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not meta.get("complete") or meta["protocol_sha256"] != sha256(protocol_path):
            raise ValueError("Incomplete/mixed protocol model")
        for file, key in [("predictions.npz", "predictions_sha256"), ("model.joblib", "model_sha256")]:
            if sha256(folder / file) != meta[key]:
                raise ValueError("Model/prediction hash mismatch")
        entries.append({"model_id": job["model_id"], "metadata_sha256": sha256(path),
                         "predictions_sha256": meta["predictions_sha256"], "model_sha256": meta["model_sha256"]})
    lock = {"schema_version": 1, "source_rows_fingerprint": context["source"]["rows_fingerprint"],
        **source_file_fields(context["source"]), "protocol_sha256": sha256(protocol_path),
        "expected_model_ids": [j["model_id"] for j in registry()], "models": entries,
        "test_evaluation_authorized": True, "locked_at_utc": utc_now()}
    json_write(out / "model_lock.json", lock)
    return lock
