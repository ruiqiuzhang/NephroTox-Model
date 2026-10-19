"""Exercise all 43 modeling APIs on 96 current development records only.

Reduced budgets check fitting, finite probabilities, deterministic repeated
inference, same-seed refitting, strict JSON metadata, and joblib round trips.
This is not nested CV, model selection, test evaluation or a performance result.
Temporary weights are deleted; only the API check report is published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def select_development_rows(context, seed=20260810):
    """Select 48 rows per class without exposing any held-out label."""
    import numpy as np
    if not np.all(context["y"][context["test"]] == -1):
        raise ValueError("Smoke testing requires the test-blind training context")
    rng = np.random.default_rng(seed)
    selected, counts = [], {0: 0, 1: 0}
    for index in rng.permutation(context["development"]):
        label = int(context["y"][index])
        if label not in counts:
            raise ValueError("Nonbinary development label")
        if counts[label] < 48:
            selected.append(int(index))
            counts[label] += 1
        if len(selected) == 96:
            break
    indices = np.asarray(sorted(selected), dtype=int)
    if len(indices) != 96 or counts != {0: 48, 1: 48}:
        raise ValueError("The development source cannot provide the prescribed 96 smoke-test rows")
    if len(set(context["groups"][indices])) < 30 or np.intersect1d(indices, context["test"]).size:
        raise ValueError("Smoke rows have too few structure groups or contain held-out records")
    return indices


def smoke_params(job):
    """Reduce fit budgets only; never overwrite any formal candidate/protocol."""
    from nephrotox.modeling_classical import CANDIDATES
    from nephrotox.modeling_graph import GRAPH_CANDIDATES
    if job["representation"] == "graph":
        params = dict(GRAPH_CANDIDATES[job["estimator_key"]][0])
        params.update(max_epochs=2, patience=1)
    else:
        params = dict(CANDIDATES[job["estimator_key"]][0])
        if job["estimator_key"] == "rf":
            params["n_estimators"] = 5
        elif job["estimator_key"] == "mlp":
            params["max_iter"] = 2
        elif job["estimator_key"] == "xgb":
            params["n_estimators"] = 3
        elif job["estimator_key"] == "dnn":
            params.update(max_epochs=2, patience=1)
    return params


def _estimator(job, params, seed, threads):
    if job["representation"] == "graph":
        from nephrotox.modeling_graph import make_graph_estimator
        return make_graph_estimator(job["estimator_key"], params, seed=seed, seeds=[seed], n_jobs=threads)
    from nephrotox.modeling_classical import make_estimator
    return make_estimator(job["estimator_key"], params, job["representation"], seed=seed,
                          seeds=[seed], n_jobs=threads)


def _probability(estimator, values):
    import numpy as np
    result = np.asarray(estimator.predict_proba(values), dtype=float)
    if result.shape != (len(values), 2) or not np.isfinite(result).all():
        raise ValueError("Prediction probability shape/finite check failed")
    # XGBoost returns float32 class probabilities, so the two columns may sum
    # to 1 +/- float32 rounding after conversion to float64.
    if np.any((result < 0) | (result > 1)) or not np.allclose(result.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Prediction probabilities do not lie on the binary simplex")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Frozen source features/splits, not trained-model results")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "outputs/MODELING_SMOKE_CHECK.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("CPU thread limit must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from nephrotox.modeling_common import prepare_count_features, raw_matrix, registry, training_context
    from nephrotox.modeling_graph import GRAPH_SCHEMA_VERSION
    context = training_context(args.results)
    indices = select_development_rows(context, args.seed)
    prepare_count_features(context)
    jobs = registry()
    matrices = {representation: raw_matrix(representation, context)[indices]
                for representation in dict.fromkeys(j["representation"] for j in jobs if j["representation"] != "graph")}
    labels, groups, ids = context["y"][indices], context["groups"][indices], context["ids"][indices]
    packages = ("numpy", "scipy", "scikit-learn", "rdkit", "torch", "xgboost", "joblib")
    report = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_scope": "reduced-budget API smoke check only; not nested CV, parameter optimization, or predictive-performance evidence",
        "formal_models_trained": False, "test_evaluated": False, "test_rows_used": 0,
        "source_excel_sha256": context["source"]["sha256"], "source_rows_fingerprint": context["source"]["rows_fingerprint"],
        "python": platform.python_version(), "versions": {name: importlib.metadata.version(name) for name in packages},
        "graph_schema_version": GRAPH_SCHEMA_VERSION, "threads": args.threads, "seed": args.seed,
        "probability_simplex_absolute_tolerance": 1e-6,
        "n_development_smoke_rows": len(indices), "n_structure_groups": len(set(groups)),
        "smoke_label_counts": {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)},
        "development_sample_ids_sha256": hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest(),
        "requested_model_count": len(jobs), "checks": []}
    scratch_parent = (ROOT.parent / "outputs").resolve()
    if not scratch_parent.is_relative_to(ROOT.parent.resolve()):
        raise ValueError("Smoke scratch directory must remain within the project outputs directory")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    # The absolute temporary directory is created inside the verified project
    # output root; TemporaryDirectory removes only its own generated contents.
    with tempfile.TemporaryDirectory(prefix="modeling-api-smoke-", dir=scratch_parent) as scratch:
        for job in jobs:
            model_id = job["model_id"]
            record = {"model_id": model_id, "status": "running"}
            try:
                params = smoke_params(job)
                values = context["smiles"][indices] if job["representation"] == "graph" else matrices[job["representation"]]
                estimator = _estimator(job, params, args.seed, args.threads)
                with threadpool_limits(limits=args.threads):
                    estimator.fit(values, labels, groups=groups, sample_ids=ids)
                    json.dumps(estimator.fit_metadata_, allow_nan=False, ensure_ascii=False)
                    first = _probability(estimator, values)
                    second = _probability(estimator, values)
                    if not np.array_equal(first, second):
                        raise ValueError("Repeated inference is not exactly deterministic")
                    checkpoint = Path(scratch) / (model_id + ".joblib")
                    joblib.dump(estimator, checkpoint, compress=3)
                    restored = joblib.load(checkpoint)
                    roundtrip = _probability(restored, values)
                    json.dumps(restored.fit_metadata_, allow_nan=False, ensure_ascii=False)
                    if not np.array_equal(first, roundtrip):
                        raise ValueError("Saved/restored estimator probability values changed")
                    repeated_fit = _estimator(job, params, args.seed, args.threads)
                    repeated_fit.fit(values, labels, groups=groups, sample_ids=ids)
                    json.dumps(repeated_fit.fit_metadata_, allow_nan=False, ensure_ascii=False)
                    refit = _probability(repeated_fit, values)
                    if not np.array_equal(first, refit):
                        raise ValueError("Same-seed refitting is not exactly deterministic")
                record.update(status="passed", probability_shape=list(first.shape), finite_binary_probabilities=True,
                    repeated_inference_exact=True, same_seed_refit_exact=True, joblib_roundtrip_exact=True,
                    metadata_strict_json=True, smoke_budget=params, evaluation_seeds=[args.seed])
                checkpoint.unlink()
                del estimator, restored, repeated_fit
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, message=str(error))
            report["checks"].append(record)
            print(f"SMOKE {len(report['checks'])}/{len(jobs)} {model_id}: {record['status']}", flush=True)
            gc.collect()
    report.update(status="passed" if all(r["status"] == "passed" for r in report["checks"]) else "failed",
                  passed_model_count=sum(r["status"] == "passed" for r in report["checks"]),
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), temporary_weights_retained=False,
                  performance_metrics_computed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed_model_count"], "requested": len(jobs)}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
