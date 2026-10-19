"""Measure current, globally locked models and exact DATA feature generation.

No model fitting, threshold selection or test-label scoring occurs. Numerical
features are rebuilt from the current source SMILES and compared exactly with
the frozen matrices. Predictions are timed after one explicit warm-up.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, records):
    records = list(records)
    if not records:
        raise ValueError("An empty benchmark is not a completed result")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for r in records for key in r)))
        writer.writeheader()
        writer.writerows(records)


def hardware():
    """Read device information without recording account names or host names."""
    result = {"os": platform.system(), "os_release": platform.release(), "os_version": platform.version(),
              "machine": platform.machine(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()}
    if platform.system() == "Windows":
        import ctypes
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_pagefile", ctypes.c_ulonglong), ("available_pagefile", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result.update(total_physical_memory_bytes=int(status.total_physical),
                          available_physical_memory_bytes_at_start=int(status.available_physical),
                          memory_load_percent=int(status.memory_load))
    elif platform.system() == "Linux":
        cpuinfo, meminfo = Path("/proc/cpuinfo"), Path("/proc/meminfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    result["processor"] = line.split(":", 1)[1].strip()
                    break
        if meminfo.exists():
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in meminfo.read_text().splitlines()}
            result["total_physical_memory_bytes"] = int(fields["MemTotal"].split()[0]) * 1024
            result["available_physical_memory_bytes_at_start"] = int(fields["MemAvailable"].split()[0]) * 1024
    return result


def _threadpools():
    from threadpoolctl import threadpool_info
    keys = ("user_api", "internal_api", "num_threads", "prefix", "version", "threading_layer", "architecture")
    return [{key: item[key] for key in keys if key in item} for item in threadpool_info()]


def _numeric_hash(matrix):
    """Transport-independent array digest retaining the exact matrix dtype."""
    import numpy as np
    matrix = np.asarray(matrix)
    dtype = matrix.dtype.newbyteorder("<")
    digest = hashlib.sha256(json.dumps({"shape": list(matrix.shape), "dtype": dtype.str}, sort_keys=True).encode())
    digest.update(matrix.astype(dtype, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _fresh_features(rows, representation):
    """Include SMILES parsing, raw numerical computation and array allocation."""
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from nephrotox.features import compute_descriptors, fingerprint_matrix, load_descriptor_names
    molecules = [Chem.MolFromSmiles(row["standardized_smiles"]) for row in rows]
    if any(mol is None for mol in molecules):
        raise ValueError("A current DATA molecule failed SMILES parsing")
    if representation == "rdkit166":
        names = load_descriptor_names(ROOT / "configs/rdkit_descriptors_166.txt")
        matrix, failures = compute_descriptors(molecules, [row["compound_id"] for row in rows], names)
        if failures:
            raise ValueError("Descriptor recalculation produced a failure during benchmarking")
        return matrix
    if "count" not in representation:
        return fingerprint_matrix(molecules, representation)
    generators = {
        "morgan_count2048": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
        "atompair_count1024": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
        "torsion_count1024": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True),
    }
    width = 2048 if representation == "morgan_count2048" else 1024
    matrix = np.zeros((len(rows), width), dtype=np.int32)
    generator = generators[representation]()
    for i, molecule in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(molecule), matrix[i])
    return matrix


def benchmark_features(rows, context, binary_repeats=3, count_repeats=1):
    import numpy as np
    from nephrotox.modeling_common import BINARY_REPRESENTATIONS, COUNT_REPRESENTATIONS, raw_matrix
    from nephrotox.modeling_graph import _cached_graph, graph_feature_schema, molecular_graph
    measurements, summary = [], []
    matrices = {}
    for representation in BINARY_REPRESENTATIONS + COUNT_REPRESENTATIONS:
        frozen = raw_matrix(representation, context)
        matrices[representation] = frozen
        times = []
        repeat_count = count_repeats if representation in COUNT_REPRESENTATIONS else binary_repeats
        for repeat in range(1, repeat_count + 1):
            started = time.perf_counter()
            rebuilt = _fresh_features(rows, representation)
            elapsed = time.perf_counter() - started
            if rebuilt.shape != frozen.shape or rebuilt.dtype != frozen.dtype or not np.array_equal(rebuilt, frozen):
                raise ValueError(f"Benchmark recalculation does not exactly match frozen feature matrix: {representation}")
            times.append(elapsed)
            measurements.append({"representation": representation, "repeat": repeat, "n_compounds": len(rows),
                "n_features": frozen.shape[1], "seconds": elapsed, "milliseconds_per_compound": 1000 * elapsed / len(rows),
                "dtype": str(rebuilt.dtype), "exact_frozen_matrix_match": True, "numeric_sha256": _numeric_hash(rebuilt),
                "timing_scope": "fresh SMILES parsing + feature generation + array allocation; excludes workbook reading and disk writing"})
        summary.append({"representation": representation, "n_compounds": len(rows), "n_features": frozen.shape[1],
            "n_repeats": repeat_count, "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
            "sample_sd_seconds": statistics.stdev(times) if len(times) > 1 else None,
            "mean_milliseconds_per_compound": 1000 * statistics.mean(times) / len(rows),
            "all_exact_frozen_matrix_match": True})
    # Graph tensors are shared by all three graph architectures. Their complete
    # frozen schema is verified against the locked fitted-model metadata below.
    graph_hash = None
    graph_times, graph_shape = [], None
    for repeat in range(1, binary_repeats + 1):
        _cached_graph.cache_clear()
        started = time.perf_counter()
        graphs = [molecular_graph(row["standardized_smiles"]) for row in rows]
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256()
        for row, graph in zip(rows, graphs):
            digest.update(row["compound_id"].encode("utf-8"))
            for value in (graph.x, graph.edge_index, graph.edge_attr, graph.reverse_edge, graph.bond_index):
                digest.update(_numeric_hash(value).encode("ascii"))
        current = digest.hexdigest()
        if graph_hash is not None and current != graph_hash:
            raise ValueError("Graph featurization changed across benchmark repeats")
        graph_hash = current
        graph_shape = {"total_atoms": sum(len(g.x) for g in graphs), "total_directed_edges": sum(g.edge_index.shape[1] for g in graphs)}
        graph_times.append(elapsed)
        measurements.append({"representation": "graph", "repeat": repeat, "n_compounds": len(rows),
            "n_features": graph_feature_schema()["atom_dimension"], "seconds": elapsed,
            "milliseconds_per_compound": 1000 * elapsed / len(rows), "dtype": "per schema", "numeric_sha256": current,
            "exact_frozen_matrix_match": "not applicable: graph tensors have a frozen schema and deterministic repeat check",
            "timing_scope": "cold molecular graph cache; fresh SMILES parsing + atom/bond tensors; excludes disk writing"})
    summary.append({"representation": "graph", "n_compounds": len(rows), "n_features": graph_feature_schema()["atom_dimension"],
        "n_repeats": binary_repeats, "mean_seconds": statistics.mean(graph_times), "median_seconds": statistics.median(graph_times),
        "sample_sd_seconds": statistics.stdev(graph_times) if len(graph_times) > 1 else None,
        "mean_milliseconds_per_compound": 1000 * statistics.mean(graph_times) / len(rows),
        "all_exact_frozen_matrix_match": "not applicable; frozen graph feature schema and repeat digest checked"})
    return {"measurements": measurements, "summary": summary, "matrices": matrices,
            "graph_feature_schema": graph_feature_schema(), "graph_numeric_sha256": graph_hash, "graph_shape": graph_shape}


def _checked_probability(raw_probability, n_rows):
    import numpy as np
    probability = np.asarray(raw_probability, dtype=float)
    if probability.shape != (n_rows, 2) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probability output during inference benchmarking")
    return probability[:, 1]


def _predict_checked(estimator, values):
    return _checked_probability(estimator.predict_proba(values), len(values))


def benchmark_models(results_dir, bundle, feature_bundle, threads, batch_repeats=3, single_repeats=10):
    import gc
    import joblib
    import numpy as np
    from nephrotox.modeling_graph import graph_feature_schema
    from threadpoolctl import threadpool_limits
    dev, test = bundle["development_index"], bundle["test_index"]
    first_ten = dev[:10]
    if len(first_ten) != 10 or len(test) != 335:
        raise ValueError("Runtime benchmarking requires the frozen 10 development examples and 335 held-out examples")
    from nephrotox.data import load_data
    rows, _ = load_data()
    ids = np.asarray([r["compound_id"] for r in rows])
    smiles = np.asarray([r["standardized_smiles"] for r in rows])
    measurements, summary = [], []
    for model_id in bundle["lock"]["expected_model_ids"]:
        metadata = bundle["metadata"][model_id]
        if metadata.get("training_threads") != threads:
            raise ValueError(f"Benchmark --threads must match locked model runtime threads: {model_id}")
        started = time.perf_counter()
        estimator = joblib.load(Path(results_dir) / model_id / metadata["model_file"])
        load_seconds = time.perf_counter() - started
        if metadata["representation"] == "graph":
            if estimator.fit_metadata_["feature_schema"] != graph_feature_schema():
                raise ValueError("Locked graph model feature schema differs from current feature calculation")
            matrix = smiles
        else:
            matrix = feature_bundle["matrices"][metadata["representation"]]
        batch_values = matrix[test]
        with threadpool_limits(limits=threads):
            # The graph cache has been populated in feature benchmarking. Every
            # representation therefore measures loaded-model prediction, with
            # feature construction recorded separately above.
            warm = _predict_checked(estimator, batch_values)
            if not np.allclose(warm, bundle["predictions"][model_id]["test"], rtol=0, atol=1e-7):
                raise ValueError(f"Benchmark model predictions differ from the locked saved probabilities: {model_id}")
            batch_times = []
            for repeat in range(1, batch_repeats + 1):
                started = time.perf_counter()
                raw_probability = estimator.predict_proba(batch_values)
                elapsed = time.perf_counter() - started
                probability = _checked_probability(raw_probability, len(test))
                if not np.allclose(probability, warm, rtol=0, atol=1e-7):
                    raise ValueError("Batch inference changed across identical benchmark repeats")
                batch_times.append(elapsed)
                measurements.append({"model_id": model_id, "measurement": "batch_335", "repeat": repeat,
                    "compound_id": "fixed held-out 335 in DATA order", "n_compounds": len(test), "seconds": elapsed,
                    "threads": threads, "model_sha256": metadata["model_sha256"]})
            single_times = []
            for index in first_ten:
                single_values = matrix[[index]]
                expected = _predict_checked(estimator, single_values)
                for repeat in range(1, single_repeats + 1):
                    started = time.perf_counter()
                    raw_probability = estimator.predict_proba(single_values)
                    elapsed = time.perf_counter() - started
                    probability = _checked_probability(raw_probability, 1)
                    if not np.allclose(probability, expected, rtol=0, atol=1e-7):
                        raise ValueError("Single-compound inference changed across benchmark repeats")
                    single_times.append(elapsed)
                    measurements.append({"model_id": model_id, "measurement": "single_compound", "repeat": repeat,
                        "compound_id": ids[index], "n_compounds": 1, "seconds": elapsed, "threads": threads,
                        "model_sha256": metadata["model_sha256"]})
        summary.append({"model_id": model_id, "algorithm": metadata["algorithm"], "representation": metadata["representation"],
            "role": metadata["role"], "threads": threads, "model_bytes": metadata["model_bytes"], "model_load_seconds": load_seconds,
            "batch_n_compounds": len(test), "batch_n_repeats": batch_repeats,
            "batch_mean_seconds": statistics.mean(batch_times), "batch_median_seconds": statistics.median(batch_times),
            "batch_sd_seconds": statistics.stdev(batch_times),
            "batch_mean_milliseconds_per_compound": 1000 * statistics.mean(batch_times) / len(test),
            "single_n_compounds": len(first_ten), "single_repeats_per_compound": single_repeats,
            "single_total_measurements": len(single_times), "single_mean_milliseconds": 1000 * statistics.mean(single_times),
            "single_median_milliseconds": 1000 * statistics.median(single_times),
            "single_p95_milliseconds": 1000 * float(np.quantile(single_times, 0.95)),
            "single_sd_milliseconds": 1000 * statistics.stdev(single_times),
            "prediction_scope": "loaded model including fitted preprocessing and probability aggregation; precomputed numerical features/cached graphs; feature construction timed separately"})
        print(f"BENCHMARK COMPLETE: {model_id}", flush=True)
        del estimator
        gc.collect()
    return measurements, summary, ids[first_ten].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    # Numerical imports follow explicit thread bounds.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import torch
    from nephrotox.data import load_data
    from nephrotox.modeling_common import registry, training_context
    from nephrotox.modeling_reporting import load_locked_predictions
    torch.set_num_threads(args.threads)
    rows, source = load_data()
    assignments = json.loads((args.results.parent / "split_payload.json").read_text(encoding="utf-8"))["assignments"]
    bundle = load_locked_predictions(args.results, rows, assignments, expected_model_ids=[r["model_id"] for r in registry()])
    output = (args.output if args.output else args.results / "benchmarks").resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = training_context(args.results.parent)
    start = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    packages = ("numpy", "scipy", "scikit-learn", "rdkit", "torch", "xgboost", "joblib", "threadpoolctl")
    environment = {"schema_version": 1, "status": "running", "started_at_utc": start,
        "hardware": hardware(), "python": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "requested_cpu_threads": args.threads, "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "device": "CPU", "threadpools_before": _threadpools(),
        "source_excel_sha256": source["sha256"], "source_rows_fingerprint": source["rows_fingerprint"],
        "global_lock_sha256": _digest(args.results / "model_lock.json"), "protocol_sha256": bundle["lock"]["protocol_sha256"],
        "batch_repetitions": 3, "single_repetitions_per_compound": 10, "single_compound_selection": "first ten development rows in current DATA order",
        "feature_repetitions_binary_and_descriptors": 3, "feature_repetitions_count": 1,
        "test_labels_scored": False, "warmup": "one 335-record batch, and one call for each single-compound example, before timed repeats",
        "feature_time_includes": "fresh SMILES parsing, feature generation and array allocation; graph cache cleared each feature repeat",
        "model_time_excludes": "workbook reading, disk writing, feature generation, model loading (load separately measured)",
        "model_load_scope": "joblib deserialization wall time; operating-system file cache is not flushed",
        "timing_warning": "Wall times depend on concurrent system workload; three batch repeats and 100 single calls are descriptive measurements"}
    _json(output / "runtime_environment.json", environment)
    features = benchmark_features(rows, context)
    _csv(output / "feature_measurements.csv", features["measurements"])
    _csv(output / "feature_benchmark.csv", features["summary"])
    _json(output / "graph_feature_benchmark.json", {"schema": features["graph_feature_schema"],
          "numeric_sha256": features["graph_numeric_sha256"], "shape": features["graph_shape"],
          "frozen_matrix_comparison": "not applicable; schema compared with all locked graph models, repeat numerical digests identical"})
    measurements, summary, single_ids = benchmark_models(args.results, bundle, features, args.threads)
    _csv(output / "runtime_measurements.csv", measurements)
    _csv(output / "runtime_benchmark.csv", summary)
    environment.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_total_seconds=time.perf_counter() - started, fixed_single_compound_ids=single_ids,
        threadpools_after=_threadpools(), torch_cpu_threads_after=torch.get_num_threads(), model_count=len(summary))
    _json(output / "runtime_environment.json", environment)
    _json(output / "benchmark_manifest.json", {"schema_version": 1, "status": "complete", "n_models": len(summary),
          "source_excel_sha256": source["sha256"], "source_rows_fingerprint": source["rows_fingerprint"],
          "global_lock_sha256": environment["global_lock_sha256"], "protocol_sha256": environment["protocol_sha256"],
          "files": {p.name: _digest(p) for p in output.iterdir() if p.name != "benchmark_manifest.json" and p.suffix in (".json", ".csv")}})
    print(json.dumps({"status": "complete", "model_count": len(summary), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
