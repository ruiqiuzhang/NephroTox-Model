"""Full DATA-only modeling: prepare, resume jobs, freeze, then evaluate."""
from pathlib import Path
import argparse
import os
import sys
import hashlib
import json
import re

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def execution_lease_path(model_id, results_dir, registered_ids):
    """Validate identity before deriving the single-model lease filename."""
    if (not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", model_id)
            or model_id not in registered_ids):
        raise ValueError(f"Unknown or unsafe registered model ID: {model_id!r}")
    return Path(results_dir) / "modeling" / "execution_leases" / f"{model_id}.lock"


def run_with_execution_lease(model_id, results_dir, threads, *, registered_ids, runner):
    """Serialize this model across CLI processes, then delegate full resume checks.

    Different registered IDs use different lock files. No completion/metadata
    decision is made before acquiring the lock: a waiter still calls the original
    runner, which validates or skips any now-completed fitting work itself.
    """
    lock_path = execution_lease_path(model_id, results_dir, registered_ids)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        return runner(model_id, results_dir, threads)


def lock_and_report(results_dir):
    """Create a first global lock, or verify/reuse its exact bytes on resume."""
    from nephrotox.data import load_data
    from nephrotox.modeling_common import registry
    from nephrotox.modeling_pipeline import lock_all
    from nephrotox.modeling_reporting import load_locked_predictions, report_predictions
    results = Path(results_dir)
    modeling = results / "modeling"
    if not (modeling / "model_lock.json").is_file():
        lock_all(results)
    rows, _ = load_data()
    split_path = results / "split_payload.json"
    assignments = json.loads(split_path.read_text(encoding="utf-8"))["assignments"]
    load_locked_predictions(modeling, rows, assignments, expected_model_ids=[job["model_id"] for job in registry()])
    protocol = json.loads((modeling / "protocol.json").read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(split_path) != protocol.get("split_payload_sha256"):
        raise ValueError("Current assignments differ from the frozen modeling split")
    fitting = protocol.get("fitting_source_sha256", {})
    if not fitting or any(Path(name).name != name or sha(ROOT / "src/nephrotox" / name) != expected
                          for name, expected in fitting.items()):
        raise ValueError("Current fitting source differs from the frozen modeling protocol")
    if protocol.get("parameter_mode") == "fixed_original":
        if protocol.get("fixed_parameter_file") != "configs/original_fixed_parameters.json" or sha(ROOT / protocol["fixed_parameter_file"]) != protocol.get("fixed_parameter_sha256"):
            raise ValueError("Original fixed-parameter source differs from the frozen protocol")
    return report_predictions(modeling, rows, assignments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--lock-and-report", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument('--parameter-mode',choices=['fixed_original'], default='fixed_original')
    parser.add_argument('--split-mode', choices=['legacy_random'], default='legacy_random', help='Must match an existing prepared run')
    args = parser.parse_args()
    if args.split_mode is not None:
        manifest = json.loads((args.results / 'run_manifest.json').read_text(encoding='utf-8'))
        csv_manifest = json.loads((ROOT / 'data/source_manifest.json').read_bytes())
        if manifest['source'].get('sha256') != csv_manifest['sha256'] or manifest['source'].get('source_format') != 'csv':
            raise ValueError("New fits require features and partitions prepared from the current release CSV")
        if manifest.get('split_mode', 'structure_group') != args.split_mode:
            raise ValueError('Requested split mode differs from prepared results; do not mix runs')
    # Bound BLAS/OpenMP before importing numerical libraries in fresh processes.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from nephrotox.modeling_common import training_context, prepare_count_features, freeze_protocol, registry
    from nephrotox.modeling_graph import GRAPH_CANDIDATES
    from nephrotox.modeling_pipeline import run_job
    from nephrotox.fixed_parameters import configure_parameter_mode
    registered_ids = {job["model_id"] for job in registry()}
    jobs = [j["model_id"] for j in registry()] if args.all else (args.model_id or [])
    # Reject every unknown/path-traversal ID before changing run configuration
    # or creating any lease/output path.
    for model_id in jobs:
        execution_lease_path(model_id, args.results, registered_ids)
    if args.parameter_mode is not None:
        configure_parameter_mode(args.results,args.parameter_mode)
    if args.prepare:
        context = training_context(args.results)
        prepare_count_features(context)
        freeze_protocol(context, GRAPH_CANDIDATES)
        print("Count features and immutable modeling protocol prepared", flush=True)
    for model_id in jobs:
        run_with_execution_lease(model_id, args.results, args.threads,
                                 registered_ids=registered_ids, runner=run_job)
    if args.lock_and_report:
        lock_and_report(args.results)
    if not any((args.prepare, jobs, args.lock_and_report)):
        parser.print_help()


if __name__ == "__main__":
    main()
