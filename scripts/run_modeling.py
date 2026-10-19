"""Full DATA-only modeling: prepare, resume jobs, freeze, then evaluate."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--lock-and-report", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    # Bound BLAS/OpenMP before importing numerical libraries in fresh processes.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from nephrotox.modeling_common import training_context, prepare_count_features, freeze_protocol, registry
    from nephrotox.modeling_graph import GRAPH_CANDIDATES
    from nephrotox.modeling_pipeline import run_job, lock_all
    if args.prepare:
        context = training_context(args.results)
        prepare_count_features(context)
        freeze_protocol(context, GRAPH_CANDIDATES)
        print("Count features and immutable modeling protocol prepared", flush=True)
    jobs = [j["model_id"] for j in registry()] if args.all else (args.model_id or [])
    for model_id in jobs:
        run_job(model_id, args.results, args.threads)
    if args.lock_and_report:
        lock_all(args.results)
        from nephrotox.data import load_data
        from nephrotox.modeling_reporting import report_predictions
        import json
        rows, _ = load_data()
        assignments = json.loads((args.results / "split_payload.json").read_text(encoding="utf-8"))["assignments"]
        report_predictions(args.results / "modeling", rows, assignments)
    if not any((args.prepare, jobs, args.lock_and_report)):
        parser.print_help()


if __name__ == "__main__":
    main()
