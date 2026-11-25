"""Train the registered models and generate the locked numerical reports."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def command(arguments, stream=None):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True,
                   stdout=stream, stderr=subprocess.STDOUT, env={**os.environ,"PYTHONUTF8":"1"})


def resolve_split_mode(results, requested=None):
    modes = {'structure_group', 'legacy_random'}
    if requested is not None and requested not in modes:
        raise ValueError('Unknown split mode')
    path = Path(results) / 'run_manifest.json'
    if not path.exists():
        return requested or 'structure_group'
    stored = json.loads(path.read_text(encoding='utf-8')).get('split_mode', 'structure_group')
    if stored not in modes or (requested is not None and stored != requested):
        raise ValueError('Requested split mode differs from prepared results; use a separate results directory')
    return stored


def postprocess(results, threads=2):
    from nephrotox.data import load_data
    from nephrotox.modeling_common import training_context, raw_matrix
    from nephrotox.modeling_reliability import report_features
    from nephrotox.modeling_explain import explain_models
    command(["scripts/run_modeling.py", "--results", results, "--lock-and-report"])
    rows, _ = load_data()
    assignments = json.loads((results / "split_payload.json").read_text(encoding="utf-8"))["assignments"]
    context = training_context(results)
    matrices = {name: raw_matrix(name, context) for name in ["morgan2048", "maccs166"]}
    report_features(rows, assignments, matrices, results / "modeling")
    command(["scripts/prepare_structural_alerts.py", "--results", results])
    explain_models(rows, assignments, results / "modeling")
    command(["scripts/benchmark_models.py", "--results", results / "modeling", "--threads", threads])
    if context.get("parameter_mode") == "fixed_original":
        # The matched comparison requires inherited, identical fixed parameters;
        # an independently tuned binary/count pair is a different experiment.
        command(["scripts/export_binary_count_tables.py", "--results", results / "modeling"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--exclude-model", action="append", default=[], help="An already-running job; it must finish before postprocessing")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument('--parameter-mode',choices=['fixed_original'], default='fixed_original')
    parser.add_argument('--split-mode', choices=['legacy_random'], default='legacy_random')
    args = parser.parse_args()
    args.results = args.results.resolve()
    args.split_mode = resolve_split_mode(args.results, args.split_mode)
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from nephrotox.modeling_common import registry, json_write, utc_now
    from nephrotox.fixed_parameters import configure_parameter_mode
    if args.parameter_mode is not None:
        configure_parameter_mode(args.results,args.parameter_mode)
    out = args.results / "modeling"
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "execution_logs"
    logdir.mkdir(exist_ok=True)
    status_path = out / "workflow_status.json"
    status = {"state":"training", "started_at_utc":utc_now(), "worker_limit":args.workers,
              "threads_per_worker":args.threads, "pid":os.getpid(), "split_mode":args.split_mode,
              "complete":[], "failed":[]}
    try:
        if not args.postprocess_only:
            if not (args.results / "run_manifest.json").exists():
                command(["reproduce.py", "--output", args.results, "--split-mode", args.split_mode])
            command(["scripts/run_modeling.py", "--results", args.results, "--prepare", "--threads", args.threads,
                     "--split-mode", args.split_mode])
            jobs = [j for j in registry() if j["model_id"] not in args.exclude_model]
            # One graph starts early; remaining jobs keep the complete registry order.
            jobs.sort(key=lambda j: (0 if j["model_id"] == "graph__DMPNN" else 1))
            graph_slot=threading.Semaphore(1)
            def train(job):
                model_id = job["model_id"]
                def fit_one():
                    with (logdir / f"{model_id}.txt").open("a", encoding="utf-8") as stream:
                        command(["scripts/run_modeling.py","--results",args.results,"--model-id",model_id,"--threads",args.threads,
                                 "--split-mode",args.split_mode],stream)
                if job['representation']=='graph':
                    with graph_slot:
                        fit_one()
                else:
                    fit_one()
                return model_id
            json_write(status_path,status)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(train,j): j["model_id"] for j in jobs}
                for future in as_completed(futures):
                    try:
                        model_id = future.result()
                        status["complete"].append(model_id)
                        print(f"COMPLETE {len(status['complete'])}/{len(jobs)}: {model_id}",flush=True)
                    except Exception as error:
                        status["failed"].append({"model_id":futures[future],"error":str(error)})
                    status["updated_at_utc"] = utc_now()
                    json_write(status_path,status)
            if status["failed"]:
                raise RuntimeError("One or more models failed; see execution logs and per-model status")
        if not args.train_only:
            status["state"]="postprocessing"
            json_write(status_path,status)
            postprocess(args.results, args.threads)
        status["state"]="complete" if not args.train_only else "training_complete"
        status["completed_at_utc"]=utc_now()
        json_write(status_path,status)
    except Exception:
        status["state"]="failed"
        status["traceback"]=traceback.format_exc()
        json_write(status_path,status)
        raise


if __name__ == "__main__":
    main()
