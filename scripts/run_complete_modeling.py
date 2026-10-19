"""Resume the registered models, then generate every locked report and figure.

No scheduler, cloud upload or external messages are used. Worker subprocesses
remain bounded and save resumable results in results/modeling. A failure stops
publication and is retained in workflow_status.json; it is never substituted
with predictions from a historical run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def command(arguments, stream=None):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True,
                   stdout=stream, stderr=subprocess.STDOUT, env={**os.environ,"PYTHONUTF8":"1"})


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
    explain_models(rows, assignments, results / "modeling")
    command(["scripts/benchmark_models.py", "--results", results / "modeling", "--threads", threads])
    command(["scripts/draw_figure1.py"])
    command(["scripts/draw_figure2.py"])
    command(["scripts/draw_model_figures.py", "--results", results / "modeling", "--supplementary"])
    command(["scripts/draw_reliability_figures.py", "--results", results / "modeling"])
    command(["scripts/draw_explain_figures.py", "--results", results / "modeling"])
    command(["scripts/draw_training_diagnostics.py", "--results", results / "modeling"])
    command(["scripts/export_modeling_materials.py", "--results", results / "modeling"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--exclude-model", action="append", default=[], help="An already-running job; it must finish before postprocessing")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    args.results = args.results.resolve()
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from nephrotox.modeling_common import registry, json_write, utc_now
    out = args.results / "modeling"
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "execution_logs"
    logdir.mkdir(exist_ok=True)
    status_path = out / "workflow_status.json"
    status = {"state":"training", "started_at_utc":utc_now(), "worker_limit":args.workers,
              "threads_per_worker":args.threads, "pid":os.getpid(), "complete":[], "failed":[]}
    try:
        if not args.postprocess_only:
            if not (args.results / "run_manifest.json").exists():
                command(["reproduce.py", "--output", args.results])
            command(["scripts/run_modeling.py", "--results", args.results, "--prepare", "--threads", args.threads])
            jobs = [j for j in registry() if j["model_id"] not in args.exclude_model]
            # One graph starts early; remaining jobs keep the complete registry order.
            jobs.sort(key=lambda j: (0 if j["model_id"] == "graph__DMPNN" else 1))
            def train(job):
                model_id = job["model_id"]
                with (logdir / f"{model_id}.txt").open("a", encoding="utf-8") as stream:
                    command(["scripts/run_modeling.py","--results",args.results,"--model-id",model_id,"--threads",args.threads],stream)
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
