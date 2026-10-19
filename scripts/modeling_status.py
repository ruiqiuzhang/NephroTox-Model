"""Show saved training progress without reading any holdout labels."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/modeling")
    args = parser.parse_args()
    if not (args.results / "protocol.json").exists():
        print(json.dumps({"state":"not_started","command":"python scripts/run_complete_modeling.py"}))
        return
    protocol = json.loads((args.results / "protocol.json").read_text(encoding="utf-8"))
    finished, active, failed = [], [], []
    outer = trials = 0
    for job in protocol["registry"]:
        path = args.results / job["model_id"]
        completed_folds = len(list(path.glob("outer_*/complete.json")))
        n_trials = len(list(path.glob("**/trial.json")))
        outer += completed_folds
        trials += n_trials
        status = json.loads((path/"status.json").read_text(encoding="utf-8")) if (path/"status.json").exists() else {"state":"pending"}
        item = {"model_id":job["model_id"], "state":status["state"], "outer_folds":completed_folds, "inner_fits":n_trials}
        if status["state"] == "complete":
            finished.append(job["model_id"])
        elif status["state"] == "failed":
            failed.append({**item,"message":status.get("message")})
        elif status["state"] == "running":
            active.append(item)
    print(json.dumps({"registered_models":len(protocol["registry"]), "complete_models":len(finished),
        "outer_folds_complete":outer,"inner_fits_complete":trials,"finished":finished,"active":active,"failed":failed},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
