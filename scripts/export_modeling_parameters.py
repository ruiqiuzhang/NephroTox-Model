"""Export every prespecified candidate and graph feature before model fitting."""
import csv
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.modeling_common import training_context, build_protocol, json_write, SEEDS
from nephrotox.data import load_data
from nephrotox.modeling_graph import GRAPH_CANDIDATES, graph_feature_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,help="Optional current-DATA feature/split directory; not model results")
    args = parser.parse_args()
    if args.results is not None:
        context = training_context(args.results)
    else:
        _, info = load_data()
        published = json.loads((ROOT / "configs/modeling_protocol.json").read_text(encoding="utf-8"))
        if published["source_excel_sha256"] != info["sha256"]:
            raise ValueError("Published parameter protocol refers to a different DATA workbook")
        context = {"source": info, "assignment_sha256": published["split_payload_sha256"]}
    protocol = build_protocol(context, GRAPH_CANDIDATES)
    json_write(ROOT / "configs/modeling_protocol.json", protocol)
    json_write(ROOT / "configs/graph_feature_schema.json", graph_feature_schema())
    jobs = protocol["registry"]
    with (ROOT / "configs/model_registry.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    candidates=[]
    for job in jobs:
        options = protocol["graph_candidate_parameters"][job["estimator_key"]] if job["representation"]=="graph" else protocol["candidate_parameters"][job["estimator_key"]]
        for number,params in enumerate(options,1):
            candidates.append({**job,"candidate":number,"parameters_json":json.dumps(params,sort_keys=True),
                "candidate_id_in_code":number-1, "inner_folds":3,
                "search_seed":SEEDS[0], "selection_metric":"mean_inner_ROC_AUC"})
    with (ROOT / "configs/model_candidate_parameters.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Exported {len(jobs)} models and {len(candidates)} searchable candidate entries")


if __name__ == "__main__":
    main()
