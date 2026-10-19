"""Assemble supplement-ready tables only after successful locked evaluation.

All tables are UTF-8 CSV; parameters with variable architecture are encoded as
JSON in named columns. Original and derived data remain distinguishable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from nephrotox.data import load_data
from nephrotox.modeling_common import json_write,sha256,utc_now
from nephrotox.modeling_reporting import load_locked_predictions


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot export empty {path.name}")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/modeling")
    args=parser.parse_args()
    rows,info=load_data()
    assignments=json.loads((args.results.parent/"split_payload.json").read_text(encoding="utf-8"))["assignments"]
    bundle=load_locked_predictions(args.results,rows,assignments)
    report=args.results/"reports"
    if not (report/"report_manifest.json").exists():
        raise ValueError("Locked statistical report must be completed before material export")
    out=args.results/"materials"
    out.mkdir(exist_ok=True)
    write_csv(out/"TableS1_Current_DATA_and_partition.csv",[{**row,"dataset_partition":a["dataset_partition"],
        "cv_validation_fold":a["cv_validation_fold"],"structure_group_id":a["structure_group_id"]}
        for row,a in zip(rows,assignments)])
    protocol=json.loads((args.results/"protocol.json").read_text(encoding="utf-8"))
    parameter_rows=[]
    trial_rows=[]
    epoch_rows=[]
    for job in protocol["registry"]:
        model_id=job["model_id"]
        folder=args.results/model_id
        metadata=bundle["metadata"][model_id]
        for i,params in enumerate(metadata["outer_params"],1):
            parameter_rows.append({"model_id":model_id,"scope":f"outer_fold_{i}","parameters_json":json.dumps(params,sort_keys=True)})
        parameter_rows.append({"model_id":model_id,"scope":"full_development_final","parameters_json":json.dumps(metadata["params"],sort_keys=True)})
        for path in sorted(folder.glob("**/trial.json")):
            trial=json.loads(path.read_text(encoding="utf-8"))
            trial_rows.append({"model_id":model_id,"scope":path.relative_to(folder).parts[0],
                "candidate":trial["candidate"],"inner_fold":trial["inner_fold"],"ROC_AUC":trial["roc_auc"],
                "elapsed_seconds":trial["elapsed_seconds"],"parameters_json":json.dumps(trial["params"],sort_keys=True),
                "training_n":len(trial["training_ids"]),"validation_n":len(trial["validation_ids"]),
                "source_file":path.relative_to(args.results).as_posix(),"source_sha256":sha256(path)})
        for path in sorted(folder.glob("**/fit_metadata.json")):
            fit=json.loads(path.read_text(encoding="utf-8"))
            for iteration,loss in enumerate(fit.get("loss_curve",[]),1):
                epoch_rows.append({"model_id":model_id,"fit_scope":path.parent.relative_to(folder).as_posix(),
                    "seed":fit.get("seed"),"phase":"mlp_training","epoch":iteration,"training_loss":loss})
            # Preserve the complete native log separately, and flatten neural
            # histories without manufacturing unavailable validation curves.
            members=fit.get("seeds_detail",fit.get("members",[]))
            for member in members:
                for key,value in member.items():
                    if isinstance(value,list) and value and isinstance(value[0],dict) and "epoch" in value[0]:
                        for epoch in value:
                            epoch_rows.append({"model_id":model_id,"fit_scope":path.parent.relative_to(folder).as_posix(),
                                "seed":member.get("seed"),"phase":key,**epoch})
    write_csv(out/"TableS2_Selected_parameters.csv",parameter_rows)
    write_csv(out/"TableS3_All_search_trials.csv",trial_rows)
    if epoch_rows:
        write_csv(out/"TableS4_Neural_learning_curves.csv",epoch_rows)
    # Report exports include the exact probabilities, confusion counts and CI.
    for source in sorted(report.glob("*.csv")):
        shutil.copy2(source,out/source.name)
    for name in ["reliability","explain","benchmarks","figures/training_diagnostics"]:
        folder=args.results/name
        if folder.exists():
            target=out/name
            target.mkdir(parents=True,exist_ok=True)
            for source in folder.glob("*.csv"):
                shutil.copy2(source,target/source.name)
    json_write(out/"material_manifest.json",{"status":"complete","generated_at_utc":utc_now(),
        "source_excel_sha256":info["sha256"],"source_rows_fingerprint":info["rows_fingerprint"],
        "global_lock_sha256":sha256(args.results/"model_lock.json"),"models":len(bundle["metadata"]),
        "source_records":len(rows),"positive":sum(r["label"] for r in rows),
        "reference_negative":sum(r["label"]==0 for r in rows),"parameter_rows":len(parameter_rows),
        "tuning_trial_rows":len(trial_rows),"files":{p.relative_to(out).as_posix():sha256(p) for p in out.rglob("*.csv")}})
    print(out)


if __name__=="__main__":
    main()
