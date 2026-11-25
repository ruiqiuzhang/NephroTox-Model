"""Summarize locked model performance and compare 15 prespecified fingerprint pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.input_provenance import source_file_fields
from nephrotox.data import load_data
from nephrotox.modeling_reporting import (METRICS, load_locked_predictions, metric_values,
                                         paired_auc_comparison)

SEED = 20260810
N_BOOTSTRAP = 2000
ALGORITHMS = ("rf", "svm", "mlp", "xgb", "dnn")
FAMILIES = (("Morgan", "morgan2048", "morgan_count2048"),
            ("AtomPair", "atompair1024", "atompair_count1024"),
            ("TopologicalTorsion", "torsion1024", "torsion_count1024"))
MAIN_REPRESENTATIONS = ("morgan2048", "maccs166", "atompair1024", "torsion1024", "rdkit166")
MODEL_IDS = tuple(rep + "__" + alg for rep in (*MAIN_REPRESENTATIONS, *(p[2] for p in FAMILIES)) for alg in ALGORITHMS) + (
    "graph__DMPNN", "graph__GINE", "graph__AttentiveFP")
# Pair direction is count minus binary. Never rank/select pairs from outcomes.
PAIRS = tuple((family, algorithm, binary + "__" + algorithm, count + "__" + algorithm)
              for family, binary, count in FAMILIES for algorithm in ALGORITHMS)
PARTITIONS = (("development_oof", "development", "development_index"), ("test", "test", "test_index"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        raise ValueError(f"Cannot export an empty report: {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def comparison_plan():
    return {"parameter_mode": "fixed_original", "direction": "count minus binary",
        "families": [p[0] for p in FAMILIES], "algorithms": list(ALGORITHMS),
        "pairs": [{"family": f, "algorithm": a.upper(), "binary_model_id": b, "count_model_id": c} for f, a, b, c in PAIRS],
        "partitions": [p[0] for p in PARTITIONS], "comparisons_per_partition": 15,
        "n_bootstrap": N_BOOTSTRAP, "seed": SEED, "confidence": 0.95,
        "resampling": "shared class-stratified compound resamples within each partition; paired saved probabilities; no refitting",
        "p_value": "two-sided centered paired bootstrap with plus-one correction",
        "multiplicity": "Holm correction across the 15 predefined pairs separately within development_oof and test",
        "parameter_rule": "Each count model inherits the complete fixed hyperparameter configuration of its binary-family/algorithm model; representation-specific preprocessing remains explicit.",
        "test_outcomes_used_to_choose_pairs": False,
        "auc_scope": "Development ROC-AUC is pooled OOF AUC; ten-outer-fold mean and sample SD are separate fields.",
        "uncertainty_scope": "Conditional on the saved fitted predictions; excludes retraining and historical parameter-selection uncertainty."}


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid report value: {key}") from error
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite report value: {key}")
    return value


def _same(actual, expected, message):
    if not np.isclose(actual, expected, atol=1e-12, rtol=0):
        raise ValueError(message)


def validate_fixed_models(bundle, protocol):
    if protocol.get("parameter_mode") != "fixed_original":
        raise ValueError("Matched binary/count comparisons require fixed_original parameters")
    if set(bundle["metadata"]) != set(MODEL_IDS) or set(bundle["lock"]["expected_model_ids"]) != set(MODEL_IDS):
        raise ValueError("Complete current 43-model lock is required")
    jobs = protocol.get("registry", [])
    if len(jobs) != 43 or {job["model_id"] for job in jobs} != set(MODEL_IDS):
        raise ValueError("Protocol model registry differs from the fixed 43-model comparison")
    definitions = protocol.get("fixed_parameter_records", {})
    if set(definitions) != set(MODEL_IDS):
        raise ValueError("Fixed parameter provenance is incomplete")
    for model_id, meta in bundle["metadata"].items():
        params = definitions[model_id].get("params")
        if params != meta.get("params") or len(meta.get("outer_params", [])) != 10 or any(p != params for p in meta["outer_params"]):
            raise ValueError(f"Locked parameters differ across fixed fitting scopes: {model_id}")
    for _, _, binary, count in PAIRS:
        reference = definitions[count].get("count_inheritance") or definitions[count].get("inherits_parameters_from")
        if reference != binary or bundle["metadata"][binary]["params"] != bundle["metadata"][count]["params"]:
            raise ValueError(f"Binary/count fixed-parameter inheritance mismatch: {count}")


def validate_prediction_records(records, bundle, rows, partition, probability_key, index_key):
    """Reconcile every report ID, label and probability to the locked arrays."""
    index = bundle[index_key]
    expected_ids = [rows[i]["compound_id"] for i in index]
    if len(records) != len(MODEL_IDS) * len(index):
        raise ValueError(f"Incomplete prediction report: {partition}")
    by_model = {name: [] for name in MODEL_IDS}
    for record in records:
        if record.get("model_id") not in by_model:
            raise ValueError("Prediction report contains an unregistered model")
        by_model[record["model_id"]].append(record)
    for model_id, saved in by_model.items():
        if [r["compound_id"] for r in saved] != expected_ids:
            raise ValueError(f"Prediction report compound IDs/order differ: {model_id}/{partition}")
        for position, (record, global_index) in enumerate(zip(saved, index)):
            if _number(record, "label") != rows[global_index]["label"]:
                raise ValueError("Prediction report label differs from current DATA")
            probability = float(bundle["predictions"][model_id][probability_key][position])
            _same(_number(record, "probability"), probability, "Prediction report differs from locked probabilities")
            _same(_number(record, "threshold"), 0.5, "Prediction threshold differs from the fixed rule")
            if _number(record, "predicted_label") != int(probability >= 0.5):
                raise ValueError("Reported predicted label differs from the saved probability")
            assignment = bundle["assignments"][global_index]
            if record.get("structure_group_id") != assignment["structure_group_id"]:
                raise ValueError("Prediction report structure group differs")
            reported_fold = record.get("cv_validation_fold")
            fold = None if reported_fold in (None, "") else int(reported_fold)
            if fold != assignment["cv_validation_fold"]:
                raise ValueError("Prediction report validation fold differs")


def build_tables(bundle, rows, protocol, summaries, fold_records, *, n_bootstrap=N_BOOTSTRAP):
    """Assemble source tables from an already hash-verified prediction bundle."""
    validate_fixed_models(bundle, protocol)
    labels = np.asarray([r["label"] for r in rows], dtype=int)
    lookup, fold_lookup = {}, {}
    for row in summaries:
        key = (row["model_id"], row["partition"])
        if key in lookup:
            raise ValueError("Duplicate model/partition performance row")
        lookup[key] = row
    if set(lookup) != {(name, partition) for name in MODEL_IDS for partition, _, _ in PARTITIONS}:
        raise ValueError("Performance source must contain all 43 models in both partitions")
    for row in fold_records:
        key = (row["model_id"], int(row["fold"]))
        if key in fold_lookup:
            raise ValueError("Duplicate per-fold performance row")
        fold_lookup[key] = row
    if set(fold_lookup) != {(name, fold) for name in MODEL_IDS for fold in range(1, 11)}:
        raise ValueError("Ten current outer-fold records are required for every model")
    performance = []
    for model_id in MODEL_IDS:
        meta = bundle["metadata"][model_id]
        fold_auc = []
        for fold in range(1, 11):
            mask = bundle["folds"] == fold
            y = labels[bundle["development_index"]][mask]
            p = bundle["predictions"][model_id]["development"][mask]
            values = metric_values(y, p)
            report = fold_lookup[(model_id, fold)]
            _same(_number(report, "n"), len(y), "Reported fold size differs")
            for key, value in values.items():
                if value is None:
                    raise ValueError("A declared outer fold lacks both classes")
                _same(_number(report, key), value, "Per-fold metrics differ from saved OOF probabilities")
            fold_auc.append(values["ROC_AUC"])
        mean, sd = float(np.mean(fold_auc)), float(np.std(fold_auc, ddof=1))
        for partition, probability_key, index_key in PARTITIONS:
            index = bundle[index_key]
            values = metric_values(labels[index], bundle["predictions"][model_id][probability_key])
            report = lookup[(model_id, partition)]
            for key, value in values.items():
                _same(_number(report, key), value, "Performance summary differs from locked probabilities")
            for key, value in (("n", len(index)), ("positive", int(labels[index].sum())),
                               ("reference_negative", int(len(index) - labels[index].sum())), ("threshold", 0.5),
                               ("development_fold_mean_ROC_AUC", mean), ("development_fold_sd_ROC_AUC", sd)):
                _same(_number(report, key), value, f"Performance population/fold summary differs: {key}")
            record = {"model_id": model_id, "algorithm": meta["algorithm"], "representation": meta["representation"],
                "role": meta["role"], "partition": partition, "n": len(index), "positive": int(labels[index].sum()),
                "reference_negative": int(len(index) - labels[index].sum()), "threshold": 0.5,
                "ROC_AUC_estimate_scope": "pooled out-of-fold probabilities" if partition == "development_oof" else "final all-development fit on internal held-out test",
                **values}
            for metric in METRICS:
                lower, upper = _number(report, metric + "_lower"), _number(report, metric + "_upper")
                if lower > upper:
                    raise ValueError("Performance confidence interval endpoints are reversed")
                record[metric + "_lower"], record[metric + "_upper"] = lower, upper
            record.update(development_fold_mean_ROC_AUC=mean, development_fold_sd_ROC_AUC=sd,
                development_fold_summary_scope="unweighted mean and sample SD of ten outer-fold AUCs; distinct from pooled OOF AUC",
                parameter_mode="fixed_original", parameters_json=json.dumps(meta["params"], sort_keys=True))
            performance.append(record)
    performance_lookup = {(r["model_id"], r["partition"]): r for r in performance}
    comparisons = []
    for partition, probability_key, index_key in PARTITIONS:
        probabilities = {model: bundle["predictions"][model][probability_key]
                         for _, _, binary, count in PAIRS for model in (binary, count)}
        paired = paired_auc_comparison(labels[bundle[index_key]], probabilities,
            pairs=[(count, binary) for _, _, binary, count in PAIRS], n_bootstrap=n_bootstrap, seed=SEED)
        for (family, algorithm, binary, count), result in zip(PAIRS, paired):
            b, c = performance_lookup[(binary, partition)], performance_lookup[(count, partition)]
            if result["model_1"] != count or result["model_2"] != binary or result["comparison_family_size"] != 15:
                raise ValueError("Paired comparisons differ from the predefined 15-pair family")
            record = {"family": family, "algorithm": algorithm.upper(), "partition": partition,
                "binary_model_id": binary, "count_model_id": count, "n": b["n"], "positive": b["positive"],
                "reference_negative": b["reference_negative"], "threshold": 0.5, "difference_direction": "count minus binary",
                "ROC_AUC_estimate_scope": b["ROC_AUC_estimate_scope"],
                "binary_ROC_AUC": b["ROC_AUC"], "binary_ROC_AUC_lower": b["ROC_AUC_lower"], "binary_ROC_AUC_upper": b["ROC_AUC_upper"],
                "count_ROC_AUC": c["ROC_AUC"], "count_ROC_AUC_lower": c["ROC_AUC_lower"], "count_ROC_AUC_upper": c["ROC_AUC_upper"],
                "delta_ROC_AUC": result["difference"], "delta_ROC_AUC_lower": result["lower"], "delta_ROC_AUC_upper": result["upper"],
                "paired_p_bootstrap": result["p_bootstrap"], "paired_p_holm": result["p_holm"],
                "holm_family": "binary_count_15_" + partition, "holm_family_size": 15,
                "significant_holm_005": result["significant_holm_005"], "n_bootstrap": n_bootstrap, "seed": SEED,
                "paired_method": result["method"], "fixed_parameters_identical": True,
                "binary_development_fold_mean_ROC_AUC": b["development_fold_mean_ROC_AUC"],
                "binary_development_fold_sd_ROC_AUC": b["development_fold_sd_ROC_AUC"],
                "count_development_fold_mean_ROC_AUC": c["development_fold_mean_ROC_AUC"],
                "count_development_fold_sd_ROC_AUC": c["development_fold_sd_ROC_AUC"],
                "fixed_parameters_json": c["parameters_json"]}
            for metric in ("BA", "MCC", "AP", "Brier"):
                record["binary_" + metric], record["count_" + metric] = b[metric], c[metric]
                record["delta_" + metric] = c[metric] - b[metric]
            comparisons.append(record)
    return {"performance": performance, "binary_count_comparisons": comparisons}


def export_tables(results, output=None):
    results = Path(results)
    # Fail before DATA/report loading when no complete global lock exists.
    if not (results / "model_lock.json").is_file():
        raise ValueError("A complete current global model lock is required for fingerprint comparisons")
    rows, info = load_data()
    assignments_path = results.parent / "split_payload.json"
    assignments = read_json(assignments_path)["assignments"]
    bundle = load_locked_predictions(results, rows, assignments, expected_model_ids=list(MODEL_IDS))
    protocol = read_json(results / "protocol.json")
    validate_fixed_models(bundle, protocol)
    report_folder = results / "reports"
    manifest = read_json(report_folder / "report_manifest.json")
    required = {"model_summary.csv", "perfold_metrics.csv", "oof_predictions.csv", "test_predictions.csv"}
    if (manifest.get("status") != "complete" or manifest.get("global_lock_sha256") != digest(results / "model_lock.json")
            or manifest.get("protocol_sha256") != bundle["lock"]["protocol_sha256"]
            or manifest.get("source_rows_fingerprint") != info["rows_fingerprint"]
            or source_file_fields(manifest) != source_file_fields(bundle["lock"])
            or manifest.get("n_bootstrap") != N_BOOTSTRAP or manifest.get("bootstrap_seed") != SEED
            or manifest.get("confidence") != 0.95 or set(manifest.get("model_ids", [])) != set(MODEL_IDS)
            or not required.issubset(manifest.get("files", {}))):
        raise ValueError("Current complete reporting manifest or registered CI procedure is missing/inconsistent")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(report_folder / name) != expected:
            raise ValueError(f"A report source changed after locking: {name}")
    for partition, probability_key, index_key in PARTITIONS:
        filename = "oof_predictions.csv" if partition == "development_oof" else "test_predictions.csv"
        validate_prediction_records(read_csv(report_folder / filename), bundle, rows, partition, probability_key, index_key)
    tables = build_tables(bundle, rows, protocol, read_csv(report_folder / "model_summary.csv"),
                          read_csv(report_folder / "perfold_metrics.csv"))
    identity = [{"compound_id": row["compound_id"], "name": row["name"], "PubChem_CID": row["PubChem_CID"],
        "standardized_smiles": row["standardized_smiles"], "label": row["label"],
        "partition": a["dataset_partition"], "cv_validation_fold": a["cv_validation_fold"],
        "structure_group_id": a["structure_group_id"]} for row, a in zip(rows, assignments)]
    source = {**source_file_fields(bundle["lock"]), "source_rows_fingerprint": info["rows_fingerprint"],
        "split_mode": bundle.get("split_mode", "structure_group"),
        "structural_overlap": bundle.get("structural_overlap"),
        "global_lock_sha256": digest(results / "model_lock.json"), "protocol_sha256": bundle["lock"]["protocol_sha256"],
        "split_payload_sha256": digest(assignments_path), "report_manifest_sha256": digest(report_folder / "report_manifest.json"),
        "report_input_hashes": {name: manifest["files"][name] for name in sorted(required)},
        "fixed_parameter_file": protocol.get("fixed_parameter_file"), "fixed_parameter_sha256": protocol.get("fixed_parameter_sha256"),
        "prediction_and_model_hashes": bundle["lock"]["models"]}
    payload = {"schema_version": 1, "status": "complete", "comparison_plan": comparison_plan(), "source": source,
        "source_identity": identity, "fixed_parameter_provenance": protocol["fixed_parameter_records"], **tables,
        "notes": ["All 43 models have development OOF and internal held-out performance rows (86 rows).",
            "The 15 binary/count pairs are fixed before examining test performance; 30 comparisons cover both partitions.",
            "AUC difference CIs use paired probability resampling, not subtraction of separate model confidence intervals.",
            "BA/MCC/AP/Brier differences are point estimates; only ROC-AUC differences receive paired CIs and tests here.",
            "Lower Brier is better; for AUC/BA/MCC/AP, higher is better.",
            "Historical fixed parameters arose from an overlapping compound collection; this is an internal re-evaluation, not prospective external validation."]}
    out = Path(output) if output else results / "fingerprint_comparisons"
    out.mkdir(parents=True, exist_ok=True)
    source_path = out / "comparison_source.json"
    performance_path = out / "all_model_performance.csv"
    comparisons_path = out / "paired_binary_count.csv"
    source_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(performance_path, tables["performance"])
    write_csv(comparisons_path, tables["binary_count_comparisons"])
    output_manifest = {"status": "complete", "source": source, "n_models": 43, "performance_rows": 86,
        "comparison_rows": 30, "files": {path.name: digest(path) for path in (source_path, performance_path, comparisons_path)}}
    (out / "report_manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/modeling", help="Complete globally locked modeling directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = export_tables(args.results, args.output)
    print(json.dumps({"status": payload["status"], "performance_rows": len(payload["performance"]),
                      "comparison_rows": len(payload["binary_count_comparisons"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
