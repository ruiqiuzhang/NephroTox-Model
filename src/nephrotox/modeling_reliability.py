"""Post-lock applicability domain, calibration and structural-association reports.

Every threshold is defined without validation/test outcomes. Test labels are
read only after the global model lock has been verified by the shared loader.
Structural-key enrichment is restricted to development labels and is an
exploratory association analysis, not evidence for toxicity mechanisms.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit.Chem import MACCSkeys
from scipy.stats import fisher_exact, rankdata
from threadpoolctl import threadpool_limits

from .modeling_reporting import calibration_bins, load_locked_predictions, metric_values


def write_csv(path, records, fieldnames=None):
    records = list(records)
    columns = list(fieldnames or dict.fromkeys(k for row in records for k in row))
    if not columns:
        columns = ["status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def tanimoto_matrix(binary_matrix):
    """Dense binary Tanimoto; empty/empty is zero, matching RDKit bit vectors."""
    x = np.asarray(binary_matrix)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError("Tanimoto requires a two-dimensional binary fingerprint matrix")
    x = np.asarray(x, dtype=np.float32)
    with threadpool_limits(limits=2):
        overlap = x @ x.T
    total = x.sum(axis=1)
    union = total[:, None] + total[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union != 0)


def _neighbors(similarity, query, reference, exclude_self=False):
    query, reference = np.asarray(query, int), np.asarray(reference, int)
    if not len(query) or not len(reference):
        raise ValueError("Neighbor scoring requires nonempty query and reference subsets")
    block = similarity[np.ix_(query, reference)].copy()
    if exclude_self:
        block[query[:, None] == reference[None, :]] = -np.inf
    available = len(reference) - int(exclude_self)
    if available < 1:
        raise ValueError("Leave-one-out AD needs at least two reference compounds")
    k = min(5, available)
    nearest = np.argmax(block, axis=1)
    highest = np.partition(block, block.shape[1] - k, axis=1)[:, -k:]
    return block[np.arange(len(query)), nearest], highest.mean(axis=1), reference[nearest]


def applicability_domain(rows, assignments, binary_matrix):
    """Predefined fifth-percentile AD, including honest fold-specific OOF scores."""
    if isinstance(assignments, dict):
        assignments = assignments["assignments"]
    ids = np.asarray([r["compound_id"] for r in rows], dtype=str)
    if ids.tolist() != [r["compound_id"] for r in assignments] or len(rows) != len(binary_matrix):
        raise ValueError("AD features, rows and split assignments are not aligned")
    development = np.flatnonzero([r["dataset_partition"] == "development" for r in assignments])
    test = np.flatnonzero([r["dataset_partition"] == "test" for r in assignments])
    groups = np.asarray([r["structure_group_id"] for r in assignments], dtype=str)
    folds = np.asarray([int(r["cv_validation_fold"] or 0) for r in assignments])
    if len(development) + len(test) != len(rows) or set(groups[development]) & set(groups[test]):
        raise ValueError("AD partitions are incomplete or structurally overlapping")
    similarity = tanimoto_matrix(binary_matrix)
    records, thresholds = [], []

    def evaluate(reference, query, partition, fold):
        if set(groups[reference]) & set(groups[query]):
            raise ValueError("AD reference and query subsets share structural groups")
        lo_max, lo_mean5, lo_nearest = _neighbors(similarity, reference, reference, True)
        tau_max = float(np.quantile(lo_max, 0.05, method="linear"))
        tau_mean5 = float(np.quantile(lo_mean5, 0.05, method="linear"))
        maximum, mean5, nearest = _neighbors(similarity, query, reference)
        thresholds.append({"partition": partition, "fold": fold, "n_reference": len(reference),
                           "n_query": len(query), "quantile": 0.05, "quantile_method": "linear",
                           "max_threshold": tau_max, "mean5_threshold": tau_mean5,
                           "reference_ids_sha256": hashlib.sha256("\n".join(ids[reference]).encode()).hexdigest(),
                           "query_reference_group_overlap": 0})
        for i, a, b, nn in zip(query, maximum, mean5, nearest):
            records.append({"compound_id": ids[i], "partition": partition, "fold": fold,
                            "structure_group_id": groups[i], "max_tanimoto": float(a),
                            "mean5_tanimoto": float(b), "nearest_reference_id": ids[nn],
                            "max_threshold": tau_max, "mean5_threshold": tau_mean5,
                            "inside_ad_max": bool(a >= tau_max), "inside_ad_mean5": bool(b >= tau_mean5),
                            "inside_ad_both": bool(a >= tau_max and b >= tau_mean5)})
        return lo_max, lo_mean5, lo_nearest

    for fold in sorted(set(folds[development])):
        if fold < 1:
            raise ValueError("Development fold IDs must be positive integers")
        evaluate(development[folds[development] != fold], development[folds[development] == fold],
                 "development_oof", int(fold))
    loo_max, loo_mean5, loo_nearest = evaluate(development, test, "test", 0)
    loo = [{"compound_id": ids[i], "max_tanimoto": float(a), "mean5_tanimoto": float(b),
            "nearest_other_development_id": ids[nn], "self_excluded": True}
           for i, a, b, nn in zip(development, loo_max, loo_mean5, loo_nearest)]
    return {"records": records, "thresholds": thresholds, "development_loo": loo,
            "similarity": similarity}


def bh_adjust(p_values):
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH adjustment requires finite P values in [0, 1]")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p), dtype=float)
    result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return result


def structural_key_associations(rows, assignments, maccs166):
    """Fisher tests/BH across all 166 prespecified keys, using development only."""
    matrix = np.asarray(maccs166)
    if matrix.shape != (len(rows), 166) or not np.isin(matrix, [0, 1]).all():
        raise ValueError("MACCS matrix must contain exactly the 166 modeled binary keys")
    development = np.asarray([r["dataset_partition"] == "development" for r in assignments])
    y = np.asarray([r["label"] for r in rows], dtype=int)[development]
    if set(y) != {0, 1}:
        raise ValueError("Development associations require both labels")
    x = matrix[development]
    special = {1: "isotope flag; RDKit implementation leaves this key unset",
               125: "more than one aromatic ring; implemented algorithmically",
               166: "more than one molecular fragment; implemented algorithmically"}
    records = []
    for column in range(166):
        key = column + 1
        smart, minimum_exclusive = MACCSkeys.smartsPatts[key]
        present = x[:, column] == 1
        a, b = int(np.sum(present & (y == 1))), int(np.sum(present & (y == 0)))
        c, d = int(np.sum(~present & (y == 1))), int(np.sum(~present & (y == 0)))
        test = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        # Display effect intervals use the stated Haldane-Anscombe correction.
        corrected = np.asarray([a, b, c, d], float) + 0.5
        log_or = float(np.log(corrected[0] * corrected[3] / (corrected[1] * corrected[2])))
        se = float(np.sqrt(np.sum(1 / corrected)))
        records.append({"MACCS_key": key, "feature_column_zero_based": column, "SMARTS": smart,
                        "match_count_must_exceed": minimum_exclusive,
                        "definition_note": special.get(key, "RDKit SMARTS/count definition"),
                        "positive_present": a, "reference_negative_present": b,
                        "positive_absent": c, "reference_negative_absent": d, "support": a + b,
                        "fisher_p": float(test.pvalue), "corrected_OR": float(np.exp(log_or)),
                        "OR_lower": float(np.exp(log_or - 1.96 * se)), "OR_upper": float(np.exp(log_or + 1.96 * se)),
                        "constant_key": bool(present.all() or not present.any()),
                        "analysis_population": "development only", "family_size": 166})
    for record, q in zip(records, bh_adjust([r["fisher_p"] for r in records])):
        record.update(q_BH=float(q), significant_BH_005=bool(q < 0.05),
                      interval_method="Haldane-Anscombe corrected log-OR Wald 95% CI; exploratory")
    return records


def label_cliff_pairs(rows, similarity, threshold=0.85):
    if not 0 <= threshold <= 1:
        raise ValueError("Label-cliff similarity threshold must lie in [0, 1]")
    y = np.asarray([r["label"] for r in rows], dtype=int)
    eligible = np.triu((similarity >= threshold) & (y[:, None] != y[None, :]), k=1)
    records = []
    for i, j in zip(*np.where(eligible)):
        positive, negative = (i, j) if y[i] == 1 else (j, i)
        a, b = rows[positive], rows[negative]
        records.append({"positive_id": a["compound_id"], "positive_name": a["name"],
                        "positive_CID": a["PubChem_CID"], "positive_SMILES": a["standardized_smiles"],
                        "reference_negative_id": b["compound_id"], "reference_negative_name": b["name"],
                        "reference_negative_CID": b["PubChem_CID"], "reference_negative_SMILES": b["standardized_smiles"],
                        "tanimoto": float(similarity[i, j]), "threshold": threshold})
    return sorted(records, key=lambda r: (-r["tanimoto"], r["positive_id"], r["reference_negative_id"]))


def group_bootstrap_ci(y, probability, groups, n_bootstrap=1000, seed=20260810):
    """Cluster percentile intervals conditional on saved predictions, no refit."""
    y, p, groups = np.asarray(y, int), np.asarray(probability, float), np.asarray(groups, str)
    if not len(y) or len(y) != len(p) or len(y) != len(groups):
        raise ValueError("Group bootstrap inputs must be nonempty and aligned")
    if n_bootstrap < 2:
        raise ValueError("At least two group bootstrap draws are required")
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = {"ROC_AUC": [], "BA": [], "error_rate": []}
    for _ in range(n_bootstrap):
        sampled = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        sy, sp = y[sampled], p[sampled]
        predicted = sp >= 0.5
        samples["error_rate"].append(float(np.mean(predicted != sy)))
        npos, nneg = int(sy.sum()), int(len(sy) - sy.sum())
        if npos and nneg:
            samples["ROC_AUC"].append(float((rankdata(sp)[sy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)))
            samples["BA"].append(float(((predicted[sy == 1]).mean() + (~predicted[sy == 0]).mean()) / 2))
    result = {}
    for metric, values in samples.items():
        result[metric + "_lower"] = float(np.quantile(values, .025)) if values else None
        result[metric + "_upper"] = float(np.quantile(values, .975)) if values else None
        result[metric + "_valid_draws"] = len(values)
    return result


def _primary_ids(bundle, root):
    selection_file = Path(root) / "reports/primary_selection.json"
    if not selection_file.is_file():
        raise ValueError("Run development-only primary selection before reliability reporting")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Primary-model selection is not demonstrably independent of test outcomes")
    primary = selection["primary_model_id"]
    metadata = bundle["metadata"]
    if primary not in metadata:
        raise ValueError("Selected primary is absent from locked models")
    preference = "maccs166" if metadata[primary]["representation"] == "rdkit166" else "rdkit166"
    candidates = [k for k, m in metadata.items() if m["role"] == "main" and m["representation"] == preference]
    mean_auc = selection["mean_outer_auc"]
    comparison = min(candidates, key=lambda n: (-mean_auc[n], n)) if candidates else primary
    graph = [k for k, m in metadata.items() if m["representation"] == "graph" and m["role"] == "main"]
    return primary, comparison, graph


def _count_comparisons(bundle, summary_path):
    if not Path(summary_path).is_file():
        return []
    summaries = read_csv(summary_path)
    result = []
    for counted in summaries:
        representation = counted["representation"]
        if "count" not in representation:
            continue
        binary_rep = representation.replace("_count", "").replace("count_", "").replace("count", "")
        matches = [r for r in summaries if r["representation"] == binary_rep and
                   r["algorithm"] == counted["algorithm"] and r["partition"] == counted["partition"]]
        for binary in matches:
            for metric in ("ROC_AUC", "BA", "MCC", "AP", "Brier"):
                a, b = float(binary[metric]), float(counted[metric])
                result.append({"binary_model_id": binary["model_id"], "count_model_id": counted["model_id"],
                               "algorithm": counted["algorithm"], "binary_representation": binary_rep,
                               "count_representation": representation, "partition": counted["partition"],
                               "metric": metric, "binary_value": a, "count_value": b, "count_minus_binary": b - a,
                               "comparison": "sensitivity analysis; no test-driven selection"})
    return result


def report_features(rows, assignments, matrices, model_results_root, output_dir=None,
                    n_bootstrap=1000, seed=20260810):
    """Generate traceable reliability source tables only after the global lock."""
    root = Path(model_results_root)
    bundle = load_locked_predictions(root, rows, assignments)
    assignments = bundle["assignments"]
    primary, comparison, graphs = _primary_ids(bundle, root)
    out = Path(output_dir) if output_dir else root / "reliability"
    out.mkdir(parents=True, exist_ok=True)
    morgan, maccs = matrices["morgan2048"], matrices["maccs166"]
    if np.shape(morgan) != (len(rows), 2048):
        raise ValueError("Reliability requires the current chiral 2048-bit Morgan representation")
    ad = applicability_domain(rows, assignments, morgan)
    write_csv(out / "applicability_domain.csv", ad["records"])
    write_csv(out / "ad_thresholds.csv", ad["thresholds"])
    write_csv(out / "development_loo_neighbors.csv", ad["development_loo"])
    alerts = structural_key_associations(rows, assignments, maccs)
    write_csv(out / "structural_key_associations.csv", alerts)
    pairs = label_cliff_pairs(rows, ad["similarity"])
    write_csv(out / "binary_label_cliff_pairs.csv", pairs, list(pairs[0]) if pairs else
              ["positive_id", "reference_negative_id", "tanimoto", "threshold"])
    ids = np.asarray([r["compound_id"] for r in rows], str)
    labels = np.asarray([r["label"] for r in rows], int)
    groups = bundle["groups"]
    id_index = {identifier: i for i, identifier in enumerate(ids)}
    ad_index = {(r["partition"], r["compound_id"]): r for r in ad["records"]}
    ci_ids = set([primary, comparison] + graphs)
    performance, calibration, errors, cliffs, cliff_performance, fold_metrics = [], [], [], [], [], []
    for model_id, prediction in bundle["predictions"].items():
        full_probability = np.full(len(rows), np.nan)
        for partition, key, index in (("development_oof", "development", bundle["development_index"]),
                                       ("test", "test", bundle["test_index"])):
            p, y = prediction[key], labels[index]
            full_probability[index] = p
            for item in calibration_bins(y, p):
                calibration.append({"model_id": model_id, "partition": partition, **item})
            for i, probability in zip(index, p):
                if int(probability >= .5) != labels[i]:
                    errors.append({"model_id": model_id, "partition": partition,
                                   "compound_id": ids[i], "name": rows[i]["name"],
                                   "PubChem_CID": rows[i]["PubChem_CID"], "label": int(labels[i]),
                                   "probability": float(probability), "predicted_label": int(probability >= .5),
                                   "normalized_threshold_distance": float(2 * abs(probability - .5)),
                                   "error_type": "FN" if labels[i] else "FP", **ad_index[(partition, ids[i])]})
            for domain in ("max", "mean5"):
                scores = np.asarray([ad_index[(partition, ids[i])][f"{domain}_tanimoto"] for i in index])
                inside = np.asarray([ad_index[(partition, ids[i])][f"inside_ad_{domain}"] for i in index])
                strata = [("all", np.ones(len(index), bool)), ("inside", inside), ("outside", ~inside)]
                if domain == "max":
                    strata += [(f"similarity_{low:.2f}_{high:.2f}", (scores >= low) & ((scores < high) if high < 1 else (scores <= high)))
                               for low, high in zip((0, .25, .5, .75), (.25, .5, .75, 1))]
                for stratum, mask in strata:
                    item = {"model_id": model_id, "partition": partition, "domain": domain,
                            "stratum": stratum, "n": int(mask.sum()), "positive": int(y[mask].sum()),
                            "reference_negative": int(mask.sum() - y[mask].sum())}
                    if mask.any():
                        values = metric_values(y[mask], p[mask])
                        if len(set(y[mask])) < 2:
                            values["BA"] = None
                        item.update(values, error_rate=1 - values["ACC"])
                        if model_id in ci_ids and stratum in {"inside", "outside"}:
                            item.update(group_bootstrap_ci(y[mask], p[mask], groups[index[mask]], n_bootstrap, seed))
                    performance.append(item)
            if partition == "development_oof":
                for fold in sorted(set(bundle["folds"])):
                    mask = bundle["folds"] == fold
                    fold_metrics.append({"model_id": model_id, "fold": int(fold), "n": int(mask.sum()),
                                         **metric_values(y[mask], p[mask])})
        endpoints = set()
        for pair in pairs:
            a, b = id_index[pair["positive_id"]], id_index[pair["reference_negative_id"]]
            endpoints.update((a, b))
            pa, pb = float(full_probability[a]), float(full_probability[b])
            cliffs.append({"model_id": model_id, **pair, "positive_probability": pa,
                           "reference_negative_probability": pb, "positive_minus_negative": pa - pb,
                           "both_correct": bool(pa >= .5 and pb < .5),
                           "positive_prediction_source": "development_oof" if a in set(bundle["development_index"]) else "test",
                           "reference_negative_prediction_source": "development_oof" if b in set(bundle["development_index"]) else "test"})
        for partition, index in (("development_oof", bundle["development_index"]), ("test", bundle["test_index"])):
            unique = np.asarray(sorted(endpoints & set(index)), int)
            if len(unique):
                cliff_performance.append({"model_id": model_id, "partition": partition,
                    "n_unique_compounds": len(unique), "pair_dependence_handling": "deduplicate endpoints before metrics",
                    **metric_values(labels[unique], full_probability[unique])})
    write_csv(out / "ad_performance.csv", performance)
    write_csv(out / "calibration_bins.csv", calibration)
    write_csv(out / "classification_errors.csv", errors)
    write_csv(out / "label_cliff_predictions.csv", cliffs)
    write_csv(out / "label_cliff_unique_endpoint_performance.csv", cliff_performance)
    write_csv(out / "perfold_metrics.csv", fold_metrics)
    count_records = _count_comparisons(bundle, root / "reports/model_summary.csv")
    write_csv(out / "binary_count_comparisons.csv", count_records)
    source = [{"compound_id": r["compound_id"], "name": r["name"], "PubChem_CID": r["PubChem_CID"],
               "standardized_smiles": r["standardized_smiles"], "label": r["label"],
               "partition": assignments[i]["dataset_partition"]} for i, r in enumerate(rows)]
    write_csv(out / "compound_identity.csv", source)
    manifest = {"schema_version": 1, "status": "complete", "source_rows_fingerprint": bundle["lock"]["source_rows_fingerprint"],
                "source_excel_sha256": bundle["lock"].get("source_excel_sha256"),
                "protocol_sha256": bundle["lock"].get("protocol_sha256"),
                "global_lock_sha256": hashlib.sha256((root / "model_lock.json").read_bytes()).hexdigest(),
                "primary_model_id": primary, "comparison_model_id": comparison, "graph_model_ids": graphs,
                "model_metadata": {k: {name: m[name] for name in ("algorithm", "representation", "role")} for k, m in bundle["metadata"].items()},
                "n_development": len(bundle["development_index"]), "n_test": len(bundle["test_index"]),
                "AD_rule": "max or mean5 similarity >= training leave-one-out 5th percentile; separate definitions",
                "AD_threshold_scope": "OOF: each outer training fold only; test: all development only",
                "AD_percentile_method": "numpy linear quantile", "AD_fingerprint": "Morgan radius2 2048bits includeChirality=True",
                "group_bootstrap": {"n_bootstrap": n_bootstrap, "seed": seed, "model_ids": sorted(ci_ids),
                                    "method": "resample complete structure groups within each displayed domain; percentile 95% CI",
                                    "scope": "conditional on saved predictions; no retraining; one-class AUC/BA draws omitted"},
                "n_label_cliff_pairs": len(pairs), "label_cliff_threshold": .85,
                "label_cliff_interpretation": "binary discordant-label pairs, not experimentally established activity cliffs",
                "association_family": "all 166 MACCS keys; development labels only; two-sided Fisher; BH family166",
                "n_significant_keys_BH005": sum(r["significant_BH_005"] for r in alerts),
                "association_limits": "exploratory structural associations; no kidney-specific toxicophore or causal claim; Fisher assumes independent compounds despite structural relatedness",
                "n_count_comparison_records": len(count_records),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.csv")}}
    write_json(out / "reliability_manifest.json", manifest)
    return manifest
