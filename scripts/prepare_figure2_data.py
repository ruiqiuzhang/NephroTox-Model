"""Recompute positive-compound chemical-space clustering from the frozen DATA.

No saved historical coordinates, class labels or representative lists are read.
The protocol is written before any clustering is run. Clusters are descriptive
partitions of molecular fingerprints, not toxicity mechanisms or natural classes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.data import load_data  # noqa: E402

BASE_SEED = 20261019
IDENTITY = ["compound_id", "name", "PubChem_CID", "label", "standardized_smiles", "standardized_InChIKey"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], headers: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol(n: int, fixed_k: int | None = None) -> dict:
    if fixed_k is not None and fixed_k not in range(2, 11):
        raise ValueError("fixed_k must be an integer in 2..10")
    return {
        "protocol_version": "1.1", "frozen_before_computation": True,
        "display_mode": "user_requested_fixed_k" if fixed_k is not None else "automatic_candidate_rule",
        "user_requested_fixed_k": fixed_k,
        "analysis_population": "All and only label=1 records in the release DATA workbook; no classifier fitting or split changes",
        "fingerprint": {"algorithm": "RDKit Morgan bit fingerprint", "radius": 2, "fpSize": 2048,
                        "includeChirality": True, "useBondTypes": True, "countSimulation": False},
        "distance": "1 - Tanimoto similarity of the binary Morgan fingerprints",
        "clustering": {
            "algorithm": "Alternating k-medoids with squared-distance-weighted k-medoids++ initialisation; not PAM",
            "candidate_k": list(range(2, 11)), "initialisations_per_k": 10,
            "seed_rule": "20261019 + 1000*K + start_index, start_index=0,...,9",
            "max_iter": 300, "objective": "sum of distances from each molecule to its assigned medoid",
            "per_k_run_selection": "Minimum objective; numerical ties within 1e-10 select the lexicographically smallest ordered medoid IDs",
            "minimum_cluster_size": max(10, math.ceil(.02 * n)),
            "k_selection": "Among per-K best-objective runs meeting the minimum size and with all 10 starts converged, maximise silhouette on original precomputed distances; within 0.01 of the maximum choose the smallest K",
            "display_k_rule": "Use user-requested fixed K when supplied; retain automatic recommendation for transparency, without claiming fixed K maximises silhouette" if fixed_k is not None else "Use automatic candidate rule",
            "tie_assignment": "Lowest medoid compound ID, with medoids retained in their own cluster",
            "cluster_numbering": "Ascending representative compound ID; numbers have no biological ranking",
            "scope": "Candidate K=2..10 defines a coarse, displayable exploratory partition; no claim of natural class boundaries",
        },
        "stability": {"validation_only_not_used_to_retune_k": True,
                      "multi_start": "Adjusted Rand index of each candidate initialisation against that K's best-objective run",
                      "subsamples": 20, "fraction": .8, "sampling": "Without replacement",
                      "sample_size_rule": "floor(0.8*n)", "initialisations_per_subsample": 5,
                      "seed_rule": "sample=20261019+100000+replicate; fit=20261019+200000+100*replicate+start",
                      "comparison": "ARI of independently reclustered subset labels versus full-data selected labels restricted to those same compounds; also medoid-matched full-data reassignment ARI"},
        "representatives": "Actual molecule maximising mean Tanimoto similarity to every other member of its assigned cluster (self excluded); numerical ties within 1e-12 choose smallest compound ID",
        "medoid_update_tie_rule": "Within-cluster distance-sum numerical ties within 1e-12 choose smallest compound ID",
        "intra_cluster_summary": "Each unordered distinct-molecule pair once; descriptive median, 25th/75th percentile and mean; no independence-based test of pairs",
        "heatmap_order": "Within each fixed cluster, average-linkage hierarchical ordering of 1-Tanimoto; for display only, not the partition algorithm",
        "tsne": {"n_components": 2, "metric": "precomputed", "init": "random", "perplexity": 40.0,
                 "random_state": BASE_SEED, "max_iter": 2000, "learning_rate": "auto", "method": "barnes_hut",
                 "early_exaggeration": 12.0, "angle": .5, "n_jobs": 1,
                 "role": "Display only; clustering, medoid selection and silhouette do not use the projection"},
        "tsne3d": {"enabled": fixed_k == 4, "n_components": 3, "metric": "precomputed", "init": "random", "perplexity": 40.0,
                   "random_state": BASE_SEED, "max_iter": 2000, "learning_rate": "auto", "method": "barnes_hut",
                   "early_exaggeration": 12.0, "angle": .5, "n_jobs": 1,
                   "role": "Companion display only in the user-requested four-cluster original layout; clustering does not use either embedding"},
        "interpretation": "Positive-label structural heterogeneity only. Similarity does not establish common nephrotoxicity mechanism, clinical risk, or prospective predictive performance.",
        "source_smiles": "Preserved verbatim; no new stereocentres assigned, no standardisation or deduplication performed here",
    }


def initialise_medoids(distances: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Distinct medoids sampled by squared distance to the closest chosen medoid."""
    rng = np.random.default_rng(seed)
    n = len(distances)
    selected = [int(rng.integers(n))]
    nearest = distances[:, selected[0]].copy()
    while len(selected) < k:
        weights = nearest ** 2
        weights[selected] = 0
        if weights.sum() > 0:
            new = int(rng.choice(n, p=weights / weights.sum()))
        else:
            new = int(rng.choice(np.setdiff1d(np.arange(n), selected)))
        selected.append(new)
        nearest = np.minimum(nearest, distances[:, new])
    return np.sort(np.asarray(selected, dtype=np.int64))


def assign(distances: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    labels = np.argmin(distances[:, medoids], axis=1)
    # Fingerprint collisions are possible; a medoid represents its own cluster.
    labels[medoids] = np.arange(len(medoids))
    return labels


def alternating_kmedoids(distances: np.ndarray, k: int, seed: int, max_iter: int = 300) -> dict:
    """Deterministic alternating assignment and exact within-cluster medoid update.

    Input ordering must be lexicographic compound ID so all argmin ties have a
    stable identity rule. This local optimisation is explicitly not PAM swaps.
    """
    medoids = initialise_medoids(distances, k, seed)
    converged = False
    previous_objective = float("inf")
    for iteration in range(1, max_iter + 1):
        labels = assign(distances, medoids)
        objective = float(distances[np.arange(len(distances)), medoids[labels]].sum())
        if objective > previous_objective + 1e-10:
            raise ValueError("Alternating k-medoids objective increased")
        previous_objective = objective
        updated = []
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                raise ValueError("Unexpected empty medoid cluster")
            totals = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[np.flatnonzero(np.isclose(totals, totals.min(), rtol=0, atol=1e-12))[0]]))
        updated = np.sort(np.asarray(updated, dtype=np.int64))
        if np.array_equal(updated, medoids):
            converged = True
            break
        medoids = updated
    labels = assign(distances, medoids)
    return {"medoids": medoids, "labels": labels, "objective": float(distances[np.arange(len(distances)), medoids[labels]].sum()),
            "iterations": iteration, "converged": converged, "seed": seed}


def best_run(runs: list[dict]) -> dict:
    minimum = min(run["objective"] for run in runs)
    tied = [run for run in runs if np.isclose(run["objective"], minimum, rtol=0, atol=1e-10)]
    return min(tied, key=lambda run: tuple(run["medoids"].tolist()))


def select_k(evaluations: list[dict]) -> dict:
    eligible = [row for row in evaluations if row["eligible"]]
    if not eligible:
        raise ValueError("No candidate partition meets the predeclared minimum cluster size; no forced K is produced")
    highest = max(row["silhouette"] for row in eligible)
    return min((row for row in eligible if row["silhouette"] >= highest - .01), key=lambda row: row["k"])


def distribution(values: np.ndarray) -> dict:
    q1, median, q3 = np.quantile(values, [.25, .5, .75], method="linear")
    return {"n_pairs": int(len(values)), "median": float(median), "q1": float(q1), "q3": float(q3),
            "IQR": float(q3 - q1), "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def prepare(output: Path, fixed_k: int | None = None) -> dict:
    all_rows, source_info = load_data()
    rows = sorted((row for row in all_rows if row["label"] == 1), key=lambda row: row["compound_id"])
    if len(all_rows) != 1673 or len(rows) != 842:
        raise ValueError("Figure 2 requires the frozen 1,673-record DATA with exactly 842 positive records")
    output.mkdir(parents=True, exist_ok=True)
    fixed = protocol(len(rows), fixed_k=fixed_k)
    write_json(output / "analysis_protocol.json", fixed)
    protocol_sha = hashlib.sha256((output / "analysis_protocol.json").read_bytes()).hexdigest()
    molecules = [Chem.MolFromSmiles(row["standardized_smiles"]) for row in rows]
    before = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in molecules]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True,
                                                          useBondTypes=True, countSimulation=False)
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    similarities = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, fingerprints) for fp in fingerprints], dtype=np.float64)
    distances = 1.0 - similarities
    if not np.array_equal(distances, distances.T) or np.any(np.diag(distances)) or not np.isfinite(distances).all():
        raise ValueError("Invalid fingerprint distance matrix")
    collision_i, collision_j = np.nonzero(np.triu(distances == 0, k=1))
    collisions = [{"compound_id_1": rows[int(i)]["compound_id"], "name_1": rows[int(i)]["name"],
                   "PubChem_CID_1": rows[int(i)]["PubChem_CID"],
                   "compound_id_2": rows[int(j)]["compound_id"], "name_2": rows[int(j)]["name"],
                   "PubChem_CID_2": rows[int(j)]["PubChem_CID"], "Tanimoto_similarity": 1.0,
                   "handling": "Both retained: identical finite-radius binary fingerprints do not imply identical molecules"}
                  for i, j in zip(collision_i, collision_j)]
    write_csv(output / "figure2_identical_fingerprint_pairs.csv", collisions,
              ["compound_id_1", "name_1", "PubChem_CID_1", "compound_id_2", "name_2", "PubChem_CID_2", "Tanimoto_similarity", "handling"])
    np.savez_compressed(output / "figure2_pairwise_matrices.npz", compound_id=np.asarray([row["compound_id"] for row in rows]),
                        tanimoto_similarity=similarities, tanimoto_distance=distances,
                        source_sha256=np.asarray(source_info["sha256"]), analysis_protocol_sha256=np.asarray(protocol_sha))
    n, all_runs, evaluations, start_records = len(rows), {}, [], []
    print(f"DATA validated: {len(all_rows)} total, {n} positive; computing K=2..10", flush=True)
    for k in fixed["clustering"]["candidate_k"]:
        runs = [alternating_kmedoids(distances, k, BASE_SEED + 1000 * k + start) for start in range(10)]
        winner = best_run(runs)
        all_runs[k] = winner
        sizes = np.bincount(winner["labels"], minlength=k)
        silhouette = float(silhouette_score(distances, winner["labels"], metric="precomputed"))
        for start, run in enumerate(runs):
            start_records.append({"k": k, "start": start, "seed": run["seed"], "objective": run["objective"],
                                  "iterations": run["iterations"], "converged": run["converged"],
                                  "ARI_vs_best_objective_run": float(adjusted_rand_score(winner["labels"], run["labels"])),
                                  "medoid_compound_ids": ";".join(rows[index]["compound_id"] for index in run["medoids"])})
        aris = [record["ARI_vs_best_objective_run"] for record in start_records if record["k"] == k]
        evaluation = {"k": k, "objective": winner["objective"], "silhouette": silhouette,
                      "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
                      "cluster_sizes": ";".join(str(int(value)) for value in sizes),
                      "eligible": bool(sizes.min() >= fixed["clustering"]["minimum_cluster_size"] and all(run["converged"] for run in runs)),
                      "best_seed": winner["seed"], "all_10_runs_converged": bool(all(run["converged"] for run in runs)),
                      "multistart_ARI_mean": float(np.mean(aris)), "multistart_ARI_min": float(np.min(aris)),
                      "multistart_ARI_median": float(np.median(aris))}
        evaluations.append(evaluation)
        print(f"K={k}: silhouette={silhouette:.6f}, sizes={sizes.tolist()}, eligible={evaluation['eligible']}", flush=True)
    automatic_selected = select_k(evaluations)
    selected = next(row for row in evaluations if row["k"] == fixed_k) if fixed_k is not None else automatic_selected
    if not selected["eligible"]:
        raise ValueError("Requested fixed K does not meet the declared minimum-size/convergence criteria")
    k = selected["k"]
    winner = all_runs[k]
    labels = winner["labels"].copy()
    for row in evaluations:
        row["selected"] = row["k"] == k
        row["automatically_recommended"] = row["k"] == automatic_selected["k"]
    write_csv(output / "figure2_k_evaluation.csv", evaluations)
    write_csv(output / "figure2_multistart_stability.csv", start_records)
    print(f"SELECTED K={k}: {selected}", flush=True)

    stability = []
    for replicate in range(20):
        sample_seed = BASE_SEED + 100000 + replicate
        sample = np.sort(np.random.default_rng(sample_seed).choice(n, size=math.floor(.8 * n), replace=False))
        subset_distances = distances[np.ix_(sample, sample)]
        fit = best_run([alternating_kmedoids(subset_distances, k, BASE_SEED + 200000 + 100 * replicate + start) for start in range(5)])
        subset_labels = fit["labels"]
        original_medoids = sample[fit["medoids"]]
        reassigned_full_labels = assign(distances, original_medoids)
        stability.append({"replicate": replicate + 1, "sampling_seed": sample_seed, "n_sample": len(sample),
                          "best_fit_seed": fit["seed"], "objective_in_subset": fit["objective"],
                          "converged": fit["converged"], "min_cluster_size_in_subset": int(np.bincount(subset_labels, minlength=k).min()),
                          "ARI_on_shared_compounds": float(adjusted_rand_score(labels[sample], subset_labels)),
                          "ARI_full_data_reassignment": float(adjusted_rand_score(labels, reassigned_full_labels)),
                          "medoid_compound_ids": ";".join(rows[index]["compound_id"] for index in original_medoids)})
    write_csv(output / "figure2_subsample_stability.csv", stability)
    print("Subsample stability complete; computing fixed-distance t-SNE", flush=True)
    tsne_parameters = {key: value for key, value in fixed["tsne"].items() if key != "role"}
    tsne = TSNE(**tsne_parameters)
    embedding = tsne.fit_transform(distances)
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite t-SNE coordinates")
    embedding3d, tsne3d_metadata = None, {"enabled": False}
    if fixed["tsne3d"]["enabled"]:
        print("Computing companion 3D t-SNE for the user-requested four-cluster layout", flush=True)
        tsne3d_parameters = {key: value for key, value in fixed["tsne3d"].items() if key not in ("enabled", "role")}
        tsne3d = TSNE(**tsne3d_parameters)
        embedding3d = tsne3d.fit_transform(distances)
        if not np.isfinite(embedding3d).all():
            raise ValueError("Non-finite three-dimensional t-SNE coordinates")
        tsne3d_metadata = {"enabled": True, **tsne3d_parameters, "kl_divergence": float(tsne3d.kl_divergence_),
                           "iterations": int(tsne3d.n_iter_), "effective_learning_rate": float(tsne3d.learning_rate_),
                           "role": fixed["tsne3d"]["role"]}

    representatives, summaries, mean_within, orders, pair_arrays = [], [], np.zeros(n), [], {}
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        sub_sim = similarities[np.ix_(members, members)]
        means = (sub_sim.sum(axis=1) - 1) / (len(members) - 1)
        mean_within[members] = means
        maximum = means.max()
        best_local = int(np.flatnonzero(np.isclose(means, maximum, rtol=0, atol=1e-12))[0])
        representative_index = int(members[best_local])
        # A cluster medoid and maximum mean similarity are equivalent here.
        representative_distance_sum = distances[representative_index, members].sum()
        if not np.isclose(representative_distance_sum, distances[np.ix_(members, members)].sum(axis=1).min(), atol=1e-10, rtol=0):
            raise ValueError("Representative is not a within-cluster medoid")
        representative = {**{key: rows[representative_index][key] for key in IDENTITY}, "cluster": cluster + 1,
                          "n_cluster": len(members), "mean_within_similarity_excluding_self": float(maximum),
                          "same_as_optimisation_medoid": bool(representative_index == winner["medoids"][cluster])}
        representatives.append(representative)
        pairs = sub_sim[np.triu_indices(len(members), k=1)]
        pair_arrays[f"cluster_{cluster + 1}"] = pairs
        summaries.append({"cluster": cluster + 1, "n_compounds": len(members),
                          "fraction_of_842": float(len(members) / n), "representative_compound_id": representative["compound_id"],
                          "representative_name": representative["name"], "representative_PubChem_CID": representative["PubChem_CID"],
                          "representative_mean_similarity_excluding_self": float(maximum), **distribution(pairs)})
        ordering = leaves_list(linkage(squareform(1 - sub_sim, checks=True), method="average", optimal_ordering=True))
        orders.extend(members[ordering].tolist())
    # Number by representative IDs, independent of random initialisation labels.
    sorted_original = sorted(range(k), key=lambda cluster: representatives[cluster]["compound_id"])
    remap = {old + 1: new + 1 for new, old in enumerate(sorted_original)}
    representatives = sorted(({**row, "cluster": remap[row["cluster"]]} for row in representatives), key=lambda row: row["cluster"])
    summaries = sorted(({**row, "cluster": remap[row["cluster"]]} for row in summaries), key=lambda row: row["cluster"])
    remapped_pairs = {f"cluster_{remap[old + 1]}": pair_arrays[f"cluster_{old + 1}"] for old in range(k)}
    reordered_heatmap = []
    for old in sorted_original:
        reordered_heatmap.extend(index for index in orders if labels[index] == old)
    representative_ids = {row["compound_id"] for row in representatives}
    records = [{**{key: row[key] for key in IDENTITY}, "cluster": remap[int(labels[index]) + 1],
                "tsne_1": float(embedding[index, 0]), "tsne_2": float(embedding[index, 1]),
                **({f"tsne_3d_{axis + 1}": float(embedding3d[index, axis]) for axis in range(3)} if embedding3d is not None else {}),
                "mean_within_similarity_excluding_self": float(mean_within[index]),
                "is_representative": row["compound_id"] in representative_ids}
               for index, row in enumerate(rows)]
    heatmap_order_records = [{"heatmap_position": position, "compound_id": rows[index]["compound_id"],
                              "matrix_index": index, "cluster": remap[int(labels[index]) + 1]} for position, index in enumerate(reordered_heatmap)]
    np.savez_compressed(output / "figure2_intra_cluster_similarities.npz", **remapped_pairs)
    write_csv(output / "figure2_compound_clusters.csv", records)
    write_csv(output / "figure2_representatives.csv", representatives)
    write_csv(output / "figure2_cluster_summaries.csv", summaries)
    write_csv(output / "figure2_heatmap_order.csv", heatmap_order_records)
    after = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in molecules]
    if before != after:
        raise ValueError("Graph or stereochemistry changed during fingerprint calculation")
    ari_shared = np.asarray([row["ARI_on_shared_compounds"] for row in stability])
    ari_full = np.asarray([row["ARI_full_data_reassignment"] for row in stability])
    metadata = {
        "source": source_info, "source_n_records": len(all_rows), "analysis_n_records": n, "analysis_label": 1,
        "source_label_counts": {str(key): count for key, count in sorted(Counter(row["label"] for row in all_rows).items())},
        "no_historical_numerical_input": True, "source_smiles_preserved_verbatim": True,
        "graph_and_stereochemistry_unchanged": before == after, "excluded_positive_records": [],
        "analysis_protocol_file": "analysis_protocol.json", "analysis_protocol_sha256": protocol_sha,
        "python_version": platform.python_version(), "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "scikit_learn_version": sklearn.__version__,
        "fingerprint": fixed["fingerprint"], "distance": fixed["distance"], "selected_k": k,
        "display_mode": fixed["display_mode"], "user_requested_fixed_k": fixed_k,
        "auto_selected_k": automatic_selected["k"],
        "k_selection": {"rule": fixed["clustering"]["display_k_rule"], "automatic_rule": fixed["clustering"]["k_selection"], "chosen": selected,
                        "automatically_recommended": automatic_selected,
                        "maximum_eligible_silhouette": max(row["silhouette"] for row in evaluations if row["eligible"]),
                        "all_candidates": "2 through 10, fixed before computation", "minimum_cluster_size": fixed["clustering"]["minimum_cluster_size"]},
        "clustering_algorithm": fixed["clustering"]["algorithm"],
        "tsne": {**tsne_parameters, "kl_divergence": float(tsne.kl_divergence_), "iterations": int(tsne.n_iter_), "effective_learning_rate": float(tsne.learning_rate_), "role": fixed["tsne"]["role"]},
        "tsne3d": tsne3d_metadata,
        "stability_summary": {"n_subsamples": 20, "n_each": math.floor(.8 * n), "fraction": .8,
                              "ARI_shared_mean": float(ari_shared.mean()), "ARI_shared_median": float(np.median(ari_shared)),
                              "ARI_shared_min": float(ari_shared.min()), "ARI_shared_max": float(ari_shared.max()),
                              "ARI_full_reassignment_mean": float(ari_full.mean()), "ARI_full_reassignment_median": float(np.median(ari_full)),
                              "interpretation": "Validation of a fixed descriptive partition, not evidence of natural or mechanistic classes; K was not retuned using stability"},
        "representative_rule": fixed["representatives"], "heatmap_order": fixed["heatmap_order"],
        "interpretation": fixed["interpretation"],
        "fingerprint_collision_pairs_distinct_records": len(collisions),
        "identical_fingerprint_pairs": collisions,
        "fingerprint_resolution_caveat": "Two distinct records may have identical finite-radius hashed binary fingerprints; all source structures are retained. This is a representation limitation, not a duplicate-identity finding.",
    }
    source = {"metadata": metadata, "records": records, "representatives": representatives,
              "cluster_summaries": summaries, "k_evaluation": evaluations,
              "multistart_stability": start_records, "subsample_stability": stability,
              "heatmap_order": heatmap_order_records}
    write_json(output / "figure2_source_data.json", source)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file() and path.name != "figure2_analysis_manifest.json"}
    write_json(output / "figure2_analysis_manifest.json", {"metadata": metadata, "files_sha256": hashes})
    print(json.dumps({"selected_k": k, "clusters": summaries, "stability": metadata["stability_summary"]}, ensure_ascii=False, indent=2), flush=True)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/figure2/source_data")
    parser.add_argument("--fixed-k", type=int, choices=range(2, 11), default=None,
                        help="Explicit user-requested display K; candidate evaluation and automatic recommendation remain disclosed")
    args = parser.parse_args()
    prepare(args.output.resolve(), fixed_k=args.fixed_k)


if __name__ == "__main__":
    main()
