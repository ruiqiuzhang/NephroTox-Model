"""Render post-lock search, stopping, seed and measured-efficiency supplements.

All source values come from the current globally locked models. This script
does not fit models or select parameters, thresholds, seeds or epochs. Missing
training/benchmark inputs fail before an output directory is created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figure_style import apply_font
from nephrotox.data import load_data
from nephrotox.modeling_reporting import load_locked_predictions, metric_values

NEURAL = {"DNN", "DMPNN", "GINE", "AttentiveFP"}
ALG_ORDER = ("RF", "SVM", "MLP", "XGB", "DNN", "DMPNN", "GINE", "AttentiveFP")
REP_LABEL = {"morgan2048": "Morgan", "maccs166": "MACCS", "atompair1024": "AtomPair",
             "torsion1024": "TopologicalTorsion", "rdkit166": "Descriptor", "graph": "Graph",
             "morgan_count2048": "Morgan count", "atompair_count1024": "AtomPair count",
             "torsion_count1024": "TopologicalTorsion count"}
ALG_LABEL = {"XGB": "XGBoost", "DMPNN": "D-MPNN"}
REP_COLOR = {"morgan2048": "#ef4e3b", "maccs166": "#367faf", "atompair1024": "#f5bf45",
             "torsion1024": "#54b58c", "rdkit166": "#847bab", "graph": "#805b47"}
SEED_COLORS = ("#367faf", "#ed793c", "#847bab")

FIGURE_CONTRACT = {
    "claim": "Describe the finite search, training-internal stopping, seed variation and actual measured computational cost without test-driven optimization.",
    "archetype": "quantitative grids", "backend": "Python / matplotlib", "font": "Arial",
    "S8": "Full-development candidate mean inner ROC-AUC: separate 40-by-3 conventional/count and 3-by-20 graph panels; actual inner-fold scores for every candidate.",
    "S9": "Eight main neural models, training-internal epoch-selector monitor and fresh full-development fixed-epoch training loss; three recorded seeds.",
    "S10": "Eleven neural configurations, individual-seed and arithmetic-ensemble development OOF/test ROC-AUC; descriptive seed variation, no confidence interval claim.",
    "S11": "Recomputed feature wall time, final development-fit wall time, repeated batch/single inference; measurements conditional on actual hardware/workload.",
    "integrity": "No historical numerical results, fabricated learning curves, test-driven selection or unobserved runtime estimates.",
    "export": "PDF, editable-text SVG, PNG and LZW TIFF at 600 dpi; complete plotted source CSV/JSON and hashes.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        raise ValueError(f"Cannot publish an empty diagnostic source: {Path(path).name}")
    fields = list(dict.fromkeys(k for record in records for k in record))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def model_label(meta):
    algorithm = ALG_LABEL.get(meta["algorithm"], meta["algorithm"])
    return algorithm if meta["representation"] == "graph" else REP_LABEL[meta["representation"]] + "–" + algorithm


def verify_table_manifest(folder, filename, lock_sha):
    manifest = read_json(folder / filename)
    if manifest.get("status") != "complete" or manifest.get("global_lock_sha256") != lock_sha:
        raise ValueError(f"Incomplete or mismatched diagnostic source manifest: {filename}")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f"Diagnostic source changed after validation: {name}")
    return manifest


def search_source(results, bundle, rows, protocol):
    """Verify full-development search against saved inner validation probabilities."""
    ids = np.asarray([r["compound_id"] for r in rows])
    labels = {r["compound_id"]: r["label"] for r in rows}
    groups = dict(zip(ids, bundle["groups"]))
    development_ids = set(ids[bundle["development_index"]])
    registry = {j["model_id"]: j for j in protocol["registry"]}
    records, sources = [], {}
    for model_id in bundle["lock"]["expected_model_ids"]:
        meta, job = bundle["metadata"][model_id], registry[model_id]
        folder = results / model_id / "full_development_search"
        path = folder / "selection.json"
        selected = read_json(path)
        sources[path.relative_to(results).as_posix()] = digest(path)
        candidates = protocol["graph_candidate_parameters"][job["estimator_key"]] if meta["representation"] == "graph" else protocol["candidate_parameters"][job["estimator_key"]]
        required_candidates = 20 if meta["representation"] == "graph" else 3
        if len(candidates) != required_candidates:
            raise ValueError(f"Declared parameter design requires exactly {required_candidates} candidates: {model_id}")
        if selected["protocol_sha256"] != bundle["lock"]["protocol_sha256"] or selected["selected_params"] != meta["params"] or len(selected["trials"]) != len(candidates):
            raise ValueError(f"Full-development candidate selection does not match the locked model: {model_id}")
        means = []
        for i, (trial_summary, params) in enumerate(zip(selected["trials"], candidates), 1):
            if trial_summary["candidate"] != i or trial_summary["params"] != params or len(trial_summary["inner_auc"]) != 3:
                raise ValueError("Candidate definitions differ from the frozen finite search")
            scores, validation_seen = [], []
            for inner in range(1, 4):
                trial_file = folder / f"candidate_{i:02d}/inner_{inner}/trial.json"
                probability_file = trial_file.parent / "validation_predictions.npz"
                trial = read_json(trial_file)
                a, b = trial["training_ids"], trial["validation_ids"]
                if (len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b)
                        or set(a) | set(b) != development_ids
                        or set(groups[n] for n in a) & set(groups[n] for n in b)):
                    raise ValueError("HPO source contains a wrong population or crossed structure group")
                if trial["params"] != params or trial["protocol_sha256"] != bundle["lock"]["protocol_sha256"] or digest(probability_file) != trial["prediction_sha256"]:
                    raise ValueError("HPO trial/probability provenance mismatch")
                with np.load(probability_file, allow_pickle=False) as saved:
                    if not np.array_equal(saved["compound_id"], b):
                        raise ValueError("HPO validation IDs differ from saved predictions")
                    score = metric_values([labels[n] for n in b], saved["probability"])["ROC_AUC"]
                if not np.isclose(score, trial["roc_auc"], atol=1e-12, rtol=0) or not np.isclose(score, trial_summary["inner_auc"][inner - 1], atol=1e-12, rtol=0):
                    raise ValueError("HPO source AUC does not match saved inner predictions")
                scores.append(score)
                validation_seen.extend(b)
                sources[trial_file.relative_to(results).as_posix()] = digest(trial_file)
                sources[probability_file.relative_to(results).as_posix()] = digest(probability_file)
            if len(validation_seen) != len(development_ids) or set(validation_seen) != development_ids:
                raise ValueError("Inner folds do not validate every development compound once")
            mean = float(np.mean(scores))
            if not np.isclose(mean, trial_summary["mean_inner_auc"], atol=1e-12, rtol=0):
                raise ValueError("HPO candidate mean is inconsistent")
            # Preserve the exact original training-time floating-point tie
            # decision after independently verifying its numerical value.
            means.append(float(trial_summary["mean_inner_auc"]))
            records.append({"model_id": model_id, "model_label": model_label(meta), "representation": meta["representation"], "candidate": i,
                "n_inner_folds": 3, "mean_inner_ROC_AUC": mean, "sample_sd_inner_ROC_AUC": float(np.std(scores, ddof=1)),
                "selected": i == selected["selected_candidate"], "parameters_json": json.dumps(params, sort_keys=True),
                "inner_auc_json": json.dumps(scores), "scope": "full-development inner grouped CV; not held-out test performance"})
        if selected["selected_candidate"] != int(np.argmax(means)) + 1:
            raise ValueError("HPO candidate choice violates the frozen maximum-mean/first-tie rule")
    return records, sources


def learning_curve_source(bundle):
    """Parse the native, locked metadata; never relabel AUC as BCE or vice versa."""
    records, selected_epochs = [], []
    for model_id in bundle["lock"]["expected_model_ids"]:
        meta = bundle["metadata"][model_id]
        if meta["role"] != "main" or meta["algorithm"] not in NEURAL:
            continue
        fit = meta["final_fit_metadata"]
        is_dnn = meta["algorithm"] == "DNN"
        members = fit["seeds_detail"] if is_dnn else fit["members"]
        if len(members) != 3 or [m["seed"] for m in members] != fit["seeds"]:
            raise ValueError("Main neural learning curves must contain all three declared seeds")
        for member in members:
            selection = member["epoch_selection_history"] if is_dnn else member["epoch_selection_curve"]
            refit = member["refit_history"] if is_dnn else member["refit_curve"]
            epoch = int(member["selected_epoch"])
            if [r["epoch"] for r in selection] != list(range(1, len(selection) + 1)) or [r["epoch"] for r in refit] != list(range(1, epoch + 1)) or not 1 <= epoch <= len(selection):
                raise ValueError("Recorded epoch selector/refit histories are incomplete")
            selected_epochs.append({"model_id": model_id, "model_label": model_label(meta), "seed": member["seed"],
                "selected_epoch": epoch, "selector_epochs_run": len(selection), "refit_epochs_run": len(refit),
                "selector_monitor": "validation ROC-AUC" if is_dnn else "validation BCE", "refit_scope": "all development records"})
            for phase, history in (("epoch_selector", selection), ("full_development_refit", refit)):
                for row in history:
                    loss = row["training_loss"] if is_dnn else row["training_weighted_bce"]
                    monitor = row.get("validation_roc_auc") if is_dnn else row.get("validation_bce")
                    if not np.isfinite(loss) or (phase == "epoch_selector" and (monitor is None or not np.isfinite(monitor))):
                        raise ValueError("Neural diagnostic history contains an invalid loss or monitor")
                    records.append({"model_id": model_id, "model_label": model_label(meta), "algorithm": meta["algorithm"],
                        "seed": member["seed"], "phase": phase, "epoch": int(row["epoch"]),
                        "training_weighted_BCE": float(loss), "selector_monitor_name": "validation ROC-AUC" if is_dnn else "validation BCE",
                        "selector_monitor_value": float(monitor) if monitor is not None else None,
                        "learning_rate": float(row["learning_rate"]), "selected_epoch": epoch})
    if len({r["model_id"] for r in records}) != 8:
        raise ValueError("Expected five main DNN and three main graph learning-curve configurations")
    return records, selected_epochs


def seed_source(results, bundle, rows):
    records = []
    labels = np.asarray([r["label"] for r in rows])
    for model_id in bundle["lock"]["expected_model_ids"]:
        meta = bundle["metadata"][model_id]
        if meta["algorithm"] not in NEURAL:
            continue
        seeds = meta["final_fit_metadata"]["seeds"]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Seed stability requires exactly three distinct declared neural seeds")
        with np.load(results / model_id / "predictions.npz", allow_pickle=False) as payload:
            for partition, index, member_key, aggregate_key in (
                ("development_oof", bundle["development_index"], "oof_seed_probability", "development"),
                ("test", bundle["test_index"], "test_seed_probability", "test")):
                members = payload[member_key].astype(float)
                if members.shape != (3, len(index)) or not np.isfinite(members).all() or ((members < 0) | (members > 1)).any():
                    raise ValueError("Invalid individual-seed prediction matrix")
                aggregate = bundle["predictions"][model_id][aggregate_key]
                if not np.allclose(members.mean(axis=0), aggregate, atol=1e-7, rtol=0):
                    raise ValueError("Seed probabilities do not reproduce locked arithmetic-mean predictions")
                for seed, p in zip(seeds, members):
                    records.append({"model_id": model_id, "model_label": model_label(meta), "partition": partition,
                        "seed": int(seed), "prediction": "individual_seed", "role": meta["role"],
                        "ROC_AUC": metric_values(labels[index], p)["ROC_AUC"]})
                records.append({"model_id": model_id, "model_label": model_label(meta), "partition": partition,
                    "seed": None, "prediction": "three_seed_probability_mean", "role": meta["role"],
                    "ROC_AUC": metric_values(labels[index], aggregate)["ROC_AUC"]})
    if len({r["model_id"] for r in records}) != 11:
        raise ValueError("Expected eight DNN (including three count) and three graph seed-stability configurations")
    return records


def runtime_source(results, bundle):
    lock_sha = digest(results / "model_lock.json")
    manifest = verify_table_manifest(results / "benchmarks", "benchmark_manifest.json", lock_sha)
    verify_table_manifest(results / "reports", "report_manifest.json", lock_sha)
    feature = read_csv(results / "benchmarks/feature_benchmark.csv")
    measured = {r["model_id"]: r for r in read_csv(results / "benchmarks/runtime_benchmark.csv")}
    fit = {r["model_id"]: r for r in read_csv(results / "reports/runtime_summary.csv")}
    expected = set(bundle["lock"]["expected_model_ids"])
    if set(measured) != expected or set(fit) != expected or manifest.get("n_models") != len(expected):
        raise ValueError("Runtime diagnostics require all registered locked models")
    records = []
    for model_id in bundle["lock"]["expected_model_ids"]:
        meta, bench, timing = bundle["metadata"][model_id], measured[model_id], fit[model_id]
        record = {"model_id": model_id, "model_label": model_label(meta), "algorithm": meta["algorithm"],
            "representation": meta["representation"], "role": meta["role"], "final_fit_seconds": float(timing["final_fit_seconds"]),
            "batch_mean_ms_per_compound": float(bench["batch_mean_milliseconds_per_compound"]),
            "batch_sd_ms_per_compound": 1000 * float(bench["batch_sd_seconds"]) / int(bench["batch_n_compounds"]),
            "single_mean_ms": float(bench["single_mean_milliseconds"]), "single_sd_ms": float(bench["single_sd_milliseconds"]),
            "single_p95_ms": float(bench["single_p95_milliseconds"]), "batch_repeats": int(bench["batch_n_repeats"]),
            "single_measurements": int(bench["single_total_measurements"]), "model_bytes": int(bench["model_bytes"]),
            "model_load_seconds": float(bench["model_load_seconds"]), "threads": int(bench["threads"])}
        if record["model_bytes"] != meta["model_bytes"] or any(record[name] <= 0 or not np.isfinite(record[name]) for name in ("final_fit_seconds", "batch_mean_ms_per_compound", "single_mean_ms")):
            raise ValueError("Invalid actual model runtime measurements")
        records.append(record)
    for row in feature:
        if float(row["mean_seconds"]) <= 0 or not np.isfinite(float(row["mean_seconds"])):
            raise ValueError("Invalid actual feature runtime measurement")
    return feature, records, read_json(results / "benchmarks/runtime_environment.json")


def style_axes(ax, letter=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.55, length=2.5, labelsize=5.5)
    if letter:
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, weight="bold", fontsize=9)


def draw_hpo(records, model_ids):
    model_ids = [model_id for model_id in model_ids if any(row["model_id"] == model_id for row in records)]
    if not model_ids:
        raise ValueError("Candidate-search diagnostics require verified search records")
    graph_ids = [model_id for model_id in model_ids if any(
        row["model_id"] == model_id and row.get("representation") == "graph" for row in records)]
    conventional_ids = [model_id for model_id in model_ids if model_id not in graph_ids]
    panels = [(ids, width, title) for ids, width, title in (
        (conventional_ids, 3, "A  Conventional/count: original three candidates"),
        (graph_ids, 20, "B  Graph models: baseline + 19 random candidates")) if ids]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.09, 10.8), squeeze=False,
                             gridspec_kw={"height_ratios": [max(4, len(ids)) for ids, _, _ in panels]})
    fig.subplots_adjust(left=0.31, right=0.89, bottom=0.11, top=0.95, hspace=0.35)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    for ax, (ids, width, title) in zip(axes[:, 0], panels):
        matrix = np.full((len(ids), width), np.nan)
        labels, selected = [], []
        for i, model_id in enumerate(ids):
            rows = [r for r in records if r["model_id"] == model_id]
            if {r["candidate"] for r in rows} != set(range(1, width + 1)) or len(rows) != width:
                raise ValueError(f"Search panel lacks its complete candidate scores: {model_id}")
            labels.append(rows[0]["model_label"])
            for row in rows:
                matrix[i, row["candidate"] - 1] = row["mean_inner_ROC_AUC"]
                if row["selected"]:
                    selected.append((i, row["candidate"] - 1))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, j in selected:
            ax.add_patch(Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, facecolor="none", edgecolor="#202020", linewidth=0.7))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=4.4 if width == 20 else 4.7,
                            color="white" if matrix[i, j] > 0.68 else "#202020")
        ax.set(xticks=np.arange(width), xticklabels=np.arange(1, width + 1), yticks=np.arange(len(ids)), yticklabels=labels,
               xlabel="Algorithm-specific candidate number", ylabel="")
        ax.tick_params(labelsize=4.9, length=0)
        ax.set_title(title, fontsize=7, pad=8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
        colorbar.set_label("Mean inner ROC-AUC", fontsize=6)
        colorbar.ax.tick_params(labelsize=5.5)
    fig.text(0.31, 0.02, "Outline: maximum mean inner AUC, first candidate wins exact ties.\nGraph search ranges are newly supplied; baseline is candidate 1.\nFinite candidate search does not establish global parameter optimality.", fontsize=5.4)
    return fig


def draw_learning(records, epochs, model_ids):
    fig, axes = plt.subplots(4, 4, figsize=(9.45, 8.4))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.95, wspace=0.5, hspace=0.88)
    for i, model_id in enumerate(model_ids):
        a, b = axes[i // 2, (i % 2) * 2:(i % 2) * 2 + 2]
        rows = [r for r in records if r["model_id"] == model_id]
        seeds = list(dict.fromkeys(r["seed"] for r in rows))
        for color, seed in zip(SEED_COLORS, seeds):
            selector = [r for r in rows if r["seed"] == seed and r["phase"] == "epoch_selector"]
            refit = [r for r in rows if r["seed"] == seed and r["phase"] == "full_development_refit"]
            a.plot([r["epoch"] for r in selector], [r["selector_monitor_value"] for r in selector], color=color, linewidth=0.85)
            a.axvline(selector[0]["selected_epoch"], color=color, linestyle="--", linewidth=0.55, alpha=0.7)
            b.plot([r["epoch"] for r in refit], [r["training_weighted_BCE"] for r in refit], color=color, linewidth=0.85)
        a.set_ylabel("Validation ROC-AUC" if rows[0]["algorithm"] == "DNN" else "Validation BCE", fontsize=5.7)
        b.set_ylabel("Training weighted BCE", fontsize=5.7)
        for ax, phase in ((a, "Epoch selector"), (b, "Fresh refit")):
            ax.set_title(rows[0]["model_label"] + "\n" + phase, fontsize=6.1, pad=4)
            ax.set_xlabel("Epoch", fontsize=5.7)
            style_axes(ax)
    seeds = list(dict.fromkeys(r["seed"] for r in records))
    fig.legend([Line2D([], [], color=c, linewidth=1) for c in SEED_COLORS], [str(s) for s in seeds],
               title="Recorded seed", loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=6, title_fontsize=6)
    return fig


def draw_seeds(records, model_ids):
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 5.7), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.27, top=0.93, hspace=0.35)
    labels = [next(r["model_label"] for r in records if r["model_id"] == model_id) for model_id in model_ids]
    seeds = list(dict.fromkeys(r["seed"] for r in records if r["seed"] is not None))
    for ax, partition, letter in zip(axes, ("development_oof", "test"), "AB"):
        for i, model_id in enumerate(model_ids):
            group = [r for r in records if r["model_id"] == model_id and r["partition"] == partition]
            for offset, color, seed in zip((-0.13, 0, 0.13), SEED_COLORS, seeds):
                value = next(r["ROC_AUC"] for r in group if r["seed"] == seed)
                ax.scatter(i + offset, value, s=14, color=color, zorder=3)
            ensemble = next(r["ROC_AUC"] for r in group if r["prediction"] == "three_seed_probability_mean")
            ax.scatter(i, ensemble, marker="D", s=15, facecolor="none", edgecolor="#202020", linewidth=0.75, zorder=4)
        ax.set(ylim=(0, 1.02), ylabel="ROC-AUC", xticks=np.arange(len(model_ids)))
        ax.set_title("Development OOF" if partition == "development_oof" else "Held-out test", fontsize=8)
        style_axes(ax, letter)
    axes[-1].set_xticklabels(labels, rotation=48, ha="right", fontsize=5.7)
    handles = [Line2D([], [], marker="o", color=c, linestyle="none", label=str(s)) for c, s in zip(SEED_COLORS, seeds)]
    handles.append(Line2D([], [], marker="D", markerfacecolor="none", color="#202020", linestyle="none", label="Probability-mean ensemble"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995), frameon=False, ncol=4, fontsize=5.5)
    return fig


def draw_runtime(features, records):
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.9))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.2, top=0.94, hspace=0.7, wspace=0.37)
    ax = axes[0, 0]
    for i, record in enumerate(features):
        value = float(record["mean_seconds"])
        ax.barh(i, value, color="#4d89b8", height=0.65)
        sd = record.get("sample_sd_seconds")
        if sd not in (None, ""):
            ax.errorbar(value, i, xerr=float(sd), color="#303030", capsize=1.5, linewidth=0.55)
    ax.set(yticks=np.arange(len(features)), yticklabels=[REP_LABEL[r["representation"]].replace("TopologicalTorsion", "Torsion") for r in features], xlabel="Feature computation (s; log scale)")
    ax.set_xscale("log")
    ax.invert_yaxis()
    style_axes(ax, "A")
    panels = ((axes[0, 1], "final_fit_seconds", None, "Final development fit (s)", "B"),
              (axes[1, 0], "batch_mean_ms_per_compound", "batch_sd_ms_per_compound", "Batch inference (ms / compound)", "C"),
              (axes[1, 1], "single_mean_ms", "single_sd_ms", "Single inference (ms)", "D"))
    reps = list(dict.fromkeys(r["representation"] for r in records))
    offsets = {rep: (-0.28 + 0.56 * i / max(1, len(reps) - 1)) for i, rep in enumerate(reps)}
    for ax, name, error, ylabel, letter in panels:
        for record in records:
            rep = record["representation"]
            base = rep.replace("_count", "")
            color = REP_COLOR[base]
            x = ALG_ORDER.index(record["algorithm"]) + offsets[rep]
            marker = "^" if "count" in rep else ("s" if rep == "graph" else "o")
            face = "none" if "count" in rep else color
            ax.scatter(x, record[name], s=12, marker=marker, facecolor=face, edgecolor=color, linewidth=0.6, zorder=3)
            if error is not None:
                # SD is descriptive repeat variability, never a confidence interval.
                sd = record[error]
                lower = max(record[name] * 0.001, record[name] - sd)
                ax.vlines(x, lower, record[name] + sd, color=color, linewidth=0.5, alpha=0.55)
        ax.set(xticks=np.arange(len(ALG_ORDER)), xticklabels=[ALG_LABEL.get(a, a) for a in ALG_ORDER], ylabel=ylabel)
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelrotation=42)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        style_axes(ax, letter)
    handles = [Line2D([], [], marker="o", color=c, linestyle="none", label=REP_LABEL[rep]) for rep, c in REP_COLOR.items()]
    handles.append(Line2D([], [], marker="^", color="#444444", markerfacecolor="none", linestyle="none", label="Count encoding (family color)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.01), ncol=3, fontsize=5.3, frameon=False)
    return fig


def export(fig, output, stem, caption):
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip() and text.get_fontfamily() != ["Arial"]:
            raise ValueError(f"A diagnostic figure contains non-Arial text: {stem}")
    files = []
    for extension in ("pdf", "svg", "png", "tiff"):
        path = output / f"{stem}.{extension}"
        kwargs = {"dpi": 600, "facecolor": "white"}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        files.append(path)
    plt.close(fig)
    (output / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.results / "model_lock.json").is_file():
        raise ValueError("Complete global model locking is required before drawing training diagnostics")
    rows, _ = load_data()
    assignments = read_json(args.results.parent / "split_payload.json")["assignments"]
    bundle = load_locked_predictions(args.results, rows, assignments)
    protocol = read_json(args.results / "protocol.json")
    if len(bundle["metadata"]) != 43:
        raise ValueError("Training diagnostics require all 43 registered configurations")
    hpo, source_hashes = search_source(args.results, bundle, rows, protocol)
    curves, epochs = learning_curve_source(bundle)
    seeds = seed_source(args.results, bundle, rows)
    features, runtime, environment = runtime_source(args.results, bundle)
    output = (args.output or args.results / "figures/training_diagnostics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    typography = apply_font()
    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 7, "axes.linewidth": 0.6,
                         "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5})
    for name, records in (("hpo_source", hpo), ("learning_curve_source", curves), ("selected_epoch_source", epochs),
                          ("seed_stability_source", seeds), ("feature_runtime_source", features), ("model_runtime_source", runtime)):
        write_csv(output / f"{name}.csv", records)
    write_json(output / "diagnostic_source.json", {"hpo": hpo, "learning_curves": curves, "selected_epochs": epochs,
        "seed_stability": seeds, "feature_runtime": features, "model_runtime": runtime,
        "runtime_environment": environment, "search_input_hashes": source_hashes})
    write_json(output / "figure_contract.json", FIGURE_CONTRACT)
    ids = bundle["lock"]["expected_model_ids"]
    curve_ids = [n for n in ids if any(r["model_id"] == n for r in curves)]
    seed_ids = [n for n in ids if any(r["model_id"] == n for r in seeds)]
    files = export(draw_hpo(hpo, ids), output, "FigureS8_Search",
        "Figure S8. Full-development candidate search using mean ROC-AUC over three structure-group inner folds. A, The 40 conventional/count combinations compare the original three supplementary candidates per algorithm. B, Each graph architecture compares its original Recommended baseline (candidate 1) with 19 unique randomly sampled configurations from the newly supplied search ranges, using seed 20260810. Every candidate is fitted and evaluated; these graph ranges were not present in the original parameter table. Outlines mark maximum mean inner AUC, with first-candidate tie-breaking. Both panels share the same AUC color scale. Complete parameters and individual inner-fold scores are provided in source tables. This is tuning evidence, not an unbiased generalization estimate, and does not use held-out outcomes.")
    files += export(draw_learning(curves, epochs, curve_ids), output, "FigureS9_Stopping",
        "Figure S9. Final-fit learning diagnostics for five main DNN representations and three graph architectures. In each model pair, the left panel shows the recorded training-internal grouped epoch-selector monitor: validation ROC-AUC for DNN, unweighted validation BCE for graph models. Dashed lines mark each seed's selected epoch. Right panels show training-weighted BCE during fresh initialization and fixed-epoch refitting on all development records. Curve lengths can differ by seed. Colors identify the three recorded seeds; no outer validation or held-out outcome selects an epoch. Source tables preserve selector training loss as well as the plotted monitor and refit history.")
    files += export(draw_seeds(seeds, seed_ids), output, "FigureS10_Seeds",
        "Figure S10. Individual-seed ROC-AUC for all eleven neural configurations (five main DNN, three count-DNN sensitivity models and three graph models), on development OOF predictions and the internal held-out test. Open diamonds show the arithmetic mean of the three positive probabilities before calculating AUC. Points describe seed-specific fitting/internal epoch-selection variation; they are not confidence intervals or independent validation cohorts. Seed outcomes do not retrospectively change the registered ensemble or model selection.")
    files += export(draw_runtime(features, runtime), output, "FigureS11_Efficiency",
        "Figure S11. Actual measured computational cost. A, Mean wall time to rebuild each representation from all current source SMILES (three repeats for binary/descriptors/graphs; one for counts); available error bars show sample SD and are omitted for one-repeat counts. B, Wall time of the final all-development fitting procedure, including training-internal epoch selection and three-seed aggregation where applicable; hyperparameter-search time is excluded. C, Mean batch inference latency per compound over three fixed 335-record batches. D, Single-compound latency over ten repeats on each of ten fixed development examples. C–D whiskers show measured SD, not CI; lower endpoints are truncated above zero for logarithmic display. Inference uses loaded models with numerical features/cached graphs, including fitted preprocessing and seed probability aggregation. Hardware, thread limits, workload caveat, model size and load timings are supplied in source data; no platform-independent speed claim is made.")
    write_json(output / "training_diagnostics_manifest.json", {"schema_version": 1, "status": "rendered; visual review required",
        "typography": typography, "source_excel_sha256": bundle["lock"]["source_excel_sha256"],
        "global_lock_sha256": digest(args.results / "model_lock.json"), "protocol_sha256": bundle["lock"]["protocol_sha256"],
        "n_configurations": len(ids), "n_main_neural_learning_configurations": len(curve_ids), "n_neural_seed_configurations": len(seed_ids),
        "all_text_Arial_verified": True, "files": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "training_diagnostics_manifest.json"}})
    print(json.dumps({"status": "complete", "figures": 4, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
