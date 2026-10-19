"""Draw locked conventional/graph model results in the supplied figure layouts.

Only predictions from the current DATA and hash-verified global model lock are
accepted. Missing training results cause a clear failure; no curves are reused.
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
from matplotlib import colors
from matplotlib.text import Text
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.data import load_data
from nephrotox.modeling_reporting import HEATMAP_METRICS, load_locked_predictions, metric_values
from figure_style import apply_font

REPRESENTATIONS = ("morgan2048", "maccs166", "atompair1024", "torsion1024", "rdkit166")
CM_ORDER = ("torsion1024", "morgan2048", "maccs166", "atompair1024", "rdkit166")
LABEL = {"morgan2048": "Morgan", "maccs166": "MACCS", "atompair1024": "AtomPair",
         "torsion1024": "TopologicalTorsion", "rdkit166": "Descriptor"}
COLOR = dict(zip(REPRESENTATIONS, ("#ef4e3b", "#367faf", "#f5bf45", "#54b58c", "#847bab")))
MARKER = dict(zip(REPRESENTATIONS, ("s", "o", "^", "v", "D")))
ALG_LABEL = {"XGB": "XGBoost", "DMPNN": "D-MPNN", "AttentiveFP": "AttentiveFP"}
GRAPH_ORDER = ("DMPNN", "GINE", "AttentiveFP")
GRAPH_COLOR = {"DMPNN": "#ed4f3b", "GINE": "#46b6d0", "AttentiveFP": "#405d92"}
METRIC_LABEL = {"ROC_AUC": "AUC", "Precision": "P", "AP": "Average precision"}

# Contracts precede all plotting. Claims are descriptive, never preset rankings.
FIGURE_CONTRACT = {
    "Figure3": {"claim": "Describe held-out discrimination and class-specific errors for all 25 conventional configurations.",
                "evidence": "Five classifier rows: ROC for all representations, then five row-normalized confusion matrices.",
                "archetype": "quantitative grid", "size_mm": [330, 246.3]},
    "Figure4": {"claim": "Compare held-out metrics with uncertainty without choosing the model on the test set.",
                "evidence": "Six metric comparisons followed by a complete 25 by 8 metric heatmap.",
                "archetype": "quantitative grid", "size_mm": [180, 197]},
    "Figure5": {"claim": "Compare graph models using development OOF and held-out predictions with paired AUC inference.",
                "evidence": "OOF ROC, test ROC/PR, metric heatmap, paired metric differences and three confusion matrices.",
                "archetype": "quantitative grid", "size_mm": [180, 230]},
    "exports": ["PDF", "SVG editable text", "PNG 600 dpi", "TIFF 600 dpi LZW"],
    "integrity": "Current DATA, fixed structure groups, probability threshold 0.5, class-stratified bootstrap 95% CI; reference-negative label 0.",
    "review_risks": ["Internal held-out set is not an external validation cohort", "No graph superiority assumed", "No clinical safety or causal claim"],
}


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _format_axes(ax, font_size=6.0):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, labelsize=font_size)
    ax.xaxis.label.set_size(font_size + 0.2)
    ax.yaxis.label.set_size(font_size + 0.2)


def _panel(ax, text, x=-0.16, y=1.05):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, weight="bold", va="bottom", ha="left")


def _roc(ax, y, prediction, label, color, linewidth=1.05):
    fpr, tpr, _ = roc_curve(y, prediction)
    auc = metric_values(y, prediction)["ROC_AUC"]
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=linewidth)


def _roc_axes(ax):
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=0.65, zorder=0)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    _format_axes(ax)


def _confusion(ax, y, probability, title, font_size=6.0, colorbar=True):
    metrics = metric_values(y, probability)
    count = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    denominator = count.sum(axis=1, keepdims=True)
    normalized = np.divide(count, denominator, out=np.zeros((2, 2)), where=denominator != 0)
    image = ax.imshow(normalized, cmap="Blues", norm=colors.Normalize(vmin=0, vmax=1), aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{100 * normalized[i, j]:.1f}%\n(n={count[i, j]})", ha="center", va="center",
                    fontsize=font_size - 0.3, color="white" if normalized[i, j] > 0.55 else "#1c1c1c")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Reference-\nnegative", "Positive"],
           yticklabels=["Reference-\nnegative", "Positive"], xlabel="Predicted class", ylabel="True class")
    ax.set_title(title, fontsize=font_size + 0.6, pad=5)
    ax.tick_params(labelsize=font_size - 0.4, length=2)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
        bar.ax.tick_params(labelsize=font_size - 1, length=1.5, width=0.4)
        bar.outline.set_linewidth(0.5)
    return count


def _heatmap(ax, matrix, row_labels, column_labels, *, difference=False, annotate=True, fontsize=6.0):
    matrix = np.asarray(matrix, dtype=float)
    if difference:
        limit = max(0.03, float(np.max(np.abs(matrix))))
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    else:
        # Fixed probability/metric scale, including a negative MCC if observed.
        lower = min(0.0, float(matrix.min()))
        norm = colors.Normalize(vmin=lower, vmax=1)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set(xticks=np.arange(len(column_labels)), yticks=np.arange(len(row_labels)),
           xticklabels=column_labels, yticklabels=row_labels)
    ax.tick_params(labelsize=fontsize, length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                label = f"{value:+.3f}" if difference else f"{value:.3f}"
                dark = abs(norm(value) - 0.5) > 0.32
                ax.text(j, i, label, ha="center", va="center", color="white" if dark else "#202020", fontsize=fontsize - 0.6)
    return image


def _exports(fig, folder, name):
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip() and text.get_fontfamily() != ["Arial"]:
            raise ValueError(f"Non-Arial text found in {name}")
    paths = []
    for extension in ("pdf", "svg", "png", "tiff"):
        path = folder / f"{name}.{extension}"
        kwargs = {"dpi": 600, "facecolor": "white"}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _model_lookup(metadata):
    lookup = {}
    for model_id, meta in metadata.items():
        if meta["role"] == "main":
            key = (meta["representation"], meta["algorithm"])
            if key in lookup:
                raise ValueError(f"Duplicate main model configuration: {key}")
            lookup[key] = model_id
    missing = [(representation, algorithm) for representation in REPRESENTATIONS
               for algorithm in ("RF", "SVM", "MLP", "XGB", "DNN") if (representation, algorithm) not in lookup]
    missing += [("graph", algorithm) for algorithm in GRAPH_ORDER if ("graph", algorithm) not in lookup]
    if missing:
        raise ValueError(f"Main figure inputs are incomplete: {missing}")
    return lookup


def draw_figure3(bundle, labels, lookup, output):
    """The original five-by-six grid, using current test predictions only."""
    width, height = FIGURE_CONTRACT["Figure3"]["size_mm"]
    fig, axes = plt.subplots(5, 6, figsize=(width / 25.4, height / 25.4),
                             gridspec_kw={"width_ratios": [1.2, 1, 1, 1, 1, 1]})
    fig.subplots_adjust(left=0.045, right=0.98, bottom=0.045, top=0.97, wspace=0.67, hspace=0.66)
    y = labels[bundle["test_index"]]
    for row_index, algorithm in enumerate(("DNN", "XGB", "MLP", "RF", "SVM")):
        roc_ax = axes[row_index, 0]
        for representation in REPRESENTATIONS:
            model = lookup[(representation, algorithm)]
            _roc(roc_ax, y, bundle["predictions"][model]["test"], LABEL[representation], COLOR[representation], 0.85)
        _roc_axes(roc_ax)
        roc_ax.set_title(f"Held-out ROC: {ALG_LABEL.get(algorithm, algorithm)}", fontsize=7.1, pad=5)
        roc_ax.legend(loc="lower right", fontsize=4.4, frameon=False, handlelength=1.6, labelspacing=0.28)
        for column_index, representation in enumerate(CM_ORDER, 1):
            model = lookup[(representation, algorithm)]
            _confusion(axes[row_index, column_index], y, bundle["predictions"][model]["test"],
                       f"{LABEL[representation]}–{ALG_LABEL.get(algorithm, algorithm)}", font_size=5.3)
    return _exports(fig, output, "Figure3")


def draw_figure4(bundle, summaries, lookup, output):
    """Six metric comparisons with CIs and a complete conventional heatmap."""
    width, height = FIGURE_CONTRACT["Figure4"]["size_mm"]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.65], left=0.17, right=0.965,
                           bottom=0.075, top=0.97, hspace=0.66, wspace=0.44)
    algorithms = ("RF", "SVM", "MLP", "XGB", "DNN")
    top_metrics = ("ROC_AUC", "Precision", "F1", "ACC", "SE", "SP")
    for index, metric in enumerate(top_metrics):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        for representation in REPRESENTATIONS:
            records = [summaries[lookup[(representation, algorithm)]] for algorithm in algorithms]
            values = np.asarray([float(r[metric]) for r in records])
            lower = np.asarray([float(r[metric + "_lower"]) for r in records])
            upper = np.asarray([float(r[metric + "_upper"]) for r in records])
            # A percentile interval need not contain the observed estimate.
            # Draw intervals directly instead of clipping erroneous yerr values.
            x = np.arange(5)
            ax.vlines(x, lower, upper, color=COLOR[representation], alpha=0.35, linewidth=0.55)
            ax.plot(x, values, color=COLOR[representation], marker=MARKER[representation], markersize=3.1,
                    linewidth=0.75, label=LABEL[representation])
        ax.set(xticks=np.arange(5), xticklabels=[ALG_LABEL.get(a, a) for a in algorithms],
               ylabel=METRIC_LABEL.get(metric, metric), xlabel="Models")
        _format_axes(ax, 5.4)
        ax.tick_params(axis="x", labelrotation=15)
        _panel(ax, chr(ord("A") + index), x=-0.31)
        if index in (0, 1, 2):
            ax.legend(fontsize=4.1, loc="upper left", frameon=False, labelspacing=0.15, handlelength=1.5)
    heat_ax = fig.add_subplot(grid[2, :])
    keys = [(representation, algorithm) for algorithm in ("SVM", "RF", "XGB", "MLP", "DNN")
            for representation in reversed(REPRESENTATIONS)]
    matrix = [[float(summaries[lookup[key]][metric]) for metric in HEATMAP_METRICS] for key in keys]
    labels = [f"{LABEL[r]}–{ALG_LABEL.get(a, a)}" for r, a in keys]
    image = _heatmap(heat_ax, matrix, labels, [METRIC_LABEL.get(m, m) for m in HEATMAP_METRICS], fontsize=4.9)
    heat_ax.set_title("Held-out model metrics", fontsize=7.2, weight="bold", pad=6)
    heat_ax.set_xlabel("Metrics", fontsize=6.2)
    _panel(heat_ax, "G", x=-0.21)
    bar = fig.colorbar(image, ax=heat_ax, fraction=0.015, pad=0.022, ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.ax.tick_params(labelsize=5, width=0.5, length=2)
    bar.outline.set_linewidth(0.5)
    return _exports(fig, output, "Figure4")


def draw_figure5(bundle, labels, summaries, paired, lookup, output):
    """Graph comparison, retaining the original ROC/PR/heatmap/CM panel map."""
    width, height = FIGURE_CONTRACT["Figure5"]["size_mm"]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    grid = fig.add_gridspec(4, 6, height_ratios=[1, 1, 0.64, 0.95],
                           left=0.12, right=0.975, bottom=0.08, top=0.97, hspace=0.73, wspace=1.45)
    axes = {"A": fig.add_subplot(grid[0, :3]), "B": fig.add_subplot(grid[0, 3:]),
            "C": fig.add_subplot(grid[1, :3]), "D": fig.add_subplot(grid[1, 3:]),
            "E": fig.add_subplot(grid[2, :]), "F": fig.add_subplot(grid[3, :2]),
            "G": fig.add_subplot(grid[3, 2:4]), "H": fig.add_subplot(grid[3, 4:])}
    dev_y, test_y = labels[bundle["development_index"]], labels[bundle["test_index"]]
    graph_ids = [lookup[("graph", algorithm)] for algorithm in GRAPH_ORDER]
    for algorithm, model_id in zip(GRAPH_ORDER, graph_ids):
        predictions = bundle["predictions"][model_id]
        label = ALG_LABEL.get(algorithm, algorithm)
        _roc(axes["A"], dev_y, predictions["development"], label, GRAPH_COLOR[algorithm])
        _roc(axes["B"], test_y, predictions["test"], label, GRAPH_COLOR[algorithm])
        precision, recall, _ = precision_recall_curve(test_y, predictions["test"])
        ap = metric_values(test_y, predictions["test"])["AP"]
        axes["C"].plot(recall, precision, color=GRAPH_COLOR[algorithm], linewidth=1.05, label=f"{label} (AP = {ap:.3f})")
    for key, title in (("A", "Development OOF ROC"), ("B", "Held-out test ROC")):
        _roc_axes(axes[key])
        axes[key].set_title(title, fontsize=7.6, weight="bold", pad=5)
        axes[key].legend(loc="lower right", fontsize=5.3, frameon=False, labelspacing=0.25)
    axes["C"].axhline(float(test_y.mean()), linestyle="--", color="#999999", linewidth=0.7)
    axes["C"].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    axes["C"].set_xticks(np.linspace(0, 1, 6))
    axes["C"].set_yticks(np.linspace(0, 1, 6))
    axes["C"].set_title("Held-out precision–recall", fontsize=7.6, weight="bold", pad=5)
    axes["C"].legend(loc="lower left", fontsize=5.3, frameon=False, labelspacing=0.25)
    _format_axes(axes["C"])
    matrix = np.asarray([[float(summaries[model_id][metric]) for metric in HEATMAP_METRICS] for model_id in graph_ids])
    _heatmap(axes["D"], matrix, [ALG_LABEL.get(a, a) for a in GRAPH_ORDER],
             [METRIC_LABEL.get(m, m) for m in HEATMAP_METRICS], fontsize=5.7)
    axes["D"].set_title("Held-out test performance", fontsize=7.6, weight="bold", pad=5)
    differences, difference_labels = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        differences.append(matrix[i] - matrix[j])
        difference_labels.append(f"{ALG_LABEL.get(GRAPH_ORDER[i], GRAPH_ORDER[i])}–{ALG_LABEL.get(GRAPH_ORDER[j], GRAPH_ORDER[j])}")
    _heatmap(axes["E"], differences, difference_labels, [METRIC_LABEL.get(m, m) for m in HEATMAP_METRICS],
             difference=True, fontsize=5.6)
    axes["E"].set_title("Metric difference (row model 1 − model 2)", fontsize=7.3, weight="bold", pad=6)
    # Formal testing was prespecified for AUC only. Do not apply uncorrected
    # significance labels to seven exploratory metric differences.
    for row_index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        matches = [r for r in paired if r["model_1"] == graph_ids[i] and r["model_2"] == graph_ids[j]]
        if len(matches) != 1:
            raise ValueError("The prespecified graph AUC pairwise report is incomplete")
        if float(matches[0]["p_holm"]) < 0.05:
            axes["E"].text(0, row_index + 0.24, "*", weight="bold", ha="center", va="center", fontsize=6.0)
    axes["E"].text(0, -0.25, "* AUC: Holm-adjusted p < 0.05; other metric differences are descriptive.",
                    transform=axes["E"].transAxes, fontsize=4.9, ha="left", va="top", color="#555555")
    for panel, algorithm, model_id in zip(("F", "G", "H"), GRAPH_ORDER, graph_ids):
        ax = axes[panel]
        _confusion(ax, test_y, bundle["predictions"][model_id]["test"], ALG_LABEL.get(algorithm, algorithm),
                   font_size=6.0, colorbar=False)
        ax.title.set_color(GRAPH_COLOR[algorithm])
        ax.title.set_weight("bold")
    for letter, ax in axes.items():
        _panel(ax, letter, x=-0.18 if letter != "E" else -0.08)
    return _exports(fig, output, "Figure5")


def draw_supplementary(bundle, labels, summaries, lookup, report_folder, output):
    """Current conventional/graph calibration and binary-versus-count sensitivity."""
    generated = []
    bins = _csv(report_folder / "calibration_bins.csv")
    primary = json.loads((report_folder / "primary_selection.json").read_text(encoding="utf-8"))["primary_model_id"]
    selected = [primary] + [lookup[("graph", algorithm)] for algorithm in GRAPH_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.0))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.94, hspace=0.44, wspace=0.31)
    for panel, model_id in enumerate(selected):
        ax = axes.flat[panel]
        meta = bundle["metadata"][model_id]
        for partition, color, label in (("development_oof", "#367faf", "Development OOF"), ("test", "#ed793c", "Held-out test")):
            records = [r for r in bins if r["model_id"] == model_id and r["partition"] == partition and int(r["n"]) > 0]
            ax.plot([float(r["mean_probability"]) for r in records], [float(r["positive_fraction"]) for r in records],
                    marker="o", markersize=3, linewidth=0.9, color=color, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=0.7)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction")
        title = ALG_LABEL.get(meta["algorithm"], meta["algorithm"])
        if meta["representation"] != "graph":
            title = LABEL[meta["representation"]] + "–" + title
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        _format_axes(ax)
        _panel(ax, chr(ord("A") + panel), x=-0.2)
    generated += _exports(fig, output, "FigureS_calibration")
    count_rep = {"morgan2048": "morgan_count2048", "atompair1024": "atompair_count1024", "torsion1024": "torsion_count1024"}
    count_lookup = {(m["representation"], m["algorithm"]): name for name, m in bundle["metadata"].items() if m["role"] == "sensitivity_count"}
    missing = [(count, algorithm) for count in count_rep.values() for algorithm in ("RF", "SVM", "MLP", "XGB", "DNN") if (count, algorithm) not in count_lookup]
    if missing:
        raise ValueError(f"Count-fingerprint sensitivity inputs are incomplete: {missing}")
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.6))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.84, wspace=0.36)
    source = []
    for panel, (representation, count) in enumerate(count_rep.items()):
        ax = axes[panel]
        algorithms = ("RF", "SVM", "MLP", "XGB", "DNN")
        for key, color, label in ((representation, COLOR[representation], "Binary"), (count, "#4d4d4d", "Count")):
            values = []
            for algorithm in algorithms:
                model_id = lookup[(key, algorithm)] if label == "Binary" else count_lookup[(key, algorithm)]
                values.append(float(summaries[model_id]["ROC_AUC"]))
                source.append({"representation_family": representation, "encoding": label, "algorithm": algorithm,
                               "model_id": model_id, "heldout_ROC_AUC": values[-1],
                               "development_fold_mean_ROC_AUC": float(summaries[model_id]["development_fold_mean_ROC_AUC"])})
            ax.plot(np.arange(5), values, marker="o", markersize=3.5, linewidth=0.9, color=color, label=label)
        ax.set(xticks=np.arange(5), xticklabels=[ALG_LABEL.get(a, a) for a in algorithms], ylabel="Held-out ROC-AUC")
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(LABEL[representation], fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        _format_axes(ax)
        _panel(ax, chr(ord("A") + panel), x=-0.25)
    generated += _exports(fig, output, "FigureS_binary_count")
    with (output / "binary_count_source.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    return generated


def write_figure_sources(bundle, labels, summaries, paired, output):
    """Export the plotted coordinates, confusion counts and metric uncertainty."""
    records = {}
    for model_id, metadata in bundle["metadata"].items():
        if metadata["role"] != "main":
            continue
        partitions = {}
        for partition, index in (("development", bundle["development_index"]), ("test", bundle["test_index"])):
            p, y = bundle["predictions"][model_id][partition], labels[index]
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            values = metric_values(y, p)
            confusion = np.array([[values["TN"], values["FP"]], [values["FN"], values["TP"]]])
            partitions[partition] = {"roc_false_positive_rate": fpr.tolist(), "roc_true_positive_rate": tpr.tolist(),
                "precision": precision.tolist(), "recall": recall.tolist(), "average_precision": values["AP"],
                "confusion_counts": confusion.tolist(), "confusion_true_row_fraction": (confusion / confusion.sum(axis=1, keepdims=True)).tolist(),
                "metrics": values}
        records[model_id] = {"algorithm": metadata["algorithm"], "representation": metadata["representation"],
                             "partitions": partitions,
                             "test_metrics_and_95_ci": {k: float(v) for k, v in summaries[model_id].items()
                                                         if k in HEATMAP_METRICS or k.endswith("_lower") or k.endswith("_upper")}}
    _json(output / "figure_source_data.json", {"threshold": 0.5, "models": records,
          "paired_auc_comparisons": paired,
          "confusion_order": ["reference-negative", "positive"],
          "source": "Current DATA and the globally locked saved predictions; no old figure numbers are reused"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/modeling/outputs")
    parser.add_argument("--supplementary", action="store_true", help="Also render calibration and binary/count sensitivity figures")
    args = parser.parse_args()
    report_folder = args.results / "reports"
    manifest_file = report_folder / "report_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("Complete locked-model reports are required before drawing; run report_predictions after all training is locked")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The locked-model report is incomplete")
    for filename, digest in manifest["files"].items():
        if _digest(report_folder / filename) != digest:
            raise ValueError(f"Model report changed after validation: {filename}")
    if _digest(args.results / "model_lock.json") != manifest["global_lock_sha256"]:
        raise ValueError("The global lock changed after model reporting")
    rows, _ = load_data()
    assignments = json.loads((ROOT / "results/split_payload.json").read_text(encoding="utf-8"))["assignments"]
    bundle = load_locked_predictions(args.results, rows, assignments)
    lookup = _model_lookup(bundle["metadata"])
    labels = np.asarray([r["label"] for r in rows])
    summaries = {r["model_id"]: r for r in _csv(report_folder / "model_summary.csv") if r["partition"] == "test"}
    paired = _csv(report_folder / "paired_auc.csv")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.65, "axes.titlesize": 7.5,
                         "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 5.5, "savefig.facecolor": "white"})
    typography = apply_font()
    _json(output / "figure_contract.json", FIGURE_CONTRACT)
    write_figure_sources(bundle, labels, summaries, paired, output)
    paths = draw_figure3(bundle, labels, lookup, output)
    paths += draw_figure4(bundle, summaries, lookup, output)
    paths += draw_figure5(bundle, labels, summaries, paired, lookup, output)
    if args.supplementary:
        paths += draw_supplementary(bundle, labels, summaries, lookup, report_folder, output)
    _json(output / "figure_manifest.json", {"status": "complete", "typography": typography,
          "source_excel_sha256": manifest["source_excel_sha256"], "source_rows_fingerprint": manifest["source_rows_fingerprint"],
          "protocol_sha256": manifest["protocol_sha256"], "global_lock_sha256": manifest["global_lock_sha256"],
          "report_manifest_sha256": _digest(manifest_file), "threshold": 0.5,
          "all_matplotlib_text_Arial_verified": True, "visual_review_status": "rendered; review the actual result layouts before submission",
          "source_data_sha256": _digest(output / "figure_source_data.json"),
          "figures": {p.name: _digest(p) for p in paths}})
    print(json.dumps({"status": "complete", "output": str(output), "files": len(paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
