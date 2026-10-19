"""Render post-lock supplementary reliability figures in the supplied layout.

All figures use the shared Arial configuration. Only newly generated locked
prediction/source tables are accepted; no historical numerical result is read.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from figure_style import FONT_FAMILY, apply_font, font_path
from nephrotox.modeling_reliability import read_csv, write_json

GRAPH_COLORS = ("#F04B35", "#45BCD6", "#3C568D")
GRAPH_MARKERS = ("o", "^", "D")
BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#9a9a9a"
WIDTH = 171 / 25.4
ALGORITHM_LABELS = {"rf": "RF", "svm": "SVM", "mlp": "MLP", "xgb": "XGBoost", "dnn": "DNN",
                    "dmpnn": "D-MPNN", "d_mpnn": "D-MPNN", "gine": "GINE", "attentivefp": "AttentiveFP"}
REP_LABELS = {"morgan2048": "Morgan", "maccs166": "MACCS", "atompair1024": "AtomPair",
              "torsion1024": "Torsion", "rdkit166": "Descriptor", "graph": ""}


def number(value):
    return float(value) if value not in (None, "", "None") else np.nan


def truth(value):
    return value is True or str(value).lower() == "true"


def panel(ax, letter, title):
    ax.text(-.14, 1.055, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=7, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, width=.65, length=3)


def molecule_image(smiles, highlight_smarts=None, size=(1100, 700)):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Cannot draw invalid source structure")
    drawing = rdMolDraw2D.MolDraw2DCairo(*size)
    drawing.drawOptions().fontFile = str(font_path())
    drawing.drawOptions().baseFontSize = .7
    highlights = []
    if highlight_smarts and highlight_smarts != "?":
        pattern = Chem.MolFromSmarts(highlight_smarts)
        if pattern is not None:
            highlights = sorted({i for match in molecule.GetSubstructMatches(pattern) for i in match})
    rdMolDraw2D.PrepareAndDrawMolecule(drawing, molecule, highlightAtoms=highlights)
    drawing.FinishDrawing()
    return Image.open(io.BytesIO(drawing.GetDrawingText())).convert("RGB")


def figure_contract(out, manifest):
    text = """# Supplementary reliability figure contract

Core claim: characterize uncertainty, structural coverage and exploratory associations of the currently locked models, without changing model selection or test membership.
Backend: Python / matplotlib only. Quantitative grids retain the supplied S1 (2x3), S2/S3/S4 (2x2), and S5 (forest plus structures) panel arrangement.
S1: conventional structural coverage, decision-threshold distance, stratified performance and error.
S2: graph-model OOF ROC, PR, calibration and ten outer group-fold metrics; points are seed-mean folds, not independent seed replicates.
S3: graph-model structural coverage and domain-stratified performance/error.
S4: current discordant-label pairs, source-labelled predictions, descriptive pair summaries and separation; endpoints are deduplicated for compound-level metrics.
S5: development-only association of all 166 prespecified MACCS keys, Fisher tests with BH; effects are exploratory and not kidney-specific toxicophores. Corrected OR Wald intervals do not model within-scaffold dependence.
Count supplement: binary versus count sensitivity analysis on the same locked split, without test-driven primary selection.
Calibration supplement: OOF and heldout reliability curves for the development-selected conventional primary and representation comparator.
AD thresholds: training-only leave-one-out 5th percentiles; each OOF fold has a distinct outer-training reference/threshold. Error bars are structure-group percentile bootstrap 95% intervals conditional on saved predictions, without retraining.
Typography: Arial for all text, including chemical atom glyphs via RDKit fontFile. No font fallback.
Export: 171 mm width; PDF and editable-text SVG; TIFF and PNG at 600 dpi; compact PNG preview. White background, black axes, supplied model colors and consistent legends. No numerical graphic is generated without complete locked current predictions.
"""
    (out / "FIGURE_CONTRACT.md").write_text(text, encoding="utf-8")


def export(fig, out, name, caption):
    fig.canvas.draw()
    fonts = {tuple(t.get_fontfamily()) for t in fig.findobj(Text) if t.get_visible() and t.get_text()}
    if any(FONT_FAMILY not in family for family in fonts):
        raise ValueError(f"Non-Arial text found in {name}")
    for extension in ("pdf", "svg", "png", "tiff"):
        kwargs = {"dpi": 600, "facecolor": "white"}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out / f"{name}.{extension}", **kwargs)
    with Image.open(out / f"{name}.png") as image:
        image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        image.save(out / f"{name}_preview.png")
    (out / f"{name}_caption.txt").write_text(caption, encoding="utf-8")
    write_json(out / f"{name}_QA.json", {"status": "rendered", "font_family": FONT_FAMILY,
        "all_matplotlib_text_Arial": True, "chemical_atom_font": FONT_FAMILY,
        "width_mm": float(fig.get_figwidth() * 25.4), "height_mm": float(fig.get_figheight() * 25.4),
        "raster_dpi": 600, "vector_text_editable": True,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob(name + ".*")}})
    plt.close(fig)


class Source:
    def __init__(self, results, reliability):
        self.results, self.reliability = Path(results), Path(reliability)
        self.manifest = json.loads((self.reliability / "reliability_manifest.json").read_text(encoding="utf-8"))
        report_manifest = json.loads((self.results / "reports/report_manifest.json").read_text(encoding="utf-8"))
        if self.manifest["source_rows_fingerprint"] != report_manifest["source_rows_fingerprint"] or self.manifest["global_lock_sha256"] != report_manifest["global_lock_sha256"]:
            raise ValueError("Reliability reports and model reports do not share the same current lock")
        for name, digest in self.manifest["files"].items():
            if hashlib.sha256((self.reliability / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Reliability source table changed: {name}")
        self.ad = read_csv(self.reliability / "applicability_domain.csv")
        self.thresholds = read_csv(self.reliability / "ad_thresholds.csv")
        self.loo = read_csv(self.reliability / "development_loo_neighbors.csv")
        self.performance = read_csv(self.reliability / "ad_performance.csv")
        self.calibration = read_csv(self.reliability / "calibration_bins.csv")
        self.fold = read_csv(self.reliability / "perfold_metrics.csv")
        self.predictions = {"test": read_csv(self.results / "reports/test_predictions.csv"),
                            "development_oof": read_csv(self.results / "reports/oof_predictions.csv")}
        self.identity = {r["compound_id"]: r for r in read_csv(self.reliability / "compound_identity.csv")}
        self.metadata = self.manifest["model_metadata"]
        order = {"dmpnn": 0, "gine": 1, "attentivefp": 2}
        self.graphs = sorted(self.manifest["graph_model_ids"], key=lambda k: order.get(
            self.metadata[k]["algorithm"].lower().replace("-", "").replace("_", ""), 99))
        self.primary = self.manifest["primary_model_id"]
        self.comparison = self.manifest["comparison_model_id"]

    def label(self, model_id):
        m = self.metadata[model_id]
        algo = ALGORITHM_LABELS.get(m["algorithm"].lower(), m["algorithm"])
        return "-".join(v for v in (REP_LABELS.get(m["representation"], m["representation"]), algo) if v)

    def prediction(self, model_id, partition):
        rows = [r for r in self.predictions[partition] if r["model_id"] == model_id]
        return rows, np.asarray([int(r["label"]) for r in rows]), np.asarray([float(r["probability"]) for r in rows])

    def domain_performance(self, model_id, domain, stratum, partition="test"):
        rows = [r for r in self.performance if r["model_id"] == model_id and r["domain"] == domain and
                r["stratum"] == stratum and r["partition"] == partition]
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous AD summary")
        return rows[0]


def s1(source, out):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, WIDTH * .56))
    fig.subplots_adjust(left=.075, right=.985, top=.89, bottom=.13, wspace=.61, hspace=.78)
    test_ad = [r for r in source.ad if r["partition"] == "test"]
    index = {r["compound_id"]: r for r in test_ad}
    threshold = number(next(r for r in source.thresholds if r["partition"] == "test")["max_threshold"])
    ax = axes[0, 0]
    for label, color, title in ((1, BLUE, "Positive"), (0, ORANGE, "Reference-negative")):
        vals = [number(r["max_tanimoto"]) for r in test_ad if int(source.identity[r["compound_id"]]["label"]) == label]
        ax.hist(vals, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1, color=color, label=title)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=.8, label=f"AD={threshold:.3f}")
    ax.set(xlabel="Maximum Morgan Tanimoto", ylabel="Test compounds")
    ax.legend(fontsize=4.8, frameon=False)
    panel(ax, "A", "Structural coverage")
    models = [source.primary, source.comparison]
    for col, model_id in enumerate(models, 1):
        ax = axes[0, col]
        pred, y, p = source.prediction(model_id, "test")
        similarity = [number(index[r["compound_id"]]["max_tanimoto"]) for r in pred]
        correct = (p >= .5) == y
        ax.scatter(similarity, 2 * np.abs(p - .5), c=np.where(correct, "#377EB8", "#E74C3C"), s=5, alpha=.7, linewidths=0)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=.7)
        ax.set(xlabel="Maximum Morgan Tanimoto", ylabel="Threshold distance\n2|p - 0.5|", ylim=(-.02, 1.02))
        ax.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=2.5, color=c, label=l)
                           for c, l in (("#377EB8", "Correct"), ("#E74C3C", "Incorrect"))], fontsize=4.8, loc="lower right", frameon=False)
        panel(ax, chr(65 + col), source.label(model_id))
    for col, model_id in enumerate(models):
        ax = axes[1, col]
        bins = [f"similarity_{a:.2f}_{b:.2f}" for a, b in zip((0, .25, .5, .75), (.25, .5, .75, 1))]
        records = [source.domain_performance(model_id, "max", name) for name in bins]
        for metric, color, marker in (("ROC_AUC", "#E74C3C", "s"), ("BA", "#377EB8", "o")):
            ax.plot(np.arange(4), [number(r.get(metric)) for r in records], color=color, marker=marker, markersize=2.4, linewidth=.9, label=metric.replace("ROC_", ""))
        for x, row in enumerate(records):
            ax.text(x, .02, f"n={row['n']}", transform=ax.get_xaxis_transform(), fontsize=4.7, ha="center")
        ax.set_xticks(range(4), ["0-.25", ".25-.5", ".5-.75", ".75-1"], rotation=15)
        ax.set(xlabel="Maximum Tanimoto bin", ylabel="Performance", ylim=(0, 1.03))
        ax.legend(fontsize=4.8, frameon=False, loc="lower right")
        panel(ax, chr(68 + col), source.label(model_id))
    ax = axes[1, 2]
    for offset, model_id in enumerate(models):
        records = [source.domain_performance(model_id, "max", domain) for domain in ("outside", "inside")]
        values = [number(r.get("error_rate")) for r in records]
        positions = np.arange(2) + (offset - .5) * .34
        ax.bar(positions, values, width=.34, color=("#377EB8", "#7B6FA6")[offset], alpha=.85, label=source.label(model_id))
        for x, value, row in zip(positions, values, records):
            if np.isfinite(value):
                low, high = number(row.get("error_rate_lower")), number(row.get("error_rate_upper"))
                if np.isfinite(low + high):
                    ax.vlines(x, low, high, color="black", linewidth=.65)
                    ax.hlines([low, high], x - .025, x + .025, color="black", linewidth=.65)
                ax.text(x, value + .015, f"{value:.2f}\nn={row['n']}", ha="center", fontsize=4.6)
    ax.set_xticks(range(2), ["Outside AD", "Inside AD"])
    ax.set(ylabel="Classification error rate", ylim=(0, 1))
    ax.legend(fontsize=4.5, frameon=False)
    panel(ax, "F", "Domain-stratified errors")
    export(fig, out, "FigureS1", "Figure S1. Structural applicability-domain analysis of conventional models. A, Test structural coverage and the development leave-one-out fifth-percentile threshold. B-C, Structural similarity versus normalized distance from 0.5; this distance is not calibrated confidence. D-E, AUC and balanced accuracy by predefined similarity interval. F, Error rates, sample counts and structure-group percentile bootstrap 95% CIs conditional on fixed predictions, without model retraining. Models are selected by development fold-mean AUC only. The fixed heldout test is internal, not an external cohort. Source: applicability_domain.csv, ad_performance.csv and locked test_predictions.csv.")


def s2(source, out):
    if not source.graphs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * .85))
    fig.subplots_adjust(left=.095, right=.975, top=.91, bottom=.11, wspace=.37, hspace=.48)
    handles = []
    for i, model_id in enumerate(source.graphs):
        color, marker = GRAPH_COLORS[i], GRAPH_MARKERS[i]
        _, y, p = source.prediction(model_id, "development_oof")
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0, 0].plot(fpr, tpr, color=color, linewidth=1.2)
        axes[0, 1].plot(recall, precision, color=color, linewidth=1.2)
        records = [r for r in source.calibration if r["model_id"] == model_id and r["partition"] == "development_oof" and int(r["n"]) > 0]
        axes[1, 0].plot([number(r["mean_probability"]) for r in records], [number(r["positive_fraction"]) for r in records], color=color, marker=marker, markersize=3, linewidth=1)
        auc_values = [number(r["ROC_AUC"]) for r in source.fold if r["model_id"] == model_id]
        box = axes[1, 1].boxplot([auc_values], positions=[i], widths=.55, patch_artist=True, showfliers=False,
                                medianprops={"color": "black", "linewidth": 1}, whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
        box["boxes"][0].set(facecolor=color, alpha=.25, edgecolor=color)
        jitter = np.linspace(-.12, .12, len(auc_values))
        axes[1, 1].scatter(i + jitter, auc_values, color=color, s=8, alpha=.7)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=3, label=f"{source.label(model_id)}: AUC {roc_auc_score(y,p):.2f}; AP {average_precision_score(y,p):.2f}"))
    axes[0, 0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=.9)
    axes[0, 1].axhline(float(y.mean()), linestyle="--", color=GRAY, linewidth=.9)
    axes[1, 0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=.9)
    titles = ["Development OOF ROC", "Development OOF precision-recall", "Development OOF calibration", "Ten outer group folds"]
    labels = [("False-positive rate", "True-positive rate"), ("Recall", "Precision"),
              ("Mean predicted probability", "Observed positive fraction"), ("", "Fold ROC-AUC")]
    for ax, title, label, letter in zip(axes.flat, titles, labels, "ABCD"):
        panel(ax, letter, title)
        ax.set(xlabel=label[0], ylabel=label[1])
        if letter != "D":
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
    axes[1, 1].set_xticks(range(len(source.graphs)), [source.label(k) for k in source.graphs], rotation=12)
    axes[1, 1].set_ylim(0, 1.02)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .99), ncol=3, frameon=False, fontsize=5.3, columnspacing=1)
    export(fig, out, "FigureS2", "Figure S2. Development-set validation of graph models. A-B, Nested group-CV out-of-fold ROC and precision-recall curves. C, Equal-width ten-bin calibration; empty bins are omitted, counts are supplied in source data. D, Ten outer-validation fold AUCs; each point uses the same three-seed probability mean as final prediction. Folds are correlated resamples and are not independent biological replicates. Boxes show median and quartiles; whiskers extend to 1.5 IQR. Source: locked oof_predictions.csv, calibration_bins.csv and perfold_metrics.csv.")


def s3(source, out):
    if not source.graphs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * .85))
    fig.subplots_adjust(left=.095, right=.975, top=.91, bottom=.11, wspace=.37, hspace=.48)
    ax = axes[0, 0]
    oof = [number(r["mean5_tanimoto"]) for r in source.ad if r["partition"] == "development_oof"]
    test = [number(r["mean5_tanimoto"]) for r in source.ad if r["partition"] == "test"]
    tau = number(next(r for r in source.thresholds if r["partition"] == "test")["mean5_threshold"])
    ax.hist(test, bins=np.linspace(0, 1, 21), density=True, color="#d8d7e7", alpha=.9)
    ax.hist(oof, bins=np.linspace(0, 1, 21), density=True, histtype="step", color="#333333", linewidth=1.2)
    ax.axvline(tau, color=GRAPH_COLORS[0], linestyle="--", linewidth=1)
    ax.set(xlabel="Mean 5NN Morgan Tanimoto", ylabel="Density", xlim=(0, 1))
    panel(ax, "A", "Structural coverage")
    for ax, metric, title, letter in ((axes[0, 1], "ROC_AUC", "Domain-stratified AUC", "B"),
                                      (axes[1, 0], "BA", "Domain-stratified BA", "C"),
                                      (axes[1, 1], "error_rate", "Domain-stratified error", "D")):
        for i, model_id in enumerate(source.graphs):
            for offset, domain in enumerate(("inside", "outside")):
                row = source.domain_performance(model_id, "mean5", domain)
                value = number(row.get(metric))
                if not np.isfinite(value):
                    continue
                x = i + (offset - .5) * .32
                ax.bar(x, value, width=.32, color=GRAPH_COLORS[i], edgecolor=GRAPH_COLORS[i], alpha=.85 if offset == 0 else .3,
                       hatch=None if offset == 0 else "///", linewidth=.9)
                lower, upper = number(row.get(metric + "_lower")), number(row.get(metric + "_upper"))
                if np.isfinite(lower + upper):
                    ax.vlines(x, lower, upper, color="black", linewidth=.7)
                    ax.hlines([lower, upper], x - .025, x + .025, color="black", linewidth=.7)
                if metric == "error_rate":
                    ax.text(x, value + .025, f"{value:.2f}\nn={row['n']}", ha="center", fontsize=5)
        ax.set_xticks(range(len(source.graphs)), [source.label(k) for k in source.graphs], rotation=12)
        ax.set(ylabel=metric.replace("ROC_", "Test ").replace("_", " "), ylim=(0, 1.08))
        panel(ax, letter, title)
    fig.legend(handles=[Line2D([], [], color="#333333", label=f"OOF (n={len(oof)})"), Patch(color="#d8d7e7", label=f"Test (n={len(test)})"),
                         Line2D([], [], color=GRAPH_COLORS[0], linestyle="--", label=f"Test AD={tau:.3f}"),
                         Patch(color="#888888", label="In domain"), Patch(facecolor="white", edgecolor="#888888", hatch="///", label="Out of domain")],
               loc="upper center", ncol=5, bbox_to_anchor=(.53, .99), fontsize=5.3, frameon=False, columnspacing=1)
    export(fig, out, "FigureS3", "Figure S3. Graph-model structural applicability domain. A, Mean-five-nearest-neighbor similarities: OOF queries use only their own outer-training fold as reference; the dashed test threshold uses all development leave-one-out values. OOF membership uses fold-specific thresholds rather than this common test line. B-D, Locked-test domain-stratified AUC, balanced accuracy and errors. Error bars show structure-group percentile 95% bootstrap CIs conditional on fixed predictions, without model retraining; single-class AUC/BA draws are omitted. Domain sample sizes are in source tables and error annotations. Source: applicability_domain.csv, ad_thresholds.csv and ad_performance.csv.")


def s4(source, out):
    if not source.graphs:
        return
    pairs = read_csv(source.reliability / "binary_label_cliff_pairs.csv")
    predictions = read_csv(source.reliability / "label_cliff_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, WIDTH * .89))
    fig.subplots_adjust(left=.095, right=.975, top=.91, bottom=.11, wspace=.37, hspace=.47)
    if not pairs:
        for ax, letter in zip(axes.flat, "ABCD"):
            panel(ax, letter, "Discordant-label pairs")
            ax.text(.5, .5, "No pairs at Tanimoto >= 0.85", transform=ax.transAxes, ha="center")
        export(fig, out, "FigureS4", "Figure S4. No current discordant-label pairs satisfy the predefined Morgan Tanimoto threshold of 0.85.")
        return
    pair = pairs[0]
    ax = axes[0, 0]
    ax.axis("off")
    for inset, prefix, label in (((.0, .14, .49, .65), "positive", "Positive"), ((.51, .14, .49, .65), "reference_negative", "Reference-negative")):
        molecule_ax = ax.inset_axes(inset)
        molecule_ax.imshow(molecule_image(pair[prefix + "_SMILES"]))
        molecule_ax.axis("off")
        molecule_ax.text(.5, -.01, f"{label}\n{pair[prefix + '_id']} | CID {pair[prefix + '_CID']}", transform=molecule_ax.transAxes,
                         ha="center", va="top", fontsize=5)
    panel(ax, "A", f"Most similar discordant-label pair\nTanimoto = {number(pair['tanimoto']):.3f}")
    ax = axes[0, 1]
    for i, model_id in enumerate(source.graphs):
        row = next(r for r in predictions if r["model_id"] == model_id and r["positive_id"] == pair["positive_id"] and r["reference_negative_id"] == pair["reference_negative_id"])
        ax.bar(i - .17, number(row["positive_probability"]), width=.34, color=BLUE, edgecolor="black", linewidth=.5)
        ax.bar(i + .17, number(row["reference_negative_probability"]), width=.34, color=ORANGE, edgecolor="black", linewidth=.5)
    ax.axhline(.5, linestyle="--", color=GRAY, linewidth=.9)
    ax.set_xticks(range(len(source.graphs)), [source.label(k) for k in source.graphs], rotation=12)
    ax.set(ylabel="Predicted positive probability", ylim=(0, 1.03))
    panel(ax, "B", "Highest-similarity pair predictions")
    ax = axes[1, 0]
    for i, model_id in enumerate(source.graphs):
        rows = [r for r in predictions if r["model_id"] == model_id]
        ranked = np.mean([number(r["positive_minus_negative"]) > 0 for r in rows])
        both = np.mean([truth(r["both_correct"]) for r in rows])
        ax.bar(i - .17, ranked, width=.34, color=GRAPH_COLORS[i], edgecolor=GRAPH_COLORS[i], linewidth=.7)
        ax.bar(i + .17, both, width=.34, color=GRAPH_COLORS[i], edgecolor=GRAPH_COLORS[i], alpha=.3, hatch="///", linewidth=.7)
    ax.set_xticks(range(len(source.graphs)), [source.label(k) for k in source.graphs], rotation=12)
    ax.set(ylabel="Fraction of discordant-label pairs", ylim=(0, 1.05))
    panel(ax, "C", f"Descriptive summary across {len(pairs)} pairs")
    ax = axes[1, 1]
    for i, model_id in enumerate(source.graphs):
        rows = [r for r in predictions if r["model_id"] == model_id]
        ax.scatter([number(r["tanimoto"]) for r in rows], [number(r["positive_minus_negative"]) for r in rows],
                   color=GRAPH_COLORS[i], marker=GRAPH_MARKERS[i], s=15, alpha=.65)
    ax.axhline(0, linestyle="--", color=GRAY, linewidth=.9)
    ax.set(xlabel="Pairwise Morgan Tanimoto", ylabel="p(positive) - p(reference-negative)", xlim=(.84, 1.01), ylim=(-1.02, 1.02))
    panel(ax, "D", "Within-pair probability separation")
    fig.legend(handles=[Patch(color=BLUE, label="Positive member"), Patch(color=ORANGE, label="Reference-negative member"),
                        Patch(color="#777777", label="Correct pair ranking"), Patch(facecolor="white", edgecolor="#777777", hatch="///", label="Both correct at 0.5")],
               loc="upper center", bbox_to_anchor=(.53, .99), ncol=4, frameon=False, fontsize=5.1, columnspacing=1)
    export(fig, out, "FigureS4", "Figure S4. Current binary discordant-label pairs at chiral Morgan Tanimoto >=0.85. A-B, Highest-similarity pair and locked predictions. Each endpoint uses OOF probability if in development and final-model probability if in test, explicitly recorded in label_cliff_predictions.csv. C, Descriptive pair ranking and both-label correctness; pairs sharing endpoints are dependent, so no binomial test or independent-pair CI is claimed. D, Signed probability separation versus similarity. Unique-endpoint compound metrics, deduplicated and separated by development/test, are in label_cliff_unique_endpoint_performance.csv. These are not experimentally established activity cliffs.")


def s5(source, out):
    records = read_csv(source.reliability / "structural_key_associations.csv")
    significant = sorted([r for r in records if truth(r["significant_BH_005"]) and not truth(r["constant_key"])],
                         key=lambda r: (number(r["q_BH"]), int(r["MACCS_key"])))
    shown = significant[:12]
    fig = plt.figure(figsize=(WIDTH, WIDTH * .85))
    forest = fig.add_axes([.1, .12, .37, .78])
    if shown:
        for i, record in enumerate(shown):
            value, low, high = [number(record[k]) for k in ("corrected_OR", "OR_lower", "OR_upper")]
            forest.errorbar(value, len(shown) - 1 - i, xerr=[[value - low], [high - value]], fmt="o", markersize=3,
                            color=GRAPH_COLORS[0] if value > 1 else "#377EB8", linewidth=1)
        forest.set_yticks(range(len(shown)), ["MACCS " + r["MACCS_key"] for r in shown[::-1]])
        forest.set_xscale("log")
    else:
        forest.text(.5, .5, "No keys pass BH q < 0.05", transform=forest.transAxes, ha="center", fontsize=7)
    forest.axvline(1, linestyle="--", color=GRAY, linewidth=.8)
    forest.set_xlabel("Corrected odds ratio (95% CI)")
    panel(forest, "A", "Development structural associations")
    fig.text(.54, .935, "B", fontsize=9, fontweight="bold")
    fig.text(.73, .935, "Representative development compounds", ha="center", fontsize=7)
    represented = []
    for record in shown:
        pattern = Chem.MolFromSmarts(record["SMARTS"]) if record["SMARTS"] != "?" else None
        if pattern is None:
            continue
        expected_label = 1 if number(record["corrected_OR"]) > 1 else 0
        choices = []
        for row in source.identity.values():
            if row["partition"] != "development" or int(row["label"]) != expected_label:
                continue
            molecule = Chem.MolFromSmiles(row["standardized_smiles"])
            if len(molecule.GetSubstructMatches(pattern)) > int(record["match_count_must_exceed"]):
                choices.append((molecule.GetNumHeavyAtoms(), row["compound_id"], row))
        if choices:
            represented.append((record, min(choices, key=lambda r: (r[0], r[1]))[2]))
        if len(represented) == 6:
            break
    for i, (record, row) in enumerate(represented):
        ax = fig.add_axes([.53 + (i % 2) * .225, .665 - (i // 2) * .25, .22, .2])
        ax.imshow(molecule_image(row["standardized_smiles"], record["SMARTS"]))
        ax.axis("off")
        ax.set_title(f"MACCS {record['MACCS_key']}; q={number(record['q_BH']):.3g}\n{row['compound_id']} | CID {row['PubChem_CID']}", fontsize=5.5, pad=1)
    fig.text(.52, .055, "Exploratory association; no kidney-specific\ntoxicophore or causal mechanism is inferred.", fontsize=5.5)
    export(fig, out, "FigureS5", f"Figure S5. Development-only associations of the 166 predefined modeled MACCS keys. A, Up to 12 keys with the lowest BH-adjusted two-sided Fisher P values among q<0.05 ({len(significant)} significant in total); all 166 results and RDKit SMARTS/count definitions are supplied in structural_key_associations.csv. Odds ratios and displayed Wald 95% CIs use a 0.5 Haldane-Anscombe correction. Tests/intervals treat compounds as observations and do not account for within-scaffold dependence; these results are exploratory. B, Up to six associated keys illustrated by development compounds with the smallest heavy-atom count and then lowest ID among matching compounds of the association direction; atom highlights show SMARTS matches. They are illustrative associations, not kidney-specific toxicophores or causal evidence. Key 1/125/166 algorithmic definitions are documented separately. Structures preserve source stereochemistry.")


def calibration_supplement(source, out):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH * .42))
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.2, wspace=.37)
    for ax, partition, letter in zip(axes, ("development_oof", "test"), "AB"):
        for i, model_id in enumerate((source.primary, source.comparison)):
            records = [r for r in source.calibration if r["model_id"] == model_id and r["partition"] == partition and int(r["n"]) > 0]
            ax.plot([number(r["mean_probability"]) for r in records], [number(r["positive_fraction"]) for r in records],
                    color=("#377EB8", "#7B6FA6")[i], marker=("o", "s")[i], markersize=3, linewidth=1, label=source.label(model_id))
        ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=.9)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", xlim=(0, 1), ylim=(0, 1.02))
        panel(ax, letter, "Development OOF calibration" if partition == "development_oof" else "Heldout test calibration")
    axes[0].legend(fontsize=5.3, frameon=False)
    export(fig, out, "FigureS_Calibration", "Supplementary conventional-model calibration. A-B, Equal-width ten-bin reliability curves for the development-selected primary and the development-selected representation comparator; these are evaluations of saved probabilities, not post-test probability recalibration. Empty bins are omitted; bin counts, Brier and log-loss are in source reports. The same primary/comparator selection applies to Figure S1.")


def count_supplement(source, out):
    records = [r for r in read_csv(source.reliability / "binary_count_comparisons.csv") if r.get("metric") == "ROC_AUC"]
    if not records:
        return
    reps = list(dict.fromkeys(r["binary_representation"] for r in records))
    fig, axes = plt.subplots(len(reps), 2, figsize=(WIDTH, WIDTH * .82), squeeze=False)
    fig.subplots_adjust(left=.095, right=.975, top=.91, bottom=.09, wspace=.36, hspace=.65)
    algorithms = ["rf", "svm", "mlp", "xgb", "dnn"]
    for row, representation in enumerate(reps):
        for col, partition in enumerate(("development_oof", "test")):
            ax = axes[row, col]
            selected = [r for r in records if r["binary_representation"] == representation and r["partition"] == partition]
            by_algo = {r["algorithm"].lower(): r for r in selected}
            for i, algorithm in enumerate(algorithms):
                if algorithm not in by_algo:
                    continue
                record = by_algo[algorithm]
                a, b = number(record["binary_value"]), number(record["count_value"])
                ax.plot([i - .12, i + .12], [a, b], color="#bbbbbb", linewidth=.8)
                ax.scatter(i - .12, a, color=BLUE, marker="o", s=13)
                ax.scatter(i + .12, b, color=ORANGE, marker="s", s=13)
            ax.set_xticks(range(5), [ALGORITHM_LABELS[a] for a in algorithms], rotation=10)
            ax.set(ylabel="ROC-AUC", ylim=(0, 1.02))
            panel(ax, chr(65 + row * 2 + col), REP_LABELS.get(representation, representation) + (" / Development OOF" if col == 0 else " / Heldout test"))
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Binary"),
                        Line2D([], [], color=ORANGE, marker="s", linestyle="none", label="Count")],
               loc="upper center", bbox_to_anchor=(.53, .99), ncol=2, fontsize=6, frameon=False)
    export(fig, out, "FigureS_BinaryCount", "Supplementary binary versus count fingerprint sensitivity analysis. Matched classifier/representation pairs use the same current structure-group holdout and outer folds, with the documented finite candidate budget. Development points are nested outer OOF AUC; test points use separately locked final fits. Lines join representations for the same algorithm and do not imply a significance test. Count-minus-binary values for AUC, BA, MCC, AP and Brier are supplied in binary_count_comparisons.csv. This sensitivity analysis does not retrospectively change the development-selected main primary.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()
    reliability = args.reliability or args.results / "reliability"
    out = args.output or args.results / "figures/reliability"
    out.mkdir(parents=True, exist_ok=True)
    typography = apply_font()
    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 5.3, "axes.linewidth": .65,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    source = Source(args.results, reliability)
    figure_contract(out, source.manifest)
    for draw in (s1, s2, s3, s4, s5, calibration_supplement, count_supplement):
        draw(source, out)
    write_json(out / "supplementary_figure_manifest.json", {"status": "rendered; visual review required",
        "typography": typography, "source_reliability_manifest_sha256": hashlib.sha256((reliability / "reliability_manifest.json").read_bytes()).hexdigest(),
        "figure_names": [p.stem for p in out.glob("*.pdf")],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != "supplementary_figure_manifest.json"}})
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for file in out.iterdir():
            if file.is_file():
                shutil.copy2(file, args.copy_to / file.name)
    print(json.dumps({"output": str(out.resolve()), "figures": [p.stem for p in out.glob("*.pdf")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
