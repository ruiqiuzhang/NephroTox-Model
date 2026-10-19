"""Render Figure 6/7 and Figure S6/S7 from new locked explanation sources.

Layouts follow the earlier bond-occlusion/SHAP figures; every plotted number is
recomputed from the current DATA and newly fitted models. Arial is mandatory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import rankdata

from figure_style import FONT_FAMILY, apply_font, font_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.modeling_explain import RF_MODELS, GRAPH_MODELS, CASE_CATEGORIES, explain_models

GRAPH_NAMES = ("D-MPNN", "GINE", "AttentiveFP")
GRAPH_COLORS = ("#F05A43", "#4CB6CB", "#355A93")
RF_COLORS = ("#8C6BA6", "#648DB7")
CATEGORY_NAMES = {"TP": "High-confidence TP", "TN": "High-confidence TN",
                  "FP": "High-confidence FP", "FN": "High-confidence FN"}
CMAP = plt.get_cmap("RdBu_r")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_sources(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "explain_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return {"manifest": manifest, "folder": folder}
    for name, expected in manifest["files"].items():
        if Path(name).name != name or not (folder / name).is_file() or digest(folder / name) != expected:
            raise ValueError(f"Explanation source missing or changed: {name}")
    result = {"manifest": manifest, "folder": folder}
    for name in ("cases", "shap_feature_summary", "shap_fold_importance", "shap_fold_stability",
                 "permutation_importance", "local_maccs_atom_sensitivity", "graph_bond_occlusion",
                 "graph_case_predictions", "graph_occlusion_correlations"):
        result[name] = read_csv(folder / f"{name}.csv")
    for case in result["cases"]:
        for key in ("row_index", "label", "cv_validation_fold", "PubChem_CID"):
            case[key] = int(case[key])
    result["shap"] = {}
    for model in RF_MODELS:
        with np.load(folder / f"{model}_shap_oof.npz", allow_pickle=False) as payload:
            result["shap"][model] = {key: np.array(payload[key], copy=True) for key in payload.files}
    return result


def style_axes(ax, grid=False):
    ax.tick_params(labelsize=6.5, width=0.65, length=2.5, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    if grid:
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)


def panel_letter(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=10, weight="bold", va="top")


def short_name(value, width=22, max_lines=3):
    lines = textwrap.wrap(str(value), width=width, break_long_words=True)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(1, width - 1)] + "…"
    return "\n".join(lines)


def molecule_image(smiles, *, atom_values=None, bond_values=None, limit=1.0, save_path=None):
    """Depict the exact supplied molecule with Arial atom lettering."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid source case molecule")
    before = Chem.MolToSmiles(molecule, isomericSmiles=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 600)
    options = drawer.drawOptions()
    options.fontFile = str(font_path())
    options.minFontSize, options.maxFontSize = 36, 44
    options.bondLineWidth = 2.5
    options.highlightBondWidthMultiplier = 12
    options.padding = 0.07
    options.fillHighlights = True
    atom_colors, bond_colors = {}, {}
    norm = colors.Normalize(-limit, limit, clip=True)
    for index, value in (atom_values or {}).items():
        atom_colors[int(index)] = tuple(float(c) for c in CMAP(norm(float(value)))[:3])
    for index, value in (bond_values or {}).items():
        bond_colors[int(index)] = tuple(float(c) for c in CMAP(norm(float(value)))[:3])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule, highlightAtoms=list(atom_colors), highlightBonds=list(bond_colors),
        highlightAtomColors=atom_colors, highlightBondColors=bond_colors,
        highlightAtomRadii={a: 0.28 for a in atom_colors},
    )
    drawer.FinishDrawing()
    content = drawer.GetDrawingText()
    if before != Chem.MolToSmiles(molecule, isomericSmiles=True):
        raise ValueError("Drawing altered source molecular stereochemistry")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(content)
    return np.array(Image.open(io.BytesIO(content)).convert("RGBA"))


def _bond_map(source, model, identifier):
    return {int(r["bond_index"]): float(r["delta_probability"]) for r in source["graph_bond_occlusion"]
            if r["model_id"] == model and r["compound_id"] == identifier}


def _case_probability(source, model, identifier):
    return float(next(r["probability"] for r in source["graph_case_predictions"]
                      if r["model_id"] == model and r["compound_id"] == identifier))


def _bond_colorbar(fig, rectangle, limit):
    axis = fig.add_axes(rectangle)
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(-limit, limit), cmap=CMAP), cax=axis,
                       orientation="horizontal")
    bar.ax.tick_params(labelsize=6, length=2, pad=1)
    return bar


def draw_figure6(source, structure_folder):
    """Four OOF case rows, three graph columns and outcome column (old Fig 7)."""
    fig = plt.figure(figsize=(8.54, 10.0))
    gs = fig.add_gridspec(4, 5, left=.025, right=.985, bottom=.125, top=.915,
                         width_ratios=[1.1, 1.5, 1.5, 1.5, .8], hspace=.12, wspace=.015)
    panel_letter(fig, .007, .99, "A")
    headers = ["OOF case (MACCS-RF)", *GRAPH_NAMES, "Dataset label"]
    for column, header in enumerate(headers):
        position = gs[0, column].get_position(fig)
        fig.text((position.x0 + position.x1) / 2, .952, header, ha="center", fontsize=8,
                 color=GRAPH_COLORS[column - 1] if 1 <= column <= 3 else "black", weight="bold")
    lookup = {r["category"]: r for r in source["cases"] if r["role"] == "main"}
    limit = (source["manifest"]["graph_bond_color_scale"] or {}).get("symmetric_color_limit", 1e-9)
    for row, category in enumerate(CASE_CATEGORIES):
        case = lookup.get(category)
        label_axis = fig.add_subplot(gs[row, 0]); label_axis.set_axis_off()
        if case is None:
            label_axis.text(.5, .5, f"{category}\nNo OOF case", ha="center", va="center", fontsize=7)
            continue
        label_axis.text(.5, .83, CATEGORY_NAMES[category], ha="center", va="center", fontsize=7, weight="bold")
        label_axis.text(.5, .52, short_name(case["name"]), ha="center", va="center", fontsize=6.7)
        label_axis.text(.5, .24, case["compound_id"], ha="center", color="#666666", fontsize=6.7)
        label_axis.text(.5, .10, f'CID {case["PubChem_CID"]}', ha="center", color="#666666", fontsize=6)
        correct = 0
        for column, model in enumerate(GRAPH_MODELS):
            ax = fig.add_subplot(gs[row, column + 1]); ax.set_axis_off()
            image = molecule_image(case["standardized_smiles"], bond_values=_bond_map(source, model, case["compound_id"]),
                                   limit=limit, save_path=structure_folder / f'Figure6_{category}_{model}.png')
            ax.imshow(image, aspect="equal")
            p = _case_probability(source, model, case["compound_id"])
            ax.text(.5, -.03, f"p = {p:.3f}", transform=ax.transAxes, ha="center", fontsize=7)
            correct += int(int(p >= .5) == case["label"])
        outcome = fig.add_subplot(gs[row, 4]); outcome.set_axis_off()
        outcome.text(.5, .68, "Positive" if case["label"] else "Reference-\nnegative", ha="center", va="center",
                     fontsize=7, weight="bold", color=GRAPH_COLORS[0] if case["label"] else "#2F6FB0")
        outcome.text(.5, .43, f"{correct}/3 correct", ha="center", fontsize=6.7)
        outcome.text(.5, .22, "threshold 0.50", ha="center", fontsize=6, color="#777777")
    _bond_colorbar(fig, [.225, .074, .61, .015], limit)
    fig.text(.53, .047, "Single-bond edge occlusion: original p − occluded p; display clipped at 95th percentile",
             ha="center", fontsize=6.3)
    fig.text(.53, .026, "Red supports p; blue opposes p. Case selection and predictions use development OOF models.",
             ha="center", fontsize=6.1)
    return fig


def _importance_bar(ax, records, color, title):
    chosen = sorted(records, key=lambda r: (-float(r["OOF_mean_absolute_shap"]), r["feature_name"]))[:12]
    y = np.arange(len(chosen))
    ax.barh(y, [float(r["OOF_mean_absolute_shap"]) for r in chosen],
            xerr=[float(r["fold_sd_absolute_shap"]) for r in chosen], color=color, height=.75,
            error_kw={"elinewidth": .7, "capsize": 1.8, "capthick": .7})
    ax.set_yticks(y, [r["feature_name"].replace("MACCS_", "MACCS ") for r in chosen], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8, weight="bold", pad=5)
    ax.set_xlabel("OOF mean absolute TreeSHAP value", fontsize=7)
    style_axes(ax, True)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    return chosen


def draw_figure7(source, structure_folder):
    """Two global bars, descriptor OOF summary and four local atom maps (old Fig 8)."""
    fig = plt.figure(figsize=(9.0, 10.1))
    ax_a = fig.add_axes([.145, .735, .32, .22])
    ax_b = fig.add_axes([.645, .735, .32, .22])
    first = [r for r in source["shap_feature_summary"] if r["model_id"] == RF_MODELS[0]]
    second = [r for r in source["shap_feature_summary"] if r["model_id"] == RF_MODELS[1]]
    _importance_bar(ax_a, first, RF_COLORS[0], "MACCS-RF comparator")
    selected = _importance_bar(ax_b, second, RF_COLORS[1], "Descriptor-RF comparator")
    panel_letter(fig, .015, .982, "A"); panel_letter(fig, .515, .982, "B")
    panel_letter(fig, .015, .68, "C")
    ax = fig.add_axes([.18, .415, .73, .235])
    payload = source["shap"][RF_MODELS[1]]
    rng = np.random.default_rng(20261019)
    for rank, record in enumerate(selected):
        column = int(record["feature_column_zero_based"])
        feature = payload["X"][:, column]
        value_rank = rankdata(feature, method="average") / len(feature)
        ax.scatter(payload["shap"][:, column], rank + rng.uniform(-.23, .23, len(feature)),
                   c=value_rank, cmap="coolwarm", vmin=0, vmax=1, s=3, alpha=.55, linewidths=0, rasterized=True)
    ax.set_yticks(np.arange(len(selected)), [r["feature_name"] for r in selected], fontsize=6.5)
    ax.invert_yaxis(); ax.axvline(0, color="#777777", linewidth=.65)
    ax.set_title("Descriptor-RF directionality in cross-fitted development predictions", fontsize=8, weight="bold", pad=6)
    ax.set_xlabel("TreeSHAP contribution to positive-class probability", fontsize=7)
    style_axes(ax, True)
    cb = fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap="coolwarm"),
                      cax=fig.add_axes([.932, .415, .012, .235]))
    cb.set_ticks([0, 1], labels=["Low", "High"]); cb.ax.tick_params(labelsize=6, length=0)
    cb.set_label("Within-feature value percentile", fontsize=6.5, labelpad=5)
    panel_letter(fig, .015, .34, "D")
    lookup = {r["category"]: r for r in source["cases"] if r["role"] == "main"}
    limit = (source["manifest"]["MACCS_atom_color_scale"] or {}).get("symmetric_color_limit", 1e-9)
    for j, category in enumerate(CASE_CATEGORIES):
        ax = fig.add_axes([.035 + j * .24, .125, .225, .18]); ax.set_axis_off()
        case = lookup.get(category)
        ax.text(.5, 1.16, CATEGORY_NAMES[category], transform=ax.transAxes, ha="center", fontsize=7, weight="bold")
        if case is None:
            ax.text(.5, .5, "No OOF case", transform=ax.transAxes, ha="center", fontsize=7)
            continue
        atom_values = {int(r["atom_index"]): float(r["atom_sensitivity"])
                       for r in source["local_maccs_atom_sensitivity"] if r["compound_id"] == case["compound_id"]}
        ax.text(.5, 1.025, case["compound_id"], transform=ax.transAxes, ha="center", fontsize=6.6)
        image = molecule_image(case["standardized_smiles"], atom_values=atom_values, limit=limit,
                               save_path=structure_folder / f"Figure7_{category}_MACCS_bit_off.png")
        ax.imshow(image, aspect="equal")
        ax.text(.5, -.07, f'OOF MACCS-RF p = {float(case["selection_probability"]):.3f}',
                transform=ax.transAxes, ha="center", fontsize=6.5)
    _bond_colorbar(fig, [.25, .061, .53, .013], limit)
    fig.text(.515, .037, "Local MACCS input bit-off sensitivity, projected over matching atoms; 95th percentile display clipping",
             ha="center", fontsize=6.1)
    fig.text(.515, .019, "Bars: OOF mean |SHAP|; errors: SD across 10 folds. Input perturbations do not establish chemical mechanisms.",
             ha="center", fontsize=6)
    return fig


def draw_s6(source, structure_folder):
    """Two additional cases, pooled attribution correlations, six-case probabilities."""
    fig = plt.figure(figsize=(8.8, 8.3))
    gs = fig.add_gridspec(2, 4, left=.025, right=.985, bottom=.44, top=.94,
                         width_ratios=[.95, 1.5, 1.5, 1.5], hspace=.10, wspace=.015)
    panel_letter(fig, .006, .99, "A")
    limit = (source["manifest"]["graph_bond_color_scale"] or {}).get("symmetric_color_limit", 1e-9)
    extras = [c for c in source["cases"] if c["role"] == "supplementary"]
    for j, name in enumerate(GRAPH_NAMES):
        position = gs[0, j + 1].get_position(fig)
        fig.text((position.x0 + position.x1) / 2, .962, name, ha="center", fontsize=8, color=GRAPH_COLORS[j], weight="bold")
    for row, case in enumerate(extras[:2]):
        ax = fig.add_subplot(gs[row, 0]); ax.set_axis_off()
        ax.text(.5, .78, "Largest OOF\nprobability spread", ha="center", fontsize=7, weight="bold")
        ax.text(.5, .48, case["compound_id"], ha="center", fontsize=7)
        ax.text(.5, .23, "Positive" if case["label"] else "Reference-\nnegative", ha="center", fontsize=6.5)
        for j, model in enumerate(GRAPH_MODELS):
            ax = fig.add_subplot(gs[row, j + 1]); ax.set_axis_off()
            image = molecule_image(case["standardized_smiles"], bond_values=_bond_map(source, model, case["compound_id"]),
                                   limit=limit, save_path=structure_folder / f'S6_{case["compound_id"]}_{model}.png')
            ax.imshow(image)
            ax.text(.5, -.03, f'p = {_case_probability(source, model, case["compound_id"]):.3f}',
                    transform=ax.transAxes, ha="center", fontsize=7)
    _bond_colorbar(fig, [.26, .392, .55, .013], limit)
    fig.text(.535, .366, "Original p − edge-occluded p; same 95th percentile scale as Figure 6", ha="center", fontsize=6.5)
    panel_letter(fig, .015, .325, "B"); panel_letter(fig, .39, .325, "C")
    ax = fig.add_axes([.11, .095, .22, .21])
    correlation = np.full((3, 3), np.nan)
    for r in source["graph_occlusion_correlations"]:
        if r["compound_id"] == "all_cases_pooled_bonds" and r["spearman_rho"]:
            correlation[GRAPH_MODELS.index(r["model_a"]), GRAPH_MODELS.index(r["model_b"])] = float(r["spearman_rho"])
    ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{correlation[i, j]:.2f}" if np.isfinite(correlation[i, j]) else "NA", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(3), GRAPH_NAMES, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(3), GRAPH_NAMES, fontsize=6)
    ax.set_title("Bond-attribution Spearman ρ\n(all selected cases pooled)", fontsize=7)
    ax.tick_params(length=0)
    ax = fig.add_axes([.435, .095, .53, .21])
    all_cases = source["cases"]
    x = np.arange(len(all_cases)); width = .24
    for j, model in enumerate(GRAPH_MODELS):
        ax.bar(x + (j - 1) * width, [_case_probability(source, model, c["compound_id"]) for c in all_cases],
               width, label=GRAPH_NAMES[j], color=GRAPH_COLORS[j])
    ax.axhline(.5, color="#666666", linewidth=.7, linestyle="--")
    ax.set_xticks(x, [c["compound_id"] for c in all_cases], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("OOF positive-class probability", fontsize=7); ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.19), handlelength=1)
    fig.text(.5, .02, "These are fitted-model edge sensitivities; agreement does not demonstrate a toxic mechanism.", ha="center", fontsize=6.4)
    return fig


def draw_s7(source):
    """Fold-specific importance heatmaps and post-lock test permutation checks."""
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 7.4))
    fig.subplots_adjust(left=.155, right=.96, bottom=.11, top=.94, wspace=.58, hspace=.52)
    for j, model in enumerate(RF_MODELS):
        records = [r for r in source["shap_feature_summary"] if r["model_id"] == model]
        selected = sorted(records, key=lambda r: (-float(r["OOF_mean_absolute_shap"]), r["feature_name"]))[:12]
        lookup = {(r["feature_name"], int(r["fold"])): float(r["mean_absolute_shap"])
                  for r in source["shap_fold_importance"] if r["model_id"] == model}
        heat = np.array([[lookup[(r["feature_name"], f)] for f in range(1, 11)] for r in selected])
        ax = axes[0, j]
        image = ax.imshow(heat, aspect="auto", cmap="Purples" if j == 0 else "Blues", vmin=0)
        ax.set_yticks(range(len(selected)), [r["feature_name"].replace("MACCS_", "MACCS ") for r in selected], fontsize=6.5)
        ax.set_xticks(range(10), range(1, 11), fontsize=6.5)
        ax.set_xlabel("Development validation fold", fontsize=7)
        ax.set_title("MACCS-RF" if j == 0 else "Descriptor-RF", fontsize=8, weight="bold")
        fig.colorbar(image, ax=ax, fraction=.036, pad=.025).ax.tick_params(labelsize=6)
        ax.tick_params(length=0)
        ax.text(-.42 if j == 1 else -.25, 1.13, "AB"[j], transform=ax.transAxes, fontsize=10, weight="bold")
        ax = axes[1, j]
        permutation = {r["feature_name"]: r for r in source["permutation_importance"] if r["model_id"] == model}
        x = np.array([float(r["OOF_mean_absolute_shap"]) for r in records])
        y = np.array([float(permutation[r["feature_name"]]["mean_auc_drop"]) for r in records])
        error = np.array([float(permutation[r["feature_name"]]["sd_auc_drop"]) for r in records])
        ax.errorbar(x, y, yerr=error, fmt="o", markersize=2.5, color=RF_COLORS[j],
                    ecolor=RF_COLORS[j], elinewidth=.35, alpha=.6, capsize=0)
        ax.axhline(0, color="#777777", linewidth=.6)
        ax.set_xlabel("Development OOF mean |TreeSHAP|", fontsize=7)
        ax.set_ylabel("Held-out permutation ΔROC-AUC", fontsize=7)
        rho = source["manifest"]["shap_checks"][model]["SHAP_to_test_permutation_spearman"]
        ax.text(.03, .96, f"Spearman ρ = {rho:.3f}" if rho is not None else "Spearman ρ undefined",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.text(-.42 if j == 1 else -.25, 1.13, "CD"[j], transform=ax.transAxes, fontsize=10, weight="bold")
        style_axes(ax, True)
    fig.text(.52, .035, "Top: fold-specific mean |SHAP|. Bottom: all 166 features; error bars are SD of 10 held-out permutations.",
             ha="center", fontsize=6.5)
    fig.text(.52, .013, "The 335-record held-out set is internal to this dataset; it is not an external validation cohort.",
             ha="center", fontsize=6.3)
    return fig


def validate_figure(fig):
    """Require Arial and readable text inside the actual rendering canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    inspected = list(fig.texts)
    for ax in fig.axes:
        inspected.extend(ax.texts)
        if ax.axison:
            # Matplotlib retains tick-label artists just beyond the locator's
            # visible limits. They are not drawn; exclude those inactive ticks.
            x_low, x_high = sorted(ax.get_xlim()); y_low, y_high = sorted(ax.get_ylim())
            x_labels = [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
                        if x_low - 1e-12 <= value <= x_high + 1e-12]
            y_labels = [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                        if y_low - 1e-12 <= value <= y_high + 1e-12]
            inspected.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *x_labels, *y_labels])
    outside = []
    for text in inspected:
        if not text.get_visible() or not text.get_text().strip():
            continue
        if FONT_FAMILY not in text.get_fontfamily():
            raise ValueError(f"Figure text is not {FONT_FAMILY}: {text.get_text()}")
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > width + 1 or b.y1 > height + 1:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside figure canvas: {outside}")


def save_figure(fig, output, stem):
    """Check figure text/font and export scientific vector/raster deliverables."""
    output.mkdir(parents=True, exist_ok=True)
    validate_figure(fig)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=600, facecolor="white")
    with Image.open(output / f"{stem}.png") as raster:
        raster.convert("RGB").save(output / f"{stem}.tiff", compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-results", "--results", dest="model_results", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/model_explanations")
    parser.add_argument("--prepare", action="store_true", help="First compute explanations after the complete global lock")
    args = parser.parse_args()
    apply_font()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "savefig.dpi": 600})
    source_folder = args.source or args.model_results / "explain"
    if args.prepare:
        from nephrotox.data import load_data
        rows, _ = load_data()
        assignments = json.loads((args.model_results.parent / "split_payload.json").read_text(encoding="utf-8"))["assignments"]
        explain_models(rows, assignments, args.model_results, source_folder)
    if not (source_folder / "explain_manifest.json").exists():
        print("PENDING: explanation sources do not yet exist; run --prepare after all models are locked.")
        return
    source = load_sources(source_folder)
    if source["manifest"].get("status") != "complete":
        print("PENDING: " + source["manifest"].get("reason", "explanations are incomplete"))
        return
    structure_folder = args.output / "structures"
    for stem, draw in (("Figure6", draw_figure6), ("Figure7", draw_figure7), ("FigureS6", draw_s6)):
        save_figure(draw(source, structure_folder), args.output, stem)
    save_figure(draw_s7(source), args.output, "FigureS7")
    manifest = {"status": "complete", "font_family": FONT_FAMILY,
                "source_manifest_sha256": digest(source_folder / "explain_manifest.json"),
                "render_source_sha256": digest(__file__), "chemical_atom_font": str(font_path().name),
                "display_percentile": 95, "historical_numerical_results_used": False,
                "files": {str(p.relative_to(args.output)).replace("\\", "/"): digest(p)
                          for p in args.output.rglob("*") if p.is_file() and p.name != "render_manifest.json"}}
    (args.output / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
