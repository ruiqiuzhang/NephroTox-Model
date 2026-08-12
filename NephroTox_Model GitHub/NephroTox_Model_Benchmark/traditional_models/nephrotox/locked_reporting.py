from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from rdkit import DataStructs
from rdkit import rdBase
from sklearn.metrics import confusion_matrix, roc_curve

from .config import project_path, write_json
from .features import load_feature_matrix
from .locked_pipeline import MODELS, REPRESENTATIONS
from .metrics import calculate_metrics
from .statistics import paired_bootstrap_difference, stratified_bootstrap_intervals
from .supplementary import export_split_tables, export_supplementary_tables


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.linewidth"] = 0.8


REPRESENTATION_LABELS = {
    "morgan2048": "Morgan",
    "maccs166": "MACCS",
    "atompair1024": "AtomPairs",
    "torsion1024": "TopologicalTorsion",
    "rdkit166": "Descriptor",
}
MODEL_LABELS = {"rf": "RF", "svm": "SVM", "mlp": "MLP", "xgb": "XGBoost", "dnn": "DNN"}
COLORS = {
    "morgan2048": "#E74C3C",
    "maccs166": "#377EB8",
    "atompair1024": "#F2C14E",
    "torsion1024": "#55B98D",
    "rdkit166": "#7A78A6",
}
MARKERS = {
    "morgan2048": "s",
    "maccs166": "o",
    "atompair1024": "^",
    "torsion1024": "v",
    "rdkit166": "D",
}
FIGURE4_MODEL_ORDER = ["dnn", "xgb", "mlp", "rf", "svm"]
FIGURE4_MATRIX_ORDER = [
    "torsion1024",
    "morgan2048",
    "maccs166",
    "atompair1024",
    "rdkit166",
]
FIGURE5_MODEL_ORDER = ["rf", "svm", "mlp", "xgb", "dnn"]
FIGURE5_HEATMAP_MODEL_ORDER = ["svm", "rf", "xgb", "mlp", "dnn"]
FIGURE5_HEATMAP_REP_ORDER = [
    "rdkit166",
    "torsion1024",
    "atompair1024",
    "maccs166",
    "morgan2048",
]


def _results_root(config: dict[str, Any]) -> Path:
    return project_path(config, config["run"]["results_root"])


def _require_completed_evaluation(config: dict[str, Any]) -> None:
    path = _results_root(config) / "evaluation/test_evaluation_manifest.json"
    if not path.exists():
        raise RuntimeError(
            "Summary generation is disabled until the locked test evaluation is complete."
        )


def _capture_environment(output: Path) -> dict[str, str]:
    environment = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdBase.rdkitVersion,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }
    write_json(environment, output / "environment.json")
    requirements = [
        f"numpy=={environment['numpy']}",
        f"pandas=={environment['pandas']}",
        f"scikit-learn=={environment['scikit_learn']}",
        f"rdkit=={environment['rdkit']}",
        f"xgboost=={environment['xgboost']}",
        f"torch=={environment['torch']}",
        f"matplotlib=={environment['matplotlib']}",
    ]
    (output / "requirements-lock.txt").write_text(
        "\n".join(requirements) + "\n", encoding="utf-8"
    )
    return environment


def _collect_results(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    evaluation_root = _results_root(config) / "evaluation"
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    reference_ids: list[str] | None = None
    for representation in REPRESENTATIONS:
        for model_name in MODELS:
            experiment = evaluation_root / representation / model_name
            metrics = json.loads((experiment / "metrics.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(experiment / "test_predictions.csv").sort_values(
                "row_index"
            )
            ids = frame["compound_id"].astype(str).tolist()
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError("Test prediction files are not row-aligned.")
            predictions[(representation, model_name)] = frame
            row = {
                "representation": representation,
                "representation_label": REPRESENTATION_LABELS[representation],
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "best_mean_cv_roc_auc": metrics["best_mean_cv_roc_auc"],
                "best_std_cv_roc_auc": metrics["best_std_cv_roc_auc"],
                "selected_candidate_id": metrics["selected_candidate_id"],
            }
            row.update({f"test_{key}": value for key, value in metrics["locked_test_metrics"].items()})
            rows.append(row)
    return pd.DataFrame(rows), predictions


def _bootstrap_all(
    config: dict[str, Any],
    summary: pd.DataFrame,
    predictions: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(config["validation"]["classification_threshold"])
    replicates = int(config["validation"]["bootstrap_replicates"])
    base_seed = int(config["seeds"]["global"])
    wide = summary.copy()
    long_rows: list[dict[str, Any]] = []
    for index, row in wide.iterrows():
        key = (row["representation"], row["model"])
        frame = predictions[key]
        intervals = stratified_bootstrap_intervals(
            frame["label"].to_numpy(dtype=int),
            frame["probability"].to_numpy(dtype=float),
            threshold,
            replicates,
            base_seed + index,
        )
        for metric, values in intervals.items():
            wide.loc[index, f"test_{metric}_ci_low"] = values["ci_low"]
            wide.loc[index, f"test_{metric}_ci_high"] = values["ci_high"]
            long_rows.append(
                {
                    "representation": row["representation"],
                    "model": row["model"],
                    "metric": metric,
                    "estimate": values["estimate"],
                    "ci_low": values["ci_low"],
                    "ci_high": values["ci_high"],
                    "confidence_level": 0.95,
                    "method": "class-stratified percentile bootstrap",
                    "n_replicates": replicates,
                }
            )
    return wide, pd.DataFrame(long_rows)


def _predeclared_comparison(
    config: dict[str, Any],
    summary: pd.DataFrame,
    predictions: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    ranking = summary.sort_values(
        ["best_mean_cv_roc_auc", "best_std_cv_roc_auc"],
        ascending=[False, True],
    )
    champion = ranking.iloc[0]
    runner_up = ranking.iloc[1]
    key_a = (champion["representation"], champion["model"])
    key_b = (runner_up["representation"], runner_up["model"])
    frame_a = predictions[key_a]
    frame_b = predictions[key_b]
    rows: list[dict[str, Any]] = []
    for metric in ["roc_auc", "mcc"]:
        comparison = paired_bootstrap_difference(
            frame_a["label"].to_numpy(dtype=int),
            frame_a["probability"].to_numpy(dtype=float),
            frame_b["probability"].to_numpy(dtype=float),
            metric=metric,
            threshold=float(config["validation"]["classification_threshold"]),
            n_replicates=int(config["validation"]["bootstrap_replicates"]),
            seed=int(config["seeds"]["global"]),
        )
        rows.append(
            {
                "selection_basis": "top two by development 10-fold mean ROC-AUC",
                "model_a": f"{REPRESENTATION_LABELS[key_a[0]]}-{MODEL_LABELS[key_a[1]]}",
                "model_b": f"{REPRESENTATION_LABELS[key_b[0]]}-{MODEL_LABELS[key_b[1]]}",
                **comparison,
            }
        )
    return pd.DataFrame(rows)


def _export_figure(fig: plt.Figure, base: Path, dpi: int = 600) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix in ["svg", "pdf", "png", "tiff"]:
        path = base.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _figure4(
    config: dict[str, Any],
    predictions: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    figure_data_dir = output_dir / "source_data/Figure4"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    roc_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(
        5,
        6,
        figsize=(15.5, 11.6),
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1, 1, 1]},
    )
    for row_index, model_name in enumerate(FIGURE4_MODEL_ORDER):
        roc_ax = axes[row_index, 0]
        for representation in REPRESENTATIONS:
            frame = predictions[(representation, model_name)]
            fpr, tpr, thresholds = roc_curve(frame["label"], frame["probability"])
            auc = float(
                json.loads(
                    (
                        _results_root(config)
                        / "evaluation"
                        / representation
                        / model_name
                        / "metrics.json"
                    ).read_text(encoding="utf-8")
                )["locked_test_metrics"]["roc_auc"]
            )
            roc_ax.plot(
                fpr,
                tpr,
                color=COLORS[representation],
                linewidth=1.15,
                label=f"{REPRESENTATION_LABELS[representation]} (AUC={auc:.2f})",
            )
            roc_rows.extend(
                {
                    "model": model_name,
                    "representation": representation,
                    "fpr": float(x),
                    "tpr": float(y),
                    "threshold": float(t) if np.isfinite(t) else np.nan,
                }
                for x, y, t in zip(fpr, tpr, thresholds)
            )
        roc_ax.plot([0, 1], [0, 1], color="#9A9A9A", linestyle="--", linewidth=0.8)
        roc_ax.set_xlim(0, 1)
        roc_ax.set_ylim(0, 1.02)
        roc_ax.set_xlabel("False Positive Rate", fontsize=7, fontweight="bold")
        roc_ax.set_ylabel("True Positive Rate", fontsize=7, fontweight="bold")
        roc_ax.set_title(f"AUC Curves for {MODEL_LABELS[model_name]}", fontsize=8)
        roc_ax.tick_params(labelsize=6, length=2)
        roc_ax.legend(fontsize=5.2, loc="lower right", handlelength=1.8)

        for column_index, representation in enumerate(FIGURE4_MATRIX_ORDER, start=1):
            ax = axes[row_index, column_index]
            frame = predictions[(representation, model_name)]
            matrix = confusion_matrix(
                frame["label"], frame["prediction_0.5"], labels=[0, 1]
            )
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            image = ax.imshow(normalized, cmap="Blues", vmin=0.15, vmax=0.9)
            for actual in range(2):
                for predicted in range(2):
                    value = normalized[actual, predicted]
                    ax.text(
                        predicted,
                        actual,
                        f"{value * 100:.2f}%",
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        color="white" if value > 0.58 else "#1F1F1F",
                    )
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "representation": representation,
                            "actual": actual,
                            "predicted": predicted,
                            "count": int(matrix[actual, predicted]),
                            "row_percentage": float(value * 100),
                        }
                    )
            ax.set_xticks([0, 1], ["Non-toxic", "Toxic"], fontsize=5.8)
            ax.set_yticks([0, 1], ["Non-toxic", "Toxic"], fontsize=5.8)
            ax.set_xlabel("Predicted", fontsize=6.5, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=6.5, fontweight="bold")
            ax.set_title(
                f"{REPRESENTATION_LABELS[representation]}-{MODEL_LABELS[model_name]}",
                fontsize=7.1,
            )
            for spine in ax.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
            colorbar.ax.tick_params(labelsize=5.2, length=2)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.98, bottom=0.055, wspace=0.52, hspace=0.62)
    pd.DataFrame(roc_rows).to_csv(figure_data_dir / "roc_curves.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        figure_data_dir / "confusion_matrices.csv", index=False
    )
    outputs = _export_figure(fig, output_dir / "Figure4_ROC_confusion_all25")
    return {"outputs": outputs, "source_data": str(figure_data_dir)}


def _metric_value(row: pd.Series, metric: str) -> float:
    return float(row[f"test_{metric}"])


def _figure5(
    summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    figure_data_dir = output_dir / "source_data/Figure5"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figure_data_dir / "all25_metrics_with_ci.csv", index=False)
    panel_metrics = [
        ("A", "roc_auc", "AUC"),
        ("B", "precision", "Precision"),
        ("C", "f1", "F1"),
        ("D", "accuracy", "ACC"),
        ("E", "sensitivity", "SE"),
        ("F", "specificity", "SP"),
    ]
    fig = plt.figure(figsize=(12.5, 11.8))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.65], hspace=0.47, wspace=0.28)
    x = np.arange(len(FIGURE5_MODEL_ORDER))
    legend_locations = ["upper left", "upper right", "upper left", "upper left", "upper left", "upper right"]
    for panel_index, (letter, metric, ylabel) in enumerate(panel_metrics):
        ax = fig.add_subplot(grid[panel_index // 3, panel_index % 3])
        values_for_limits: list[float] = []
        for representation in REPRESENTATIONS:
            values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for model_name in FIGURE5_MODEL_ORDER:
                row = summary.loc[
                    summary["representation"].eq(representation)
                    & summary["model"].eq(model_name)
                ].iloc[0]
                value = _metric_value(row, metric)
                values.append(value)
                values_for_limits.append(value)
                if metric == "roc_auc":
                    ci_low = float(row["test_roc_auc_ci_low"])
                    ci_high = float(row["test_roc_auc_ci_high"])
                    lower_errors.append(value - ci_low)
                    upper_errors.append(ci_high - value)
                    values_for_limits.extend([ci_low, ci_high])
            ax.plot(
                x,
                values,
                color=COLORS[representation],
                marker=MARKERS[representation],
                markersize=4.2,
                linewidth=1.0,
                label=REPRESENTATION_LABELS[representation],
            )
            if metric == "roc_auc":
                ax.errorbar(
                    x,
                    values,
                    yerr=np.vstack([lower_errors, upper_errors]),
                    fmt="none",
                    ecolor=COLORS[representation],
                    elinewidth=0.6,
                    capsize=1.6,
                    alpha=0.55,
                )
        value_span = max(values_for_limits) - min(values_for_limits)
        lower_margin = max(0.012, value_span * 0.18)
        legend_headroom = max(0.018, value_span * 0.45)
        ax.set_ylim(
            max(0.0, min(values_for_limits) - lower_margin),
            min(1.0, max(values_for_limits) + legend_headroom),
        )
        ax.set_xticks(x, [MODEL_LABELS[name] for name in FIGURE5_MODEL_ORDER], fontsize=7.2)
        ax.set_xlabel("Models", fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.legend(
            fontsize=5.8,
            loc=legend_locations[panel_index],
            handlelength=1.35,
            handletextpad=0.35,
            labelspacing=0.22,
            borderaxespad=0.35,
            frameon=False,
        )

    heat_ax = fig.add_subplot(grid[2, :])
    heat_metrics = [
        ("roc_auc", "AUC"),
        ("accuracy", "ACC"),
        ("precision", "P"),
        ("sensitivity", "SE"),
        ("specificity", "SP"),
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("balanced_accuracy", "BA"),
    ]
    matrix_rows: list[list[float]] = []
    row_labels: list[str] = []
    heat_source_rows: list[dict[str, Any]] = []
    for model_name in FIGURE5_HEATMAP_MODEL_ORDER:
        for representation in FIGURE5_HEATMAP_REP_ORDER:
            row = summary.loc[
                summary["representation"].eq(representation)
                & summary["model"].eq(model_name)
            ].iloc[0]
            values = [_metric_value(row, metric) for metric, _ in heat_metrics]
            matrix_rows.append(values)
            label = f"{REPRESENTATION_LABELS[representation]}-{MODEL_LABELS[model_name]}"
            row_labels.append(label)
            heat_source_rows.append(
                {
                    "model_combination": label,
                    **{display: value for (_, display), value in zip(heat_metrics, values)},
                }
            )
    matrix = np.asarray(matrix_rows, dtype=float)
    image = heat_ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=float(matrix.min()), vmax=float(matrix.max()))
    norm = Normalize(vmin=float(matrix.min()), vmax=float(matrix.max()))
    cmap = plt.get_cmap("RdBu_r")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            r, g, b, _ = cmap(norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            heat_ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if luminance < 0.47 else "black",
            )
    heat_ax.set_xticks(
        range(len(heat_metrics)), [display for _, display in heat_metrics], rotation=90, fontsize=8
    )
    heat_ax.set_yticks(range(len(row_labels)), row_labels, fontsize=6.3, fontweight="bold")
    heat_ax.set_xlabel("Metrics", fontsize=9, fontweight="bold")
    heat_ax.set_title("Model Metrics Heatmap", fontsize=10, fontweight="bold", pad=8)
    heat_ax.text(-0.14, 1.02, "G", transform=heat_ax.transAxes, fontsize=14, fontweight="bold")
    heat_ax.tick_params(length=0)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.018, pad=0.018)
    colorbar.ax.tick_params(labelsize=7)
    pd.DataFrame(heat_source_rows).to_csv(
        figure_data_dir / "heatmap_metrics.csv", index=False
    )
    fig.subplots_adjust(left=0.24, right=0.95, top=0.97, bottom=0.06)
    outputs = _export_figure(fig, output_dir / "Figure5_performance_all25")
    return {"outputs": outputs, "source_data": str(figure_data_dir)}


def _bitvectors_from_binary_matrix(matrix: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    vectors: list[DataStructs.ExplicitBitVect] = []
    for row in matrix:
        vector = DataStructs.ExplicitBitVect(int(matrix.shape[1]))
        vector.SetBitsFromList([int(value) for value in np.flatnonzero(row)])
        vectors.append(vector)
    return vectors


def _safe_subset_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    if len(labels) == 0:
        return {"n": 0, "label_0": 0, "label_1": 0}
    output: dict[str, float | int] = {
        "n": int(len(labels)),
        "label_0": int(np.sum(labels == 0)),
        "label_1": int(np.sum(labels == 1)),
    }
    if len(np.unique(labels)) < 2:
        prediction = (probabilities >= threshold).astype(int)
        output.update(
            {
                "roc_auc": np.nan,
                "accuracy": float(np.mean(prediction == labels)),
                "balanced_accuracy": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "precision": np.nan,
                "f1": np.nan,
                "mcc": np.nan,
            }
        )
        return output
    metrics = calculate_metrics(labels, probabilities, threshold)
    output.update(metrics)
    return output


def _applicability_domain_analysis(
    config: dict[str, Any],
    summary: pd.DataFrame,
    predictions: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    source_dir = output_dir / "source_data/FigureS_AD"
    source_dir.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(
        project_path(config, config["run"]["development_file"])
    ).sort_values("row_index")
    test_features = pd.read_csv(
        project_path(config, config["run"]["test_features_file"])
    ).sort_values("row_index")
    test_labels = pd.read_csv(
        project_path(config, config["run"]["sealed_test_labels_file"])
    ).sort_values("row_index")
    morgan = load_feature_matrix(config, "morgan2048")
    development_rows = development["row_index"].to_numpy(dtype=int)
    test_rows = test_features["row_index"].to_numpy(dtype=int)
    development_vectors = _bitvectors_from_binary_matrix(morgan[development_rows])
    test_vectors = _bitvectors_from_binary_matrix(morgan[test_rows])

    development_loo_max = np.empty(len(development_vectors), dtype=float)
    development_loo_neighbor = np.empty(len(development_vectors), dtype=int)
    for index, vector in enumerate(development_vectors):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(vector, development_vectors), dtype=float
        )
        similarities[index] = -1.0
        neighbor = int(np.argmax(similarities))
        development_loo_neighbor[index] = neighbor
        development_loo_max[index] = similarities[neighbor]
    ad_threshold = float(np.quantile(development_loo_max, 0.05))

    test_max = np.empty(len(test_vectors), dtype=float)
    test_neighbor = np.empty(len(test_vectors), dtype=int)
    for index, vector in enumerate(test_vectors):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(vector, development_vectors), dtype=float
        )
        neighbor = int(np.argmax(similarities))
        test_neighbor[index] = neighbor
        test_max[index] = similarities[neighbor]

    development_loo = development[
        ["row_index", "compound_id", "name", "label", "standardized_smiles"]
    ].copy()
    development_loo["nearest_development_row_index"] = development.iloc[
        development_loo_neighbor
    ]["row_index"].to_numpy(dtype=int)
    development_loo["nearest_development_compound_id"] = development.iloc[
        development_loo_neighbor
    ]["compound_id"].to_numpy()
    development_loo["leave_one_out_max_morgan_tanimoto"] = development_loo_max
    development_loo["ad_threshold_5th_percentile"] = ad_threshold
    development_loo.to_csv(
        source_dir / "development_leave_one_out_similarity.csv", index=False
    )

    test_ad = test_features[
        ["row_index", "compound_id", "name", "standardized_smiles", "scaffold_group"]
    ].merge(test_labels, on=["row_index", "compound_id"], validate="one_to_one")
    nearest = development.iloc[test_neighbor]
    test_ad["nearest_development_row_index"] = nearest["row_index"].to_numpy(dtype=int)
    test_ad["nearest_development_compound_id"] = nearest["compound_id"].to_numpy()
    test_ad["nearest_development_name"] = nearest["name"].to_numpy()
    test_ad["nearest_development_label"] = nearest["label"].to_numpy(dtype=int)
    test_ad["max_morgan_tanimoto"] = test_max
    test_ad["ad_threshold_5th_percentile"] = ad_threshold
    test_ad["inside_applicability_domain"] = test_max >= ad_threshold
    bin_edges = [-0.001, 0.30, 0.50, 0.70, 1.001]
    bin_labels = ["<0.30", "0.30-<0.50", "0.50-<0.70", ">=0.70"]
    test_ad["similarity_bin"] = pd.cut(
        test_ad["max_morgan_tanimoto"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    ).astype(str)

    ranking = summary.sort_values(
        ["best_mean_cv_roc_auc", "best_std_cv_roc_auc"], ascending=[False, True]
    )
    primary = ranking.iloc[0]
    primary_key = (primary["representation"], primary["model"])
    test_best = summary.sort_values(
        ["test_roc_auc", "best_mean_cv_roc_auc"], ascending=[False, False]
    ).iloc[0]
    descriptive_key = (test_best["representation"], test_best["model"])
    model_specs = [
        ("primary", primary_key, "CV-selected primary model"),
        ("descriptive", descriptive_key, "Test-best descriptive model"),
    ]
    for prefix, model_key, role in model_specs:
        model_predictions = predictions[model_key][
            ["row_index", "compound_id", "probability", "prediction_0.5"]
        ].rename(
            columns={
                "probability": f"{prefix}_probability",
                "prediction_0.5": f"{prefix}_prediction_0.5",
            }
        )
        test_ad = test_ad.merge(
            model_predictions,
            on=["row_index", "compound_id"],
            validate="one_to_one",
        )
        test_ad[f"{prefix}_correct_prediction"] = (
            test_ad[f"{prefix}_prediction_0.5"].astype(int)
            == test_ad["label"].astype(int)
        )
        test_ad[f"{prefix}_normalized_threshold_distance"] = 2.0 * np.abs(
            test_ad[f"{prefix}_probability"] - 0.5
        )
        test_ad[f"{prefix}_model"] = (
            f"{REPRESENTATION_LABELS[model_key[0]]}-{MODEL_LABELS[model_key[1]]}"
        )
        test_ad[f"{prefix}_model_role"] = role
    test_ad["cv_selected_champion"] = test_ad["primary_model"]
    test_ad["test_best_descriptive_model"] = test_ad["descriptive_model"]
    test_ad.to_csv(source_dir / "test_applicability_domain.csv", index=False)

    threshold = float(config["validation"]["classification_threshold"])
    performance_rows: list[dict[str, Any]] = []
    for (representation, model_name), frame in predictions.items():
        aligned = test_ad[["row_index", "inside_applicability_domain", "similarity_bin"]].merge(
            frame[["row_index", "label", "probability"]], on="row_index", validate="one_to_one"
        )
        groups: list[tuple[str, pd.Series]] = [
            ("inside_AD", aligned["inside_applicability_domain"]),
            ("outside_AD", ~aligned["inside_applicability_domain"]),
        ]
        groups.extend(
            (f"similarity_{label}", aligned["similarity_bin"].eq(label))
            for label in bin_labels
        )
        for group_name, mask in groups:
            subset = aligned.loc[mask]
            metrics = _safe_subset_metrics(
                subset["label"].to_numpy(dtype=int),
                subset["probability"].to_numpy(dtype=float),
                threshold,
            )
            performance_rows.append(
                {
                    "representation": representation,
                    "model": model_name,
                    "group": group_name,
                    **metrics,
                }
            )
    ad_performance = pd.DataFrame(performance_rows)
    ad_performance.to_csv(source_dir / "ad_performance_all25.csv", index=False)

    # The AD panels contain two-line scientific labels and role subtitles.  A
    # slightly wider canvas plus explicit inter-panel padding prevents the
    # y-axis labels and titles from intruding into adjacent panels in PNG/PPT.
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.7))
    ax = axes[0, 0]
    for label, color, name in [(0, "#377EB8", "Reference-negative"), (1, "#E74C3C", "Nephrotoxic")]:
        values = test_ad.loc[test_ad["label"].eq(label), "max_morgan_tanimoto"]
        ax.hist(values, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.5, color=color, label=name)
    ax.axvline(ad_threshold, color="#272727", linestyle="--", linewidth=1.1, label=f"AD threshold={ad_threshold:.2f}")
    ax.set_xlabel("Maximum Morgan Tanimoto to development", fontweight="bold")
    ax.set_ylabel("Test compounds", fontweight="bold")
    ax.legend(fontsize=7)
    ax.text(-0.14, 1.03, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    scatter_specs = [
        (axes[0, 1], "primary", primary_key, "CV-selected primary model", "B"),
        (axes[0, 2], "descriptive", descriptive_key, "Test-best descriptive model", "C"),
    ]
    for ax, prefix, model_key, role, letter in scatter_specs:
        colors = np.where(
            test_ad[f"{prefix}_correct_prediction"], "#377EB8", "#E74C3C"
        )
        ax.scatter(
            test_ad["max_morgan_tanimoto"],
            test_ad[f"{prefix}_normalized_threshold_distance"],
            c=colors,
            s=16,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.axvline(ad_threshold, color="#272727", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Maximum Morgan Tanimoto to development", fontweight="bold")
        ax.set_ylabel(
            "Decision-threshold distance\n(2|p−0.5|)",
            fontsize=8.3,
            fontweight="bold",
            labelpad=2.0,
        )
        model_label = f"{REPRESENTATION_LABELS[model_key[0]]}-{MODEL_LABELS[model_key[1]]}"
        ax.set_title(f"{model_label}\n{role}", fontsize=8.4, fontweight="bold")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#377EB8", markeredgecolor="white", label="Correct"),
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#E74C3C", markeredgecolor="white", label="Incorrect"),
            ],
            fontsize=6.7,
            loc="lower right",
            frameon=False,
        )
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")

    performance_specs = [
        (axes[1, 0], primary_key, "CV-selected primary model", "D"),
        (axes[1, 1], descriptive_key, "Test-best descriptive model", "E"),
    ]
    positions = np.arange(len(bin_labels))
    for ax, model_key, role, letter in performance_specs:
        model_perf = ad_performance.loc[
            ad_performance["representation"].eq(model_key[0])
            & ad_performance["model"].eq(model_key[1])
            & ad_performance["group"].isin(
                [f"similarity_{label}" for label in bin_labels]
            )
        ].copy()
        model_perf["bin_order"] = model_perf["group"].map(
            {f"similarity_{label}": index for index, label in enumerate(bin_labels)}
        )
        model_perf = model_perf.sort_values("bin_order")
        ax.plot(positions, model_perf["roc_auc"], color="#E74C3C", marker="s", label="AUC")
        ax.plot(positions, model_perf["balanced_accuracy"], color="#377EB8", marker="o", label="BA")
        for x_value, n_value in zip(positions, model_perf["n"]):
            ax.text(x_value, 0.49, f"n={int(n_value)}", ha="center", va="top", fontsize=7)
        ax.set_xticks(positions, bin_labels)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("Maximum Tanimoto similarity bin", fontweight="bold")
        ax.set_ylabel("Performance", fontweight="bold")
        model_label = f"{REPRESENTATION_LABELS[model_key[0]]}-{MODEL_LABELS[model_key[1]]}"
        ax.set_title(f"{model_label}\n{role}", fontsize=8.4, fontweight="bold")
        ax.legend(fontsize=7)
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax = axes[1, 2]
    ad_groups = [False, True]
    labels_display = ["Outside AD", "Inside AD"]
    counts = [int(test_ad["inside_applicability_domain"].eq(value).sum()) for value in ad_groups]
    group_positions = np.arange(len(ad_groups))
    width = 0.34
    all_error_rates: list[float] = []
    for offset, (prefix, model_key, role, color) in enumerate(
        [
            ("primary", primary_key, "CV-selected primary model", "#377EB8"),
            ("descriptive", descriptive_key, "Test-best descriptive model", "#7B6FA6"),
        ]
    ):
        error_rates = [
            1.0
            - float(
                test_ad.loc[
                    test_ad["inside_applicability_domain"].eq(value),
                    f"{prefix}_correct_prediction",
                ].mean()
            )
            for value in ad_groups
        ]
        all_error_rates.extend(error_rates)
        bars = ax.bar(
            group_positions + (offset - 0.5) * width,
            error_rates,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=(
                f"{REPRESENTATION_LABELS[model_key[0]]}-{MODEL_LABELS[model_key[1]]}\n{role}"
            ),
        )
        for bar, rate, count in zip(bars, error_rates, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.012,
                f"{rate:.2f}\nn={count}",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )
    ax.set_xticks(group_positions, labels_display)
    ax.set_ylim(0, min(1.0, max(all_error_rates + [0.1]) + 0.17))
    ax.set_ylabel("Classification error rate", fontweight="bold")
    ax.legend(fontsize=6.3, loc="upper right", frameon=False)
    ax.text(-0.14, 1.03, "F", transform=ax.transAxes, fontsize=14, fontweight="bold")

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    fig.tight_layout(pad=1.7, w_pad=3.0, h_pad=2.0)
    outputs = _export_figure(fig, output_dir / "FigureS_AD_Tanimoto_applicability")
    summary_payload = {
        "fingerprint": "Morgan radius 2, 2048 bits, chirality enabled",
        "similarity": "Tanimoto",
        "ad_definition": "inside if test maximum similarity is at least the 5th percentile of development leave-one-out maxima",
        "ad_threshold": ad_threshold,
        "development_loo_median": float(np.median(development_loo_max)),
        "test_similarity_median": float(np.median(test_max)),
        "test_similarity_iqr": [float(np.quantile(test_max, 0.25)), float(np.quantile(test_max, 0.75))],
        "n_inside_ad": int(np.sum(test_max >= ad_threshold)),
        "n_outside_ad": int(np.sum(test_max < ad_threshold)),
        "normalized_threshold_distance_definition": "2 * abs(predicted probability - 0.5); 0 denotes the decision boundary and 1 denotes an extreme probability",
        "cv_selected_champion": test_ad["cv_selected_champion"].iloc[0],
        "cv_selected_champion_role": "CV-selected primary model",
        "test_best_descriptive_model": test_ad["test_best_descriptive_model"].iloc[0],
        "test_best_descriptive_model_role": "Test-best descriptive model; reported descriptively and not used to revise model selection",
        "outputs": outputs,
        "source_data": str(source_dir),
    }
    write_json(summary_payload, source_dir / "ad_summary.json")
    return summary_payload


def _write_results_markdown(summary: pd.DataFrame, comparisons: pd.DataFrame, path: Path) -> None:
    ranking = summary.sort_values(
        ["best_mean_cv_roc_auc", "best_std_cv_roc_auc"], ascending=[False, True]
    )
    lines = [
        "# Scaffold-aware 10-fold modeling results",
        "",
        "All 25 representation-model combinations used the same 1,482-compound development set. Hyperparameters were selected by development-only 10-fold grouped CV before the 371-compound test set was opened once.",
        "",
        "## Ranking fixed before test evaluation",
        "",
        "| Rank | Combination | Development CV AUC | Test AUC (95% CI) | ACC | SE | SP | Precision | F1 | MCC | BA |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
        combination = f"{row['representation_label']}-{row['model_label']}"
        lines.append(
            f"| {rank} | {combination} | {row['best_mean_cv_roc_auc']:.3f} +/- {row['best_std_cv_roc_auc']:.3f} | "
            f"{row['test_roc_auc']:.3f} ({row['test_roc_auc_ci_low']:.3f}-{row['test_roc_auc_ci_high']:.3f}) | "
            f"{row['test_accuracy']:.3f} | {row['test_sensitivity']:.3f} | {row['test_specificity']:.3f} | "
            f"{row['test_precision']:.3f} | {row['test_f1']:.3f} | {row['test_mcc']:.3f} | {row['test_balanced_accuracy']:.3f} |"
        )
    lines.extend(["", "## Predeclared paired comparison", ""])
    for _, row in comparisons.iterrows():
        lines.append(
            f"- {row['metric']}: {row['model_a']} minus {row['model_b']} = "
            f"{row['difference_a_minus_b']:.3f} (95% CI {row['ci_low']:.3f} to {row['ci_high']:.3f}; "
            f"paired-bootstrap P={row['p_value_two_sided']:.4f})."
        )
    lines.extend(
        [
            "",
            "Model selection and ranking were not revised after viewing the test outcomes. All threshold metrics use the fixed probability threshold of 0.5.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_locked_results(config: dict[str, Any]) -> pd.DataFrame:
    _require_completed_evaluation(config)
    results_root = _results_root(config)
    summary_dir = results_root / "summary"
    figure_dir = results_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _capture_environment(results_root)
    summary, predictions = _collect_results(config)
    summary.to_csv(summary_dir / "all_metrics.csv", index=False)
    summary_with_ci, ci_long = _bootstrap_all(config, summary, predictions)
    summary_with_ci.to_csv(summary_dir / "all_metrics_with_ci.csv", index=False)
    ci_long.to_csv(summary_dir / "confidence_intervals_long.csv", index=False)
    ranking = summary_with_ci.sort_values(
        ["best_mean_cv_roc_auc", "best_std_cv_roc_auc"], ascending=[False, True]
    )
    ranking.to_csv(summary_dir / "ranking_by_development_cv.csv", index=False)
    comparisons = _predeclared_comparison(config, summary_with_ci, predictions)
    comparisons.to_csv(summary_dir / "paired_comparisons.csv", index=False)
    figure_manifest = {
        "figure_contract": {
            "core_conclusion": (
                "Compare all 25 locked models on an independent stratified-random "
                "test set after development-only 10-fold selection."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/Matplotlib",
            "style_reference": "original manuscript Figure 4 and Figure 5",
            "primary_exports": ["SVG", "PDF"],
            "raster_export": "TIFF and PNG at 600 dpi",
        },
        "Figure4": _figure4(config, predictions, figure_dir),
        "Figure5": _figure5(summary_with_ci, figure_dir),
        "FigureS_AD": _applicability_domain_analysis(
            config, summary_with_ci, predictions, figure_dir
        ),
        "data_tables": export_split_tables(config),
        "supplementary_tables": export_supplementary_tables(
            config, summary_with_ci
        ),
    }
    write_json(figure_manifest, figure_dir / "figure_manifest.json")
    _write_results_markdown(
        summary_with_ci, comparisons, results_root / "RESULTS.md"
    )
    return summary_with_ci
