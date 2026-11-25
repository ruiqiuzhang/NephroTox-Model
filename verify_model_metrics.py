"""Recalculate the reported ROC-AUC, AP, Brier score and confusion counts from probabilities."""

import argparse
import csv
from collections import Counter, defaultdict
from itertools import groupby
import math
from pathlib import Path
import sys
from zipfile import BadZipFile


REPRESENTATIONS = (
    "morgan2048", "maccs166", "atompair1024", "torsion1024", "rdkit166",
    "morgan_count2048", "atompair_count1024", "torsion_count1024",
)
MODELS = {f"{rep}__{alg}" for rep in REPRESENTATIONS for alg in ("rf", "svm", "mlp", "xgb", "dnn")}
MODELS.update(("graph__DMPNN", "graph__GINE", "graph__AttentiveFP"))
PARTITION_SIZE = {"development_oof": 1338, "test": 335}
FOLDS = set(range(1, 11))
TOLERANCE = 1e-12
METRIC_COLUMNS = {
    "ROC_AUC": "ROC-AUC", "AP": "Average precision", "Brier": "Brier score",
    "n": "n", "TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        headers = rows.fieldnames or []
        require(len(set(headers)) == len(headers), f"{path}: duplicate column header")
        yield from rows


def split_predictions(path, partition):
    by_model = defaultdict(dict)
    labels_and_folds = {}
    for line, row in enumerate(read_csv(path), 2):
        where = f"{path.name} line {line}"
        model, compound = row["model_id"], row["compound_id"]
        require(model in MODELS, f"{where}: unexpected model {model!r}")
        require(compound and compound not in by_model[model], f"{where}: missing or duplicate ID {compound!r} for {model}")
        label = int(row["label"])
        score = float(row["probability"])
        require(label in (0, 1) and 0 <= score <= 1, f"{where}: invalid label or probability")
        require(float(row["threshold"]) == 0.5, f"{where}: expected threshold 0.5")
        require(int(row["predicted_label"]) == int(score >= 0.5), f"{where}: predicted label disagrees with probability")
        fold = None
        if partition == "development_oof":
            fold = int(row["cv_validation_fold"])
            require(fold in FOLDS, f"{where}: invalid validation fold")
        else:
            require(row["cv_validation_fold"] == "", f"{where}: test compound has a validation fold")
        if compound in labels_and_folds:
            require(labels_and_folds[compound] == (label, fold), f"{where}: inconsistent label/fold for {compound}")
        labels_and_folds[compound] = (label, fold)
        by_model[model][compound] = score

    require(set(by_model) == MODELS, f"{path}: expected all {len(MODELS)} manuscript models")
    require(len(labels_and_folds) == PARTITION_SIZE[partition], f"{path}: unexpected number of unique compounds")
    require({fold for _, fold in labels_and_folds.values()} == (FOLDS if partition == "development_oof" else {None}),
            f"{path}: incomplete or unexpected folds")
    for model, scores in by_model.items():
        require(scores.keys() == labels_and_folds.keys(), f"{path}: compound coverage differs for {model}")
    return by_model, labels_and_folds


def calculate(observations):
    # Group tied scores for both the ROC trapezoids and the average-precision steps.
    observations = sorted(observations, reverse=True)
    positives = sum(label for _, label in observations)
    negatives = len(observations) - positives
    require(positives and negatives, "cannot score a partition or fold with a missing class")
    prior_positives = prior_total = 0
    concordant = ap_sum = 0.0
    for _, tied in groupby(observations, key=lambda item: item[0]):
        labels = [label for _, label in tied]
        group_positives = sum(labels)
        group_negatives = len(labels) - group_positives
        concordant += group_negatives * (prior_positives + 0.5 * group_positives)
        prior_positives += group_positives
        prior_total += len(labels)
        ap_sum += group_positives * prior_positives / prior_total
    cells = Counter((label, int(score >= 0.5)) for score, label in observations)
    return {
        "n": len(observations), "ROC_AUC": concordant / (positives * negatives),
        "AP": ap_sum / positives, "Brier": math.fsum((p - y) ** 2 for p, y in observations) / len(observations),
        "TN": cells[0, 0], "FP": cells[0, 1], "FN": cells[1, 0], "TP": cells[1, 1],
    }


def calculate_tables(directory):
    partitions, folds = {}, {}
    prior_ids = set()
    for partition, filename in (("development_oof", "oof_predictions.csv"), ("test", "test_predictions.csv")):
        models, metadata = split_predictions(directory / filename, partition)
        require(prior_ids.isdisjoint(metadata), "a compound is present in both development and test")
        prior_ids.update(metadata)
        for model, scores in models.items():
            observations = []
            per_fold = defaultdict(list)
            for compound, probability in scores.items():
                label, fold = metadata[compound]
                item = (probability, label)
                observations.append(item)
                if fold is not None:
                    per_fold[fold].append(item)
            partitions[model, partition] = calculate(observations)
            for fold, items in per_fold.items():
                folds[model, fold] = calculate(items)
    return partitions, folds


def check_rows(rows, expected, source, by_fold, workbook=False):
    model_column = "model id" if workbook else "model_id"
    seen = set()
    largest = 0.0
    for line, row in enumerate(rows, 2):
        key = (row[model_column], int(row["fold"]) if by_fold else row["partition"])
        require(key in expected and key not in seen, f"{source} row {line}: duplicate or unexpected key {key}")
        seen.add(key)
        for metric, excel_column in METRIC_COLUMNS.items():
            value = float(row[excel_column if workbook else metric])
            target = expected[key][metric]
            delta = abs(value - target)
            require(math.isfinite(value) and delta <= TOLERANCE,
                    f"{source}: {key} {metric}: reported {value}, calculated {target}")
            largest = max(largest, delta)
    require(seen == set(expected), f"{source}: missing {len(set(expected) - seen)} rows")
    return largest


def workbook_rows(sheet):
    rows = sheet.values
    headers = next(rows)
    require(len(set(headers)) == len(headers), f"{sheet.title}: duplicate header")
    for record in rows:
        if any(item is not None for item in record):
            yield dict(zip(headers, record))


def run(args):
    partitions, folds = calculate_tables(args.reports)
    max_difference = 0.0
    tables = [
        ("model_summary.csv", "TableS7B All Performance", partitions, False),
        ("perfold_metrics.csv", "TableS3A Fold Metrics", folds, True),
    ]
    for filename, _, expected, by_fold in tables:
        path = args.reports / filename
        max_difference = max(max_difference, check_rows(read_csv(path), expected, filename, by_fold))
    if args.workbook:
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError("install openpyxl to use --workbook") from exc
        book = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
        try:
            for _, sheet, expected, by_fold in tables:
                difference = check_rows(workbook_rows(book[sheet]), expected, sheet, by_fold, workbook=True)
                max_difference = max(max_difference, difference)
        finally:
            book.close()
    print(f"Verified {len(MODELS)} models, {len(folds)} fold rows and {len(partitions)} partition rows.")
    print(f"Checked ROC-AUC, average precision, Brier score, sample count and TN/FP/FN/TP. "
          f"Maximum absolute difference: {max_difference:.3g}.")
    if args.workbook:
        print("Submission workbook tables S3A and S7B match the recalculated metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reference/modeling/reports"),
                        help="directory of the four reference CSVs (default: reference/modeling/reports)")
    parser.add_argument("--workbook", type=Path, help="also check tables S3A and S7B in the submission XLSX")
    try:
        run(parser.parse_args())
    except (ValueError, OSError, csv.Error, KeyError, TypeError, BadZipFile) as error:
        sys.exit(f"Verification failed: {error}")

