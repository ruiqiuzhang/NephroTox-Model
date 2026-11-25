"""Check the manuscript repository, the supplementary workbook, or a fresh primary fit."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
REF = ROOT / "reference"
MODELS = REF / "modeling"
INPUT = SOURCE / "data/NephroTox_Modeling_Data.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        hasher = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def modeling_rows():
    source = read_json(REF / "split_payload.json")["manifest"]["source"]
    with INPUT.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        require(reader.fieldnames == source["source_columns"], "CSV column names or order differ")
        rows = []
        for row in reader:
            require(None not in row and all(row.values()), f"Incomplete CSV record on line {reader.line_num}")
            for column in ("PubChem_CID", "label", "heavy_atoms"):
                row[column] = int(row[column])
            row["MW"] = float(row["MW"])
            rows.append(row)
    require(len(rows) == source["n_rows"] == 1673, "Unexpected number of CSV records")
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(sha256(encoded.encode("utf-8")).hexdigest() == source["rows_fingerprint"]
            == read_json(MODELS / "protocol.json")["source_rows_fingerprint"],
            "Ordered, typed CSV records differ from the original training data")
    return rows


def verify_package():
    count = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text("utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(digest(ROOT / name) == expected, f"File checksum differs: {name}")
        count += 1

    protocol = read_json(MODELS / "protocol.json")
    lock = read_json(MODELS / "model_lock.json")
    require(digest(MODELS / "protocol.json") == lock["protocol_sha256"]
            == read_json(MODELS / "protocol_freeze.json")["protocol_sha256"], "Protocol freeze differs")
    require(digest(REF / "split_payload.json") == protocol["split_payload_sha256"], "Split payload differs")
    input_dir = SOURCE / "data"
    provenance = read_json(input_dir / "input_provenance.json")
    require(provenance["archived_workbook_sha256"] == protocol["source_excel_sha256"],
            "Archived input differs from the training protocol")
    manifest = read_json(input_dir / "source_manifest.json")
    require(manifest["source_file"] == provenance["source_file"] == INPUT.relative_to(SOURCE).as_posix()
            and manifest["source_format"] == provenance["source_format"] == "csv", "Expected the release CSV")
    require(digest(INPUT) == provenance["distributed_csv_sha256"] == manifest["sha256"], "CSV checksum differs")
    require(manifest["rows_fingerprint"] == provenance["source_rows_fingerprint"] == protocol["source_rows_fingerprint"],
            "Input row fingerprint differs from the training protocol")
    rows = modeling_rows()
    modules = read_json(input_dir / "fitting_source_provenance.json")
    original = protocol["fitting_source_sha256"]
    distributed = modules["distributed_fitting_source_sha256"]
    require(modules["archived_fitting_source_sha256"] == original and distributed.keys() == original.keys(),
            "Training module inventory differs from the archived protocol")
    changed = {name for name in original if original[name] != distributed[name]}
    require(changed == {"modeling_common.py", "modeling_pipeline.py"}, "Unexpected changed training modules")
    snapshots = modules["archived_module_snapshots"]
    require(set(snapshots) == changed, "Archived training module snapshots differ from the changed modules")
    for name, relative_path in snapshots.items():
        require(relative_path == f"reference/fitting_source/{name}", f"Unexpected snapshot path: {name}")
        require(digest(ROOT / relative_path) == original[name], f"Archived fitting source differs: {name}")
    for name, expected in distributed.items():
        require(digest(SOURCE / "src/nephrotox" / name) == expected, f"Fitting source differs: {name}")
    for key in ("fixed_parameter", "parameter_design", "search_design"):
        require(digest(SOURCE / protocol[f"{key}_file"]) == protocol[f"{key}_sha256"],
                f"Parameter source differs: {key}")
    for model in lock["models"]:
        directory = MODELS / model["model_id"]
        for name, key in (("model.joblib", "model_sha256"), ("metadata.json", "metadata_sha256"),
                          ("predictions.npz", "predictions_sha256")):
            require(digest(directory / name) == model[key], f"Model lock differs: {directory.name}/{name}")
    require(len(lock["models"]) == 43, "Unexpected number of models")
    print(f"Verified {count} repository files, {len(rows):,} ordered CSV records, the archived protocol and "
          "split, parameter designs, all seven original fitting modules and both CSV provenance updates, "
          "and 43 archived model/metadata/prediction triples.")


def verify_workbook(path):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        assignments = {row["compound_id"]: row for row in read_csv(REF / "split_assignments.csv")}
        inputs = modeling_rows()
        actual = list(workbook["Data1 Modeling Dataset"].values)[1:]
        require(len(inputs) == len(actual) == len(assignments) == 1673, "Data1 length differs")
        for source_row, data_row in zip(inputs, actual):
            cid = data_row[0]
            require(tuple(source_row.values()) == data_row[:12], f"Input row differs: {cid}")
            assignment = assignments[cid]
            expected_fold = assignment["cv_validation_fold"]
            require((data_row[12], str(data_row[13] or ""), data_row[14])
                    == (assignment["dataset_partition"], expected_fold, assignment["structure_group_id"]),
                    f"Data1 partition/fold/group differs: {cid}")

        max_diff = 0.0
        count = 0
        model_hashes = {m["model_id"]: m["model_sha256"] for m in read_json(MODELS / "model_lock.json")["models"]}
        for sheet, name in (("Data4 OOF Probabilities", "oof_predictions.csv"),
                            ("Data5 Test Probabilities", "test_predictions.csv")):
            reference = {(r["compound_id"], r["model_id"]): float(r["probability"])
                         for r in read_csv(MODELS / "reports" / name)}
            rows = iter(workbook[sheet].values)
            models = next(rows)[7:]
            require(set(models) == model_hashes.keys() and len(models) == 43, f"Model header differs: {sheet}")
            for row in rows:
                for model, value in zip(models, row[7:]):
                    expected = reference.pop((row[0], model))
                    difference = abs(expected - value)
                    require(math.isfinite(difference) and difference <= 1e-12,
                            f"Probability differs: {sheet}, {row[0]}, {model}")
                    max_diff = max(max_diff, difference)
                    count += 1
            require(not reference, f"Missing predictions in {sheet}")
        require(count == 71939, "Unexpected probability count")

        schema = read_json(SOURCE / "configs/graph_feature_schema.json")
        expected = {(family, i, name) for family in ("atom", "bond")
                    for i, name in enumerate(schema[f"{family}_features"])}
        reported = {(r[0], r[1], r[2]) for r in list(workbook["TableS2B Graph Features"].values)[1:]}
        require(reported == expected and len(reported) == 254, "S2B ordered graph vocabulary differs")
        reported_hashes = {(r[0], r[7]) for r in list(workbook["TableS8C Inference Repeats"].values)[1:]}
        require(reported_hashes == set(model_hashes.items()), "S8C model hashes differ")

        records = {(r[-2], r[-1]) for r in list(workbook["TableS9A Selected Epochs"].values)[1:]}
        require(len(records) == 121, "Unexpected number of neural fit records")
        for name, expected_digest in records:
            require(digest(MODELS / name) == expected_digest, f"S9A fit record differs: {name}")
        print(f"Workbook verified: 1,673 original rows and assignments, {count:,} probabilities "
              f"(maximum absolute difference {max_diff:.3g}), 254 ordered graph features, "
              "43 final model file hashes, and 121 neural fit hashes.")
    finally:
        workbook.close()


def check_rf(path):
    import numpy as np

    model = "maccs166__rf"
    for filename in ("split_assignments.csv", "cross_validation_folds.csv"):
        require(digest(path / filename) == digest(REF / filename), f"RF refit used different assignments: {filename}")
    run_protocol = read_json(path / "modeling/protocol.json")
    archived = read_json(MODELS / "protocol.json")
    provenance = read_json(SOURCE / "data/input_provenance.json")
    require(run_protocol["source_rows_fingerprint"] == archived["source_rows_fingerprint"],
            "RF refit used different compound records")
    require(run_protocol.get("source_file") == provenance["source_file"] == INPUT.relative_to(SOURCE).as_posix()
            and run_protocol.get("source_format") == "csv"
            and run_protocol.get("source_file_sha256") == provenance["distributed_csv_sha256"] == digest(INPUT),
            "RF refit did not use the distributed CSV")
    modules = read_json(SOURCE / "data/fitting_source_provenance.json")["distributed_fitting_source_sha256"]
    require(run_protocol["fitting_source_sha256"] == modules, "RF refit used different training modules")
    require(run_protocol["split_payload_sha256"] == digest(path / "split_payload.json"),
            "RF refit split differs from its recorded protocol")
    require(read_json(path / "parameter_mode.json")["parameter_mode"] == run_protocol["parameter_mode"]
            == "fixed_original" and run_protocol["split_mode"] == "legacy_random",
            "RF refit used a different parameter mode")
    for key in ("fixed_parameter_sha256", "fixed_parameter_records", "execution_seed_rule"):
        require(run_protocol[key] == archived[key], f"RF refit setting differs: {key}")
    fit_dir = path / "modeling" / model
    fit = read_json(fit_dir / "metadata.json")
    require(fit["complete"] and fit["model_id"] == model and
            fit["protocol_sha256"] == digest(path / "modeling/protocol.json")
            == read_json(path / "modeling/protocol_freeze.json")["protocol_sha256"],
            "RF fit metadata does not match its frozen run")
    require(fit["predictions_sha256"] == digest(fit_dir / "predictions.npz")
            and fit["model_sha256"] == digest(fit_dir / "model.joblib"), "RF fit artifact hashes differ")
    fit_scopes = [f"outer_{fold:02d}/refit" for fold in range(1, 11)] + ["final_fit"]
    for scope in fit_scopes:
        observed = read_json(fit_dir / scope / "fit_metadata.json")
        expected = read_json(MODELS / model / scope / "fit_metadata.json")
        for key in ("model", "representation", "params", "seed", "training_ids", "split_mode"):
            require(observed[key] == expected[key], f"RF refit differs in {scope}: {key}")
    with (np.load(path / "modeling" / model / "predictions.npz", allow_pickle=False) as run,
          np.load(MODELS / model / "predictions.npz", allow_pickle=False) as reference):
        for key in ("compound_id", "development_index", "test_index"):
            require(np.array_equal(reference[key], run[key]), f"RF refit IDs/order differ: {key}")
        for key in ("oof_probability", "test_probability"):
            require(run[key].shape == reference[key].shape, f"RF refit shape differs: {key}")
            delta = float(np.max(np.abs(run[key] - reference[key])))
            require(math.isfinite(delta) and delta <= 1e-12, f"RF refit probability differs: {key}: {delta}")
            print(f"{key}: {len(run[key])} values match; maximum absolute difference {delta:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, help="Also compare the supplied supplementary workbook")
    parser.add_argument("--check-rf", type=Path, metavar="RESULTS", help="Also compare a fresh MACCS–RF run")
    args = parser.parse_args()
    verify_package()
    if args.workbook:
        verify_workbook(args.workbook)
    if args.check_rf:
        check_rf(args.check_rf)


if __name__ == "__main__":
    main()
