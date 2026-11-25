"""Represent input-file identity consistently across CSV and archived workbook runs."""
import json
from pathlib import Path


def source_file_fields(record):
    """Return provenance fields for a run, output manifest or validated input."""
    if "source_format" in record:
        digest = record.get("source_file_sha256") or record["sha256"]
        return {"source_file": record["source_file"], "source_format": record["source_format"],
                "source_file_sha256": digest}
    return {"source_file": "data/NephroTox_Modeling_Data.xlsx", "source_format": "xlsx",
            "source_file_sha256": record["source_excel_sha256"]}


def verify_input_identity(record, source, source_directory):
    """Accept the current CSV or its documented, row-identical archived input."""
    if record["source_rows_fingerprint"] != source["rows_fingerprint"]:
        raise ValueError("Compound records differ from the recorded run")
    identity = source_file_fields(record)
    if identity == source_file_fields(source):
        return
    provenance = json.loads((Path(source_directory) / "data/input_provenance.json").read_bytes())
    if (identity["source_format"] != "xlsx"
            or identity["source_file_sha256"] not in
            (provenance["archived_workbook_sha256"], provenance["distributed_workbook_sha256"])
            or source["sha256"] != provenance["distributed_csv_sha256"]
            or source["rows_fingerprint"] != provenance["source_rows_fingerprint"]):
        raise ValueError("Modeling input file or scientific row fingerprint differs from the recorded run")
