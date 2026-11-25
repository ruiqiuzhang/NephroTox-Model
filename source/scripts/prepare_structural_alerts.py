"""Prepare independent development-only BRENK/NIH/ZINC/PAINS source data."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nephrotox.data import load_data
from nephrotox.structural_alerts import export_structural_alerts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, source = load_data()
    split = args.results / "split_payload.json"
    assignments = json.loads(split.read_text(encoding="utf-8"))["assignments"]
    protocol = args.results / "modeling/protocol.json"
    output = args.output or args.results / "modeling/structural_alerts_catalog"
    manifest = export_structural_alerts(rows, assignments, source, split, output,
                                       protocol_path=protocol if protocol.exists() else None)
    print(json.dumps({"status": manifest["status"], "output": str(output), "n_development": manifest["n_development"],
                      "bh_family_size": manifest["bh_family_size"], "n_display_significant": manifest["n_display_significant"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
