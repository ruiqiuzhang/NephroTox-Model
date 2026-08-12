from __future__ import annotations

import pandas as pd

from nephrotox.config import load_config, project_path
from nephrotox.supplementary import export_split_tables, export_supplementary_tables


if __name__ == "__main__":
    config = load_config()
    results_root = project_path(config, config["run"]["results_root"])
    summary = pd.read_csv(results_root / "summary/all_metrics_with_ci.csv")
    print(export_split_tables(config))
    print(export_supplementary_tables(config, summary))
