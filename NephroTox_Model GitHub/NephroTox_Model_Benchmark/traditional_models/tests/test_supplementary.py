from __future__ import annotations

import pandas as pd

from nephrotox.config import load_config, project_path
from nephrotox.supplementary import export_split_tables, export_supplementary_tables


def test_public_split_and_supplementary_tables() -> None:
    config = load_config()
    split_outputs = export_split_tables(config)
    assert split_outputs["dataset_split_rows"] == 1853
    assert split_outputs["cv_folds_rows"] == 1482

    dataset_split = pd.read_csv(project_path(config, "data/splits/dataset_split.csv"))
    cv_folds = pd.read_csv(project_path(config, "data/splits/cv_folds.csv"))
    assert dataset_split["dataset_split"].value_counts().to_dict() == {
        "development": 1482,
        "independent_test": 371,
    }
    assert set(cv_folds["cv_fold"]) == set(range(1, 11))

    results_root = project_path(config, config["run"]["results_root"])
    summary = pd.read_csv(results_root / "summary/all_metrics_with_ci.csv")
    export_supplementary_tables(config, summary)
    output_dir = results_root / "supplementary_tables"
    assert len(pd.read_csv(output_dir / "Supplementary_Table_S1.csv")) == 25
    assert len(pd.read_csv(output_dir / "Supplementary_Table_S2A_search_spaces.csv")) == 15
    assert len(pd.read_csv(output_dir / "Supplementary_Table_S2B_optimized.csv")) == 25
