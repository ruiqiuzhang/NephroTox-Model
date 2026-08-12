import pandas as pd

from nephrotox.config import load_config, project_path
from nephrotox.data import load_and_validate_dataset, make_random_split


def test_formal_random10_split_invariants() -> None:
    config = load_config()
    data, _ = load_and_validate_dataset(config)
    assignment = make_random_split(data, config).sort_values("row_index").reset_index(drop=True)

    assert len(assignment) == 1853
    assert set(assignment["partition"]) == {"development", "test"}
    development = assignment.loc[assignment["partition"] == "development"]
    test = assignment.loc[assignment["partition"] == "test"]
    assert len(development) == 1482
    assert len(test) == 371
    assert development["label"].value_counts().sort_index().to_dict() == {0: 737, 1: 745}
    assert test["label"].value_counts().sort_index().to_dict() == {0: 185, 1: 186}
    assert set(development["fold"]) == set(range(10))
    assert set(test["fold"]) == {-1}

    locked = pd.read_csv(project_path(config, config["run"]["split_file"]))
    pd.testing.assert_frame_equal(
        assignment.drop(columns=["label"]).reset_index(drop=True),
        locked.reset_index(drop=True),
        check_dtype=False,
    )

