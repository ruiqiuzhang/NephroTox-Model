from __future__ import annotations

import argparse

from nephrotox.config import load_config
from nephrotox.data import prepare_locked_random_data
from nephrotox.features import generate_features
from nephrotox.locked_pipeline import (
    MODELS,
    REPRESENTATIONS,
    evaluate_locked_test,
    lock_candidates,
    tune_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the random10 pipeline through candidate locking."
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After locking, execute the one-time independent-test evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = load_config()
    prepare_locked_random_data(configuration)
    generate_features(configuration)
    tune_experiments(
        configuration,
        representations=REPRESENTATIONS,
        models=MODELS,
    )
    lock_candidates(configuration)
    if args.evaluate_test:
        evaluate_locked_test(configuration)
    else:
        print(
            "Candidates are locked. Run scripts/04_evaluate_locked_test.py once "
            "when final independent-test evaluation is authorized."
        )
