from __future__ import annotations

import argparse

from nephrotox.config import load_config
from nephrotox.locked_pipeline import MODELS, REPRESENTATIONS, tune_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune NephroTox models using development-only 10-fold CV."
    )
    parser.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=REPRESENTATIONS
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the first candidate only and write to the isolated smoke directory.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute selected development-only runs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_experiments(
        load_config(),
        representations=args.representations,
        models=args.models,
        quick=args.quick,
        force=args.force,
    )
