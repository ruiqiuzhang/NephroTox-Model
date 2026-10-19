from __future__ import annotations

import json

from nephrotox.config import load_config
from nephrotox.locked_pipeline import evaluate_locked_test


if __name__ == "__main__":
    print(json.dumps(evaluate_locked_test(load_config()), indent=2, ensure_ascii=False))
