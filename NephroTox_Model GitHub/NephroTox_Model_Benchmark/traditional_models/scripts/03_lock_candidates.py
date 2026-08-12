from __future__ import annotations

import json

from nephrotox.config import load_config
from nephrotox.locked_pipeline import lock_candidates


if __name__ == "__main__":
    print(json.dumps(lock_candidates(load_config()), indent=2, ensure_ascii=False))
