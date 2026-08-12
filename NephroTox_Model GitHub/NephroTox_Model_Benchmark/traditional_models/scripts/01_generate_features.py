from __future__ import annotations

import json

from nephrotox.config import load_config
from nephrotox.features import generate_features


if __name__ == "__main__":
    print(json.dumps(generate_features(load_config()), indent=2, ensure_ascii=False))

