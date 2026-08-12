from __future__ import annotations

import json

from nephrotox.config import load_config
from nephrotox.data import prepare_locked_random_data


if __name__ == "__main__":
    print(
        json.dumps(
            prepare_locked_random_data(load_config()), indent=2, ensure_ascii=False
        )
    )
