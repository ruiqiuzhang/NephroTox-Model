from __future__ import annotations

import json

from nephrotox.config import load_config
from nephrotox.locked_validation import validate_locked_outputs


if __name__ == "__main__":
    print(json.dumps(validate_locked_outputs(load_config()), indent=2, ensure_ascii=False))
