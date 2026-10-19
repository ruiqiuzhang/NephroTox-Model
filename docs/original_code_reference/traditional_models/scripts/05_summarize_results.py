from __future__ import annotations

from nephrotox.config import load_config
from nephrotox.locked_reporting import summarize_locked_results


if __name__ == "__main__":
    frame = summarize_locked_results(load_config())
    print(frame.to_string(index=False))
