"""Evaluate a captured real-robot handoff tuple without publishing commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from task2_real.runtime_core import (
    CalibrationError,
    evaluate_handoff,
    validate_site_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--now", type=float)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    try:
        validate_site_profile(profile, require_verified=True)
        result = evaluate_handoff(
            profile,
            snapshot,
            now=time.monotonic() if args.now is None else args.now,
        )
    except (CalibrationError, KeyError, TypeError, ValueError) as error:
        result = {
            "ready": False,
            "reasons": ["profile_or_snapshot_invalid"],
            "error": str(error),
            "command_publications": 0,
        }
    else:
        result["command_publications"] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
