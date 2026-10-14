#!/usr/bin/env python3
"""Run one paper discovery pass for cron or a manual operations check."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_reader.app import create_app  # noqa: E402
from src.paper_reader.config import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the Paper Reader discovery inbox once.")
    parser.add_argument("--library-root", type=Path, help="Override PAPER_READER_LIBRARY_DIR")
    parser.add_argument("--force-profile", action="store_true", help="Regenerate the 24-hour interest profile cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = replace(load_config(), discovery_enabled=False)
    app = create_app(args.library_root, config=config)
    try:
        if args.force_profile:
            app.discovery_service.interest_profile(force=True)
        result = app.discovery_service.run()
        print(json.dumps({"fetched": result.fetched, "eligible": result.eligible, "surfaced": result.surfaced}, ensure_ascii=False))
        return 0
    finally:
        app.action_queue.stop()
        app.job_queue.stop_all()
        app.discovery_scheduler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
