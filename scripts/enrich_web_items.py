#!/usr/bin/env python3
"""Resume-safe rich-media and attachment enrichment for existing web items."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_reader.app import create_app  # noqa: E402
from src.paper_reader.config import load_config  # noqa: E402
from src.paper_reader.llm import get_llm_client  # noqa: E402
from src.paper_reader.web_resources import ImportTarget, classify_import_target  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich existing web snapshots with local media and attachments.")
    parser.add_argument("--library-root", type=Path, help="Override PAPER_READER_LIBRARY_DIR")
    parser.add_argument("--limit", type=int, help="Maximum number of pending web items to process")
    parser.add_argument("--qps", type=float, default=0.2, help="Maximum item processing rate (default: 0.2)")
    parser.add_argument("--model", help="Attachment judge model (default: PAPER_READER_LLM_CHEAP_MODEL)")
    return parser.parse_args(argv)


def enrich_web_items(
    app: Any,
    *,
    client: Any,
    model: str,
    limit: int | None,
    qps: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if qps <= 0:
        raise ValueError("--qps must be positive")

    records = [record for record in app.library.scan(include_done=True).papers if record.resource_kind != "paper"]
    stats = {"scanned": len(records), "processed": 0, "skipped": 0, "failed": 0, "attachments": 0}
    interval = 1.0 / qps
    for record in records:
        if app.library.web_resources.is_enriched(record.rel_path):
            stats["skipped"] += 1
            continue
        if limit is not None and stats["processed"] + stats["failed"] >= limit:
            break
        metadata = app.library.web_resources.metadata_for_rel_path(record.rel_path) or {}
        canonical_url = str(metadata.get("canonical_url") or record.source_url or "").strip()
        if not canonical_url:
            stats["failed"] += 1
            print(json.dumps({"rel_path": record.rel_path, "status": "failed", "error": "missing canonical URL"}, ensure_ascii=False))
            continue
        started = time.monotonic()
        try:
            classified = classify_import_target(canonical_url)
            target = ImportTarget(
                raw_url=canonical_url,
                canonical_url=canonical_url,
                route="web",
                resource_kind=str(metadata.get("resource_kind") or record.resource_kind),
                source_type="canonical_url",
                source_value=canonical_url,
                source_url=canonical_url,
            ) if classified.route != "web" else classified
            snapshot = app.library.web_resources.fetch_and_store(target, existing_rel_path=record.rel_path)
            refreshed = app.library.build_record_for_rel_path(snapshot.rel_path, [])
            app.team_store.sync_papers([refreshed])
            attachments = app.library.web_resources.discover_attachments(
                snapshot.rel_path,
                client=client,
                model=model,
            )
            app.team_store.replace_attachments(snapshot.rel_path, attachments)
            stats["processed"] += 1
            stats["attachments"] += len(attachments)
            print(json.dumps({"rel_path": record.rel_path, "status": "enriched", "assets": len((app.library.web_resources.metadata_for_rel_path(record.rel_path) or {}).get("assets") or []), "attachments": len(attachments)}, ensure_ascii=False))
        except Exception as exc:
            stats["failed"] += 1
            print(json.dumps({"rel_path": record.rel_path, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        elapsed = time.monotonic() - started
        if elapsed < interval:
            sleep_fn(interval - elapsed)
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    if args.qps <= 0:
        raise ValueError("--qps must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    throttled_config = replace(config, llm_qps=min(config.llm_qps, args.qps))
    app = create_app(args.library_root, config=throttled_config)
    try:
        stats = enrich_web_items(
            app,
            client=get_llm_client(throttled_config),
            model=args.model or throttled_config.llm_cheap_model,
            limit=args.limit,
            qps=args.qps,
        )
        print(json.dumps({"summary": stats}, ensure_ascii=False, sort_keys=True))
        return 1 if stats["failed"] else 0
    finally:
        app.action_queue.stop()
        app.job_queue.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
