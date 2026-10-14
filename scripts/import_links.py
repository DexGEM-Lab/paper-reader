#!/usr/bin/env python3
"""Import Feishu-history JSONL links through the application's ingestion pipeline.

The script runs against the configured local library/store so provenance, dedup,
web snapshots, prompt jobs, and collaboration records follow the same code path
as `/import-link`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_reader.app import create_app  # noqa: E402
from src.paper_reader.config import load_config  # noqa: E402
from src.paper_reader.llm import get_llm_client  # noqa: E402
from src.paper_reader.screening import candidate_packet, screen_packets  # noqa: E402
from src.paper_reader.web_resources import classify_import_target  # noqa: E402

TERMINAL_STATUSES = {"completed", "failed", "stopped"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import research links from JSONL with Feishu provenance.")
    parser.add_argument("jsonl", type=Path, help="JSONL file containing url/chat_name/msg_time_iso/context_text/sender_open_id")
    parser.add_argument("--dry-run", action="store_true", help="Classify and validate records without creating files or opening the DB")
    parser.add_argument("--no-resume", action="store_true", help="Refresh URLs that are already registered instead of skipping them")
    parser.add_argument("--library-root", type=Path, help="Override PAPER_READER_LIBRARY_DIR")
    parser.add_argument("--as-user", help="Existing local username credited as importer and used for automatic prompt jobs")
    parser.add_argument("--timeout", type=float, default=300.0, help="Maximum seconds to wait for each queued import")
    parser.add_argument("--screen", action="store_true", help="Fail closed unless the semantic LLM screen returns KEEP for each URL")
    parser.add_argument("--screen-model", default="deepseek-flash", help="LiteLLM model for --screen (default: deepseek-flash)")
    parser.add_argument("--screen-qps", type=float, help="Override LLM QPS for --screen")
    return parser.parse_args(argv)


def load_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        url = str(payload.get("url") or "").strip()
        chat_name = str(payload.get("chat_name") or "").strip()
        if not url or not chat_name:
            raise ValueError(f"{path}:{line_number}: url and chat_name are required")
        records.append(
            {
                "url": url,
                "chat_name": chat_name,
                "msg_time_iso": str(payload.get("msg_time_iso") or "").strip(),
                "context_text": str(payload.get("context_text") or "").strip(),
                "sender_open_id": str(payload.get("sender_open_id") or "").strip(),
            }
        )
    return records


def group_records(records: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for record in records:
        grouped.setdefault(record["url"], []).append(record)
    return list(grouped.items())


def wait_for_action(action_queue, action_id: str, timeout: float):  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        action = action_queue.get(action_id)
        if action is None:
            raise RuntimeError(f"Import action disappeared: {action_id}")
        if action.status in TERMINAL_STATUSES:
            return action
        time.sleep(0.1)
    raise TimeoutError(f"Import action timed out after {timeout:g}s: {action_id}")


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_records(args.jsonl)
    grouped = group_records(records)

    classified: list[tuple[str, list[dict[str, str]], Any]] = []
    failures = 0
    for url, provenance in grouped:
        try:
            target = classify_import_target(url)
        except Exception as exc:
            failures += 1
            emit({"url": url, "status": "invalid", "error": str(exc)})
            continue
        classified.append((url, provenance, target))
        if args.dry_run:
            emit(
                {
                    "url": url,
                    "status": "would-import",
                    "route": target.route,
                    "resource_kind": target.resource_kind,
                    "canonical_url": target.canonical_url,
                    "provenance_count": len(provenance),
                }
            )

    screening_decisions: dict[str, Any] = {}
    if args.screen:
        if args.screen_qps is not None and args.screen_qps <= 0:
            raise ValueError("--screen-qps must be positive")
        screening_config = load_config()
        if args.screen_qps is not None:
            screening_config = replace(screening_config, llm_qps=args.screen_qps)
        packets = [
            candidate_packet(
                url=url,
                title=target.canonical_url,
                kind=target.resource_kind,
                context_text="\n".join(record["context_text"] for record in provenance if record["context_text"]),
            )
            for url, provenance, target in classified
        ]
        for decision in screen_packets(packets, get_llm_client(screening_config), model=args.screen_model):
            screening_decisions[decision.url] = decision
        screened: list[tuple[str, list[dict[str, str]], Any]] = []
        for url, provenance, target in classified:
            decision = screening_decisions[url]
            emit({"url": url, "status": f"screen-{decision.decision}", "reason": decision.reason, "error": decision.error})
            if decision.decision == "keep" and decision.error is None:
                screened.append((url, provenance, target))
            else:
                failures += 1
        classified = screened

    if args.dry_run:
        emit({"summary": {"input_records": len(records), "unique_urls": len(grouped), "valid": len(classified), "invalid_or_screened_out": failures}})
        return 1 if failures else 0

    runtime_config = load_config()
    app = create_app(args.library_root, config=runtime_config)
    imported = 0
    skipped = 0
    try:
        actor = None
        if args.as_user:
            actor = app.team_store.get_user_by_username(args.as_user)
            if actor is None:
                raise ValueError(f"Unknown --as-user account: {args.as_user}")

        for url, provenance, target in classified:
            existing = app.team_store.find_paper_by_source(target.source_type, target.source_value)
            if existing and not args.no_resume:
                skipped += 1
                emit({"url": url, "status": "skipped", "rel_path": existing, "reason": "already imported"})
                continue
            action = app.action_queue.submit(
                kind="remote-import",
                title=f"Seed import: {url}",
                payload={"raw_targets": url, "recommendation_reason": "", "provenance": provenance},
                source="seed-script",
                requested_by_user_id=(actor.id if actor else None),
                requested_by_display_name=(actor.display_name if actor else None),
            )
            completed = wait_for_action(app.action_queue, action.id, args.timeout)
            items = completed.result.get("items", []) if isinstance(completed.result, dict) else []
            item = items[0] if items else {}
            if completed.status != "completed" or str(item.get("status") or "failed") == "failed":
                failures += 1
                emit(
                    {
                        "url": url,
                        "status": "failed",
                        "rel_path": item.get("rel_path"),
                        "error": completed.error or item.get("message") or completed.message,
                    }
                )
                continue
            imported += 1
            emit({"url": url, "status": item.get("status"), "rel_path": item.get("rel_path")})
    finally:
        app.action_queue.stop()
        app.job_queue.stop_all()

    emit(
        {
            "summary": {
                "input_records": len(records),
                "unique_urls": len(grouped),
                "imported_or_refreshed": imported,
                "skipped": skipped,
                "failed": failures,
            }
        }
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
