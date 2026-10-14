#!/usr/bin/env python3
"""Screen the existing library with the shared semantic screening policy.

Run once to create an audit, review its drop records, then run again with
``--apply-drops --confirm-reviewed-drops``.  The latter consumes the reviewed
JSONL rather than asking the model again, making the operation resumable and
keeping destructive work separate from classification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_reader.app import create_app  # noqa: E402
from src.paper_reader.config import load_config  # noqa: E402
from src.paper_reader.llm import get_llm_client  # noqa: E402
from src.paper_reader.screening import (  # noqa: E402
    ScreeningDecision,
    ScreeningPacket,
    delete_library_item,
    library_packet,
    screen_packets,
    update_corrected_kind,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-screen deployed Paper Reader entries before cleanup.")
    parser.add_argument("--library-root", type=Path, help="Override PAPER_READER_LIBRARY_DIR")
    parser.add_argument("--audit", type=Path, help="JSONL audit path (default: <data-dir>/library-screening.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions only; do not write audit, metadata, DB, or files.")
    parser.add_argument("--apply-drops", action="store_true", help="Delete only previously audited DROP decisions.")
    parser.add_argument("--confirm-reviewed-drops", action="store_true", help="Required acknowledgement before --apply-drops.")
    parser.add_argument("--rescreen", action="store_true", help="Ignore matching prior audit decisions and query the LLM again.")
    parser.add_argument("--batch-size", type=int, default=8, help="Items per LLM call (default: 8).")
    parser.add_argument("--model", default="deepseek-flash", help="Allowed LiteLLM model (default: deepseek-flash).")
    parser.add_argument("--qps", type=float, help="Override LLM request QPS; production cleanup should use 0.2 or less.")
    return parser.parse_args(argv)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def audit_event(*, packet: ScreeningPacket, decision: ScreeningDecision, event: str, **extra: Any) -> dict[str, Any]:
    return {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packet": packet.summary(),
        "packet_fingerprint": packet.fingerprint,
        "decision": decision.as_dict(),
        **extra,
    }


def append_audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_decisions(path: Path) -> dict[str, tuple[str, ScreeningDecision]]:
    """Return the latest screen result per relative path, ignoring malformed lines."""
    latest: dict[str, tuple[str, ScreeningDecision]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
            packet = entry["packet"]
            raw_decision = entry["decision"]
            rel_path = str(packet.get("rel_path") or "")
            fingerprint = str(entry["packet_fingerprint"])
            if entry.get("event") != "decision" or not rel_path:
                continue
            latest[rel_path] = (
                fingerprint,
                ScreeningDecision(
                    item_id=str(raw_decision["item_id"]),
                    url=str(raw_decision.get("url") or ""),
                    decision=str(raw_decision["decision"]),
                    reason=str(raw_decision["reason"]),
                    corrected_kind=(str(raw_decision["corrected_kind"]) if raw_decision.get("corrected_kind") else None),
                    error=(str(raw_decision["error"]) if raw_decision.get("error") else None),
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return latest


def apply_audited_drops(app: Any, packets_by_path: dict[str, tuple[Any, ScreeningPacket]], audit: Path) -> dict[str, int]:
    latest = latest_decisions(audit)
    deleted = missing = failures = retagged = 0
    for rel_path, (record, packet) in packets_by_path.items():
        prior = latest.get(rel_path)
        if prior is None or prior[0] != packet.fingerprint or prior[1].decision != "drop" or prior[1].error:
            continue
        decision = prior[1]
        try:
            delete_library_item(app, rel_path)
            deleted += 1
            append_audit(audit, audit_event(packet=packet, decision=decision, event="deleted"))
            emit({"rel_path": rel_path, "url": decision.url, "status": "deleted", "reason": decision.reason})
        except FileNotFoundError:
            missing += 1
            append_audit(audit, audit_event(packet=packet, decision=decision, event="already-absent"))
        except Exception as exc:
            failures += 1
            append_audit(audit, audit_event(packet=packet, decision=decision, event="delete-failed", error=str(exc)))
            emit({"rel_path": rel_path, "status": "delete-failed", "error": str(exc)})
    return {"deleted": deleted, "already_absent": missing, "delete_failed": failures, "retagged": retagged}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.qps is not None and args.qps <= 0:
        raise ValueError("--qps must be positive")
    if args.apply_drops and not args.confirm_reviewed_drops:
        raise ValueError("--apply-drops requires --confirm-reviewed-drops after human review of the audit.")

    runtime_config = load_config()
    if args.qps is not None:
        runtime_config = replace(runtime_config, llm_qps=args.qps)
    audit = args.audit or (runtime_config.data_dir / "library-screening.jsonl")
    app = create_app(args.library_root, config=runtime_config)
    try:
        scan = app.library.scan(force=True, include_done=True)
        packets_by_path = {record.rel_path: (record, library_packet(app, record)) for record in scan.papers}
        if args.apply_drops:
            result = apply_audited_drops(app, packets_by_path, audit)
            emit({"summary": {"library_items": len(scan.papers), **result, "audit": str(audit)}})
            return 1 if result["delete_failed"] else 0

        old = latest_decisions(audit)
        to_screen: list[ScreeningPacket] = []
        existing: dict[str, ScreeningDecision] = {}
        for rel_path, (_, packet) in packets_by_path.items():
            prior = old.get(rel_path)
            if not args.rescreen and prior is not None and prior[0] == packet.fingerprint and prior[1].actionable:
                existing[rel_path] = prior[1]
            else:
                to_screen.append(packet)

        decisions = screen_packets(to_screen, get_llm_client(runtime_config), model=args.model, batch_size=args.batch_size)
        by_path = {decision.item_id: decision for decision in decisions}
        summary = {"keep": 0, "drop": 0, "error": 0, "resumed": len(existing), "retagged": 0}
        for rel_path, (_, packet) in packets_by_path.items():
            decision = existing.get(rel_path) or by_path[rel_path]
            summary[decision.decision] = summary.get(decision.decision, 0) + 1
            payload = {"rel_path": rel_path, "url": decision.url, "decision": decision.decision, "reason": decision.reason}
            if decision.error:
                payload["error"] = decision.error
            emit(payload)
            if rel_path in existing:
                continue
            if not args.dry_run:
                append_audit(audit, audit_event(packet=packet, decision=decision, event="decision"))
                record = packets_by_path[rel_path][0]
                if decision.decision == "keep" and update_corrected_kind(app, record, decision.corrected_kind):
                    summary["retagged"] += 1
                    append_audit(audit, audit_event(packet=packet, decision=decision, event="retagged"))
        emit({"summary": {"library_items": len(scan.papers), **summary, "audit": str(audit), "dry_run": args.dry_run}})
        return 0
    finally:
        app.action_queue.stop()
        app.job_queue.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
