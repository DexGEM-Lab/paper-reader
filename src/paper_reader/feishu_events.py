"""Verified, allowlisted Feishu message ingestion."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, request

LOGGER = logging.getLogger(__name__)
blueprint = Blueprint("feishu_events", __name__)
URL = re.compile(r'https?://[^\s<>"\'\u3000-\u303f\uff00-\uffef]+', re.IGNORECASE)


def extract_urls(content: Any) -> list[str]:
    """Extract URLs from textual and rich-text fields, preserving first occurrence."""
    strings: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
    visit(content)
    found: dict[str, None] = {}
    for text in strings:
        for match in URL.findall(text):
            url = match.rstrip('.,!?;:)]}、，。！？；：）】》')
            if url:
                found[url] = None
    return list(found)


def decode_payload(raw: bytes, encrypt_key: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("invalid payload")
    if "encrypt" in payload:
        if not encrypt_key:
            raise ValueError("encrypted events are not configured")
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        ts = request.headers.get("X-Lark-Request-Timestamp", "")
        nonce = request.headers.get("X-Lark-Request-Nonce", "")
        signature = request.headers.get("X-Lark-Signature", "")
        expected = hashlib.sha256((ts + nonce + encrypt_key).encode() + raw).hexdigest()
        if not ts or not nonce or not hmac.compare_digest(expected, signature):
            raise ValueError("invalid signature")
        data = base64.b64decode(payload["encrypt"], validate=True)
        decryptor = Cipher(algorithms.AES(hashlib.sha256(encrypt_key.encode()).digest()), modes.CBC(data[:16])).decryptor()
        padded = decryptor.update(data[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        payload = json.loads(unpadder.update(padded) + unpadder.finalize())
        if not isinstance(payload, dict):
            raise ValueError("invalid decrypted payload")
    elif encrypt_key:
        raise ValueError("unencrypted event received while encryption is configured")
    return payload


def process_event(payload: dict[str, Any], db_path: Path, queue: Any, allowed: frozenset[str]) -> None:
    header = payload.get("header") or {}
    if not isinstance(header, dict) or header.get("event_type") != "im.message.receive_v1":
        return
    event_id = header.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("missing event id")
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        raise ValueError("invalid event")
    message, sender = event.get("message") or {}, event.get("sender") or {}
    if not isinstance(message, dict) or not isinstance(sender, dict):
        raise ValueError("invalid message")
    chat_id = message.get("chat_id")
    if chat_id not in allowed or message.get("chat_type") != "group" or sender.get("sender_type") == "app":
        return
    if message.get("message_type") not in {"text", "post", "rich_text"}:
        return
    content = json.loads(message.get("content") or "{}")
    urls = extract_urls(content)
    if not urls:
        return
    # A single queued action avoids partial enqueue across URLs. The SQLite write lock
    # serializes duplicate callbacks, even across threads/processes.
    context = " ".join(_text_fragments(content).split())[:200]
    sender_id = sender.get("sender_id") or {}
    sender_open_id = sender_id.get("open_id", "") if isinstance(sender_id, dict) else ""
    time_iso = ""
    try:
        time_iso = datetime.fromtimestamp(int(message.get("create_time")) / 1000, timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    provenance = [{"source_type": "feishu", "chat_name": chat_id, "chat_id": chat_id,
                   "message_id": str(message.get("message_id") or ""), "sender_open_id": sender_open_id,
                   "context_text": context, "msg_time_iso": time_iso, "url": url} for url in urls]
    with sqlite3.connect(db_path, timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS feishu_processed_events (event_id TEXT PRIMARY KEY)")
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM feishu_processed_events WHERE event_id = ?", (event_id,)).fetchone():
            return
        queue.submit(kind="remote-import", title=f"飞书链接导入（{len(urls)} 项）",
                     payload={"raw_targets": "\n".join(urls), "recommendation_reason": "", "provenance": provenance},
                     source="feishu")
        db.execute("INSERT INTO feishu_processed_events VALUES (?)", (event_id,))
    LOGGER.info("Feishu event queued: chat=%s message=%s links=%d", chat_id, message.get("message_id"), len(urls))


def _text_fragments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_text_fragments(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text_fragments(item) for item in value.values())
    return ""


@blueprint.post("/feishu/events")
def webhook() -> Any:
    config = current_app.config
    token = config["FEISHU_EVENT_TOKEN"]
    if not token:
        return {"error": "event verification is not configured"}, 503
    try:
        payload = decode_payload(request.get_data(), config["FEISHU_ENCRYPT_KEY"])
        header = payload.get("header") or {}
        supplied = header.get("token", "") if isinstance(header, dict) else ""
        supplied = supplied or payload.get("token", "")
        if not isinstance(supplied, str) or not hmac.compare_digest(token, supplied):
            return {"error": "invalid token"}, 403
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge")
            if not isinstance(challenge, str):
                raise ValueError("missing challenge")
            return {"challenge": challenge}
        process_event(payload, Path(config["LIBRARY_ROOT"]) / ".paper-reader-feishu-events.sqlite3", current_app.action_queue, config["FEISHU_MONITOR_CHAT_IDS"])
    except (ValueError, TypeError, KeyError) as exc:
        LOGGER.warning("Rejected Feishu event: %s", exc)
        return {"error": "invalid event"}, 400
    return {"ok": True}
