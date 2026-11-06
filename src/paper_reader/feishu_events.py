"""Verified, allowlisted Feishu message ingestion."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from urllib.request import Request, urlopen
from urllib.parse import quote
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
    # Persist each accepted URL separately, so retries resume after a partial enqueue.
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
        db.execute("CREATE TABLE IF NOT EXISTS feishu_event_urls (event_id TEXT, url TEXT, PRIMARY KEY (event_id, url))")
        db.execute("CREATE TABLE IF NOT EXISTS feishu_processed_events (event_id TEXT PRIMARY KEY)")
        for url, entry in zip(urls, provenance):
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM feishu_processed_events WHERE event_id = ?", (event_id,)).fetchone():
                db.commit()
                return
            if db.execute("SELECT 1 FROM feishu_event_urls WHERE event_id = ? AND url = ?", (event_id, url)).fetchone():
                db.commit()
                continue
            queue.submit(kind="remote-import", title=f"飞书链接导入：{url}",
                         payload={"raw_targets": url, "recommendation_reason": "", "provenance": [entry]},
                         source="feishu")
            db.execute("INSERT INTO feishu_event_urls VALUES (?, ?)", (event_id, url))
            db.commit()
        db.execute("INSERT OR IGNORE INTO feishu_processed_events VALUES (?)", (event_id,))
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
        allowed = config["FEISHU_MONITOR_CHAT_IDS"] | config["FEISHU_FEEDBACK_CHAT_IDS"]
        process_event(payload, Path(config["LIBRARY_ROOT"]) / ".paper-reader-feishu-events.sqlite3", current_app.action_queue, allowed)
        enqueue_feedback(payload, current_app.feedback_agent, config["FEISHU_FEEDBACK_CHAT_IDS"])
    except (ValueError, TypeError, KeyError) as exc:
        LOGGER.warning("Rejected Feishu event: %s", exc)
        return {"error": "invalid event"}, 400
    return {"ok": True}


class FeishuClient:
    """Minimal tenant-token client; errors are surfaced for durable retry."""
    def __init__(self, base: str, app_id: str, app_secret: str):
        self.base, self.app_id, self.app_secret = base.rstrip('/'), app_id, app_secret
        self._token, self._expires = '', 0.0
        self._lock = threading.Lock()

    def _request(self, path: str, body: dict | None = None, token: str = '') -> dict:
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        with urlopen(Request(self.base + path, data=data, headers=headers), timeout=15) as response:
            payload = json.load(response)
        if payload.get('code') != 0:
            raise RuntimeError(f"Feishu API error {payload.get('code')}: {payload.get('msg')}")
        return payload

    def token(self) -> str:
        with self._lock:
            if time.monotonic() >= self._expires:
                result = self._request('/open-apis/auth/v3/tenant_access_token/internal',
                                       {'app_id': self.app_id, 'app_secret': self.app_secret})
                self._token = result['tenant_access_token']
                self._expires = time.monotonic() + max(1, int(result['expire']) - 120)
            return self._token

    def reply(self, message_id: str, text: str, key: str) -> str:
        result = self._request('/open-apis/im/v1/messages/' + quote(message_id, safe='') + '/reply',
                               {'msg_type': 'text', 'content': json.dumps({'text': text}, ensure_ascii=False),
                                'reply_in_thread': True, 'uuid': str(uuid.uuid5(uuid.NAMESPACE_URL, key))}, self.token())
        sent = result.get('data', {}).get('message_id')
        if not sent:
            raise RuntimeError('Feishu reply returned no message_id')
        return sent

    def sender_name(self, open_id: str) -> str:
        result = self._request('/open-apis/contact/v3/users/' + quote(open_id, safe='') + '?user_id_type=open_id', token=self.token())
        return result.get('data', {}).get('user', {}).get('name', '')

    def message_text(self, message_id: str) -> str:
        result = self._request('/open-apis/im/v1/messages/' + quote(message_id, safe=''), token=self.token())
        items = result.get('data', {}).get('items', [])
        if not items:
            return ''
        content = items[0].get('body', {}).get('content', '{}')
        return _text_fragments(json.loads(content))[:2000]


def enqueue_feedback(payload: dict[str, Any], agent: Any, allowed: frozenset[str]) -> None:
    header, event = payload.get('header') or {}, payload.get('event') or {}
    if not isinstance(header, dict) or header.get('event_type') != 'im.message.receive_v1' or not isinstance(event, dict):
        return
    message, sender = event.get('message') or {}, event.get('sender') or {}
    if not isinstance(message, dict) or not isinstance(sender, dict):
        return
    if message.get('chat_id') not in allowed or message.get('chat_type') != 'group' or sender.get('sender_type') == 'app':
        return
    if message.get('message_type') not in {'text', 'post', 'rich_text'}:
        return
    content = json.loads(message.get('content') or '{}')
    text = ' '.join(_text_fragments(content).split())
    if not text or not URL.sub('', text).strip(' \t\r\n.,!?;:()[]{}、，。！？；：（）【】《》'):
        return
    sender_id = sender.get('sender_id') or {}
    agent.enqueue(str(message.get('message_id') or ''), message['chat_id'],
                  str(sender_id.get('open_id') or '') if isinstance(sender_id, dict) else '', text,
                  str(message.get('parent_id') or message.get('root_id') or ''))


def authorized() -> bool:
    token = current_app.config['INTERNAL_TOKEN']
    supplied = request.args.get('token', '') or request.headers.get('Authorization', '').removeprefix('Bearer ')
    return bool(token and supplied and hmac.compare_digest(token, supplied))


@blueprint.get('/internal/feedback/pending')
def feedback_pending() -> Any:
    if not authorized():
        return {'error': 'unauthorized'}, 403
    try:
        cursor = int(request.args.get('cursor', '0'))
        if cursor < 0:
            raise ValueError('negative cursor')
    except ValueError:
        return {'error': 'invalid cursor'}, 400
    return current_app.feedback_agent.pending(cursor)


@blueprint.post('/internal/feedback/resolve')
def feedback_resolve() -> Any:
    if not authorized():
        return {'error': 'unauthorized'}, 403
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or any(not isinstance(data.get(k, ''), str) for k in ('issue','status','resolution_note','commit_ref')):
        return {'error': 'invalid request'}, 400
    try:
        result = current_app.feedback_agent.resolve(*(data.get(k, '').strip() for k in ('issue','status','resolution_note','commit_ref')))
    except ValueError as exc:
        return {'error': str(exc)}, 400
    except LookupError as exc:
        return {'error': str(exc)}, 404
    return {'entry': result, 'reply': 'queued'}
