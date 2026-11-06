"""Durable asynchronous feedback triage and issue ledger."""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import get_llm_client

LOG = logging.getLogger(__name__)
PRODUCT = ("DexGEM Paper Reader 是团队论文和网页阅读器：支持链接导入书库、阅读论文及网页、"
           "中文解读、基于内容的我爱学/一起学 agent 对话、发现推荐、洞察。")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_triage(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I)
    value = json.loads(text)
    if not isinstance(value, dict) or value.get('category') not in {'bug', 'feature_request', 'question', 'noise'}:
        raise ValueError('invalid triage category')
    confidence = value.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('invalid triage confidence')
    summary = value.get('summary_zh')
    if not isinstance(summary, str) or (value['category'] != 'noise' and not summary.strip()):
        raise ValueError('invalid triage summary')
    value['summary_zh'] = summary.strip()[:300]
    if confidence < 0.7:
        value['category'] = 'noise'
    return value


def reply_text(entry: dict[str, Any]) -> str:
    if entry['category'] == 'bug':
        return f"已记录为 {entry['issue']}（{entry['summary']}），状态：修复中。修复上线后我会在这里回复。"
    return f"已记录为功能请求 {entry['issue']}（{entry['summary']}），已转给逸文评估，讨论后回复你。"


class FeedbackAgent:
    def __init__(self, path: Path, config: Any):
        self.path, self.config = path, config
        self.wake = threading.Event()
        self._client = None
        self._init()

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (message_id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, sender TEXT NOT NULL,
                text TEXT NOT NULL, parent_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS issues (issue TEXT PRIMARY KEY, message_id TEXT UNIQUE NOT NULL, chat_id TEXT NOT NULL,
                sender TEXT NOT NULL, text TEXT NOT NULL, category TEXT NOT NULL, summary TEXT NOT NULL,
                severity TEXT NOT NULL, confidence REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, resolution_note TEXT NOT NULL DEFAULT '', commit_ref TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS changes (seq INTEGER PRIMARY KEY AUTOINCREMENT, issue TEXT NOT NULL, snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS bot_replies (message_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, message_id TEXT NOT NULL, text TEXT NOT NULL,
                sent_id TEXT NOT NULL DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '');
            ''')

    def enqueue(self, message_id: str, chat_id: str, sender: str, text: str, parent_id: str):
        if not message_id:
            raise ValueError('missing feedback message id')
        with self.connect() as db:
            if db.execute('SELECT 1 FROM bot_replies WHERE message_id=?', (message_id,)).fetchone():
                return
            db.execute('INSERT OR IGNORE INTO jobs(message_id,chat_id,sender,text,parent_id) VALUES(?,?,?,?,?)',
                       (message_id, chat_id, sender, text, parent_id))
        self.wake.set()

    def start(self):
        threading.Thread(target=self.run, name='feedback-agent', daemon=True).start()

    def client(self):
        if self._client is None:
            from .feishu_events import FeishuClient
            self._client = FeishuClient(self.config.feishu_api_base_url, self.config.feishu_app_id, self.config.feishu_app_secret)
        return self._client

    def run(self):
        while True:
            try:
                self.process_one()
            except Exception:
                LOG.exception('feedback worker error')
            self.wake.wait(3)
            self.wake.clear()

    def process_one(self):
        with self.connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE state='pending' AND next_at<=? ORDER BY rowid LIMIT 1", (time.time(),)).fetchone()
        if job:
            try:
                self.triage(dict(job))
            except Exception as exc:
                LOG.exception('feedback triage failed for %s', job['message_id'])
                with self.connect() as db:
                    db.execute('UPDATE jobs SET attempts=attempts+1,next_at=?,error=? WHERE message_id=?',
                               (time.time() + min(3600, 30 * 2 ** min(job['attempts'], 6)), str(exc)[:500], job['message_id']))
        with self.connect() as db:
            outgoing = db.execute("SELECT * FROM outbox WHERE sent_id='' AND next_at<=? ORDER BY rowid LIMIT 1", (time.time(),)).fetchone()
        if outgoing:
            try:
                sent_id = self.client().reply(outgoing['message_id'], outgoing['text'], outgoing['key'])
                with self.connect() as db:
                    db.execute('INSERT OR IGNORE INTO bot_replies VALUES (?)', (sent_id,))
                    db.execute('UPDATE outbox SET sent_id=?,error=\'\' WHERE key=?', (sent_id, outgoing['key']))
            except Exception as exc:
                LOG.exception('feedback reply failed for %s', outgoing['message_id'])
                with self.connect() as db:
                    db.execute('UPDATE outbox SET attempts=attempts+1,next_at=?,error=? WHERE key=?',
                               (time.time() + min(3600, 30 * 2 ** min(outgoing['attempts'], 6)), str(exc)[:500], outgoing['key']))

    def triage(self, job: dict[str, Any]):
        sender = job['sender']
        try:
            sender = self.client().sender_name(sender) or sender
        except Exception:
            LOG.warning('Could not resolve feedback sender %s', sender, exc_info=True)
        context = ''
        if job['parent_id']:
            with self.connect() as db:
                parent = db.execute('SELECT text FROM jobs WHERE message_id=?', (job['parent_id'],)).fetchone()
            if parent:
                context = parent['text']
            else:
                try:
                    context = self.client().message_text(job['parent_id'])
                except Exception:
                    LOG.warning('Could not fetch feedback thread context', exc_info=True)
        prompt = (f"{PRODUCT}\n将用户反馈分类为 bug、feature_request、question、noise。只输出 JSON，字段 category, summary_zh, severity, confidence (0到1)。"
                  "对于 question，若仅凭上述产品信息可以回答，额外返回 answer_zh，否则 answer_zh 为空。纯闲聊、非反馈、对已有 ISSUE/FR 的登记确认、修复或评估进展通告均为 noise；仅在消息提出新的问题或请求时创建新条目。"
                  f"\n发送者：{sender}\n回复上下文：{context[:2000]}\n消息：{job['text'][:4000]}")
        result = parse_triage(get_llm_client(self.config).complete([{'role': 'user', 'content': prompt}], model=self.config.llm_cheap_model))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT state FROM jobs WHERE message_id=?', (job['message_id'],)).fetchone()['state'] != 'pending':
                return
            category = result['category']
            if category in {'bug', 'feature_request'}:
                prefix = 'ISSUE' if category == 'bug' else 'FR'
                # Allocate from the max numeric suffix, not row count: deleted
                # records (misclassified/merged) must never make ids go backwards
                # or collide with previously announced numbers.
                cut = len(prefix) + 2
                count = db.execute(
                    "SELECT COALESCE(MAX(CAST(SUBSTR(issue, ?) AS INTEGER)), 0) + 1 FROM issues WHERE issue GLOB ?",
                    (cut, prefix + '-[0-9]*'),
                ).fetchone()[0]
                issue, stamp = f'{prefix}-{count:03d}', now()
                status = 'fixing' if category == 'bug' else 'evaluating'
                db.execute('INSERT INTO issues VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           (issue, job['message_id'], job['chat_id'], sender, job['text'], category, result['summary_zh'],
                            str(result.get('severity') or ''), result['confidence'], status, stamp, stamp, '', ''))
                entry = dict(db.execute('SELECT * FROM issues WHERE issue=?', (issue,)).fetchone())
                db.execute('INSERT INTO changes(issue,snapshot) VALUES (?,?)', (issue, json.dumps(entry, ensure_ascii=False)))
                db.execute('INSERT INTO outbox(key,message_id,text) VALUES(?,?,?)', ('initial:' + job['message_id'], job['message_id'], reply_text(entry)))
            elif category == 'question':
                answer = result.get('answer_zh')
                answer = answer.strip()[:2000] if isinstance(answer, str) else ''
                db.execute('INSERT INTO outbox(key,message_id,text) VALUES(?,?,?)', ('initial:' + job['message_id'], job['message_id'], answer or '已记录，稍后回复。'))
            db.execute("UPDATE jobs SET state='done',sender=?,error='' WHERE message_id=?", (sender, job['message_id']))

    def pending(self, cursor: int):
        with self.connect() as db:
            rows = db.execute('SELECT seq,snapshot FROM changes WHERE seq>? ORDER BY seq LIMIT 100', (cursor,)).fetchall()
        return {'entries': [dict(json.loads(row['snapshot']), cursor=row['seq']) for row in rows],
                'cursor': rows[-1]['seq'] if rows else cursor}

    def resolve(self, issue: str, status: str, note: str, commit: str):
        if status not in {'open', 'fixing', 'fixed', 'wontfix', 'evaluating'} or not issue or not note:
            raise ValueError('issue, valid status and resolution_note required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM issues WHERE issue=?', (issue,)).fetchone()
            if not row:
                raise LookupError('issue not found')
            if row['status'] != status or row['resolution_note'] != note or row['commit_ref'] != commit:
                db.execute('UPDATE issues SET status=?,resolution_note=?,commit_ref=?,updated_at=? WHERE issue=?', (status,note,commit,now(),issue))
                entry = dict(db.execute('SELECT * FROM issues WHERE issue=?', (issue,)).fetchone())
                seq = db.execute('INSERT INTO changes(issue,snapshot) VALUES (?,?)', (issue,json.dumps(entry,ensure_ascii=False))).lastrowid
                if issue.startswith('FR-'):
                    text = f'{issue} 评估结果：{note}'
                elif status == 'fixed':
                    text = f'{issue} 已修复并上线：{note}' + (f' (commit {commit})' if commit else '')
                else:
                    text = f'{issue} 状态更新：{note}'
                db.execute('INSERT INTO outbox(key,message_id,text) VALUES(?,?,?)', (f'resolve:{seq}',row['message_id'],text))
            else:
                entry = dict(row)
        self.wake.set()
        return entry
