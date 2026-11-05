import base64
import hashlib
import json
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.paper_reader.app import create_app
from src.paper_reader.config import load_config
from src.paper_reader.feishu_events import extract_urls


@pytest.fixture
def app(tmp_path):
    config = load_config(project_dir=tmp_path, cwd=tmp_path, environ={
        "PAPER_READER_SECRET_KEY": "test", "PAPER_READER_FEISHU_EVENT_TOKEN": "secret",
        "PAPER_READER_FEISHU_MONITOR_CHAT_IDS": "oc_allowed, oc_other",
        "PAPER_READER_DISCOVERY_ENABLED": "false",
    })
    app = create_app(tmp_path / "library", tmp_path / "sources", config=config)
    app.testing = True
    app.action_queue.stop()
    app.action_queue.submit = Mock()
    yield app
    app.job_queue.stop_all()


def event(content=None):
    return {"schema": "2.0", "header": {"event_id": "event-1", "event_type": "im.message.receive_v1", "token": "secret"},
            "event": {"sender": {"sender_type": "user", "sender_id": {"open_id": "ou_sender"}},
                      "message": {"message_id": "om_1", "chat_id": "oc_allowed", "chat_type": "group",
                                  "message_type": "text", "create_time": "1700000000000",
                                  "content": json.dumps(content or {"text": "Read https://example.org/a and https://example.org/a"})}}}


def test_challenge_and_auth(app):
    c = app.test_client()
    assert c.post('/feishu/events', json={'type': 'url_verification', 'challenge': 'abc', 'token': 'secret'}).json == {'challenge': 'abc'}
    assert c.post('/feishu/events', json={'type': 'url_verification', 'challenge': 'abc', 'token': 'bad'}).status_code == 403
    app.config['FEISHU_EVENT_TOKEN'] = ''
    assert c.post('/feishu/events', json={}).status_code == 503


def test_allowlist_dedup_and_provenance(app):
    c = app.test_client()
    p = event()
    assert c.post('/feishu/events', json=p).status_code == 200
    assert c.post('/feishu/events', json=p).status_code == 200
    assert app.action_queue.submit.call_count == 1
    call = app.action_queue.submit.call_args.kwargs
    assert call['kind'] == 'remote-import'
    assert call['payload']['raw_targets'] == 'https://example.org/a'
    assert call['payload']['provenance'][0] == {'source_type': 'feishu', 'url': 'https://example.org/a', 'chat_id': 'oc_allowed', 'chat_name': 'oc_allowed', 'message_id': 'om_1', 'sender_open_id': 'ou_sender', 'context_text': 'Read https://example.org/a and https://example.org/a', 'msg_time_iso': '2023-11-14T22:13:20+00:00'}
    app.action_queue.submit.reset_mock()
    p['header']['event_id'] = 'event-2'
    p['event']['message']['chat_id'] = 'oc_unlisted'
    assert c.post('/feishu/events', json=p).status_code == 200
    app.config['FEISHU_MONITOR_CHAT_IDS'] = frozenset()
    p['event']['message']['chat_id'] = 'oc_allowed'
    assert c.post('/feishu/events', json=p).status_code == 200
    app.action_queue.submit.assert_not_called()


def test_bot_and_post(app):
    c = app.test_client()
    p = event({'zh_cn': {'title': 'Research', 'content': [[{'tag': 'a', 'text': 'paper', 'href': 'https://arxiv.org/abs/1'}, {'tag': 'text', 'text': 'https://arxiv.org/abs/1， https://x.org/t.'}]]}})
    p['event']['message']['message_type'] = 'post'
    p['event']['sender']['sender_type'] = 'app'
    assert c.post('/feishu/events', json=p).status_code == 200
    app.action_queue.submit.assert_not_called()
    p['event']['sender']['sender_type'] = 'user'
    assert c.post('/feishu/events', json=p).status_code == 200
    assert [call.kwargs['payload']['raw_targets'] for call in app.action_queue.submit.call_args_list] == ['https://arxiv.org/abs/1', 'https://x.org/t']
    assert extract_urls({'text': 'https://a.org, https://a.org'}) == ['https://a.org']


def test_encrypted_and_signature(app):
    app.config['FEISHU_ENCRYPT_KEY'] = 'encrypt-secret'
    data = json.dumps({'type': 'url_verification', 'challenge': 'abc', 'token': 'secret'}).encode()
    padder = padding.PKCS7(128).padder(); padded = padder.update(data) + padder.finalize()
    iv = bytes(range(16)); enc = Cipher(algorithms.AES(hashlib.sha256(b'encrypt-secret').digest()), modes.CBC(iv)).encryptor()
    body = json.dumps({'encrypt': base64.b64encode(iv + enc.update(padded) + enc.finalize()).decode()}).encode()
    signature = hashlib.sha256(b'123nonceencrypt-secret' + body).hexdigest()
    headers = {'X-Lark-Request-Timestamp': '123', 'X-Lark-Request-Nonce': 'nonce', 'X-Lark-Signature': signature, 'Content-Type': 'application/json'}
    c = app.test_client()
    assert c.post('/feishu/events', data=body, headers=headers).json == {'challenge': 'abc'}
    headers['X-Lark-Signature'] = 'bad'
    assert c.post('/feishu/events', data=body, headers=headers).status_code == 400
    app.config['FEISHU_ENCRYPT_KEY'] = ''
    assert c.post('/feishu/events', data=body, headers=headers).status_code == 400


def test_queue_failure_retries(app):
    c = app.test_client()
    app.action_queue.submit.side_effect = RuntimeError('queue unavailable')
    with pytest.raises(RuntimeError):
        c.post('/feishu/events', json=event())
    app.action_queue.submit.side_effect = None
    assert c.post('/feishu/events', json=event()).status_code == 200
    assert app.action_queue.submit.call_count == 2


def test_dedup_persists_and_source_message_id(app):
    from src.paper_reader.feishu_events import process_event
    from src.paper_reader.team_store import TeamStore
    p = event()
    db = app.config['LIBRARY_ROOT'] / '.paper-reader-feishu-events.sqlite3'
    process_event(p, db, app.action_queue, frozenset({'oc_allowed'}))
    queue_after_restart = Mock()
    process_event(p, db, queue_after_restart, frozenset({'oc_allowed'}))
    queue_after_restart.submit.assert_not_called()
    p['event']['message']['content'] = json.dumps({'text': 'https://new.example'})
    process_event(p, db, queue_after_restart, frozenset({'oc_allowed'}))
    queue_after_restart.submit.assert_not_called()
    with app.team_store._connect() as conn:
        conn.execute("INSERT INTO papers(rel_path, file_name, folder, extension, title, display_title, modified_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ("item.md", "item.md", "", ".md", "Item", "Item", "2026", "2026", "2026"))
    app.team_store.add_source('item.md', source_type='feishu', source_value='oc_allowed', source_url='https://example.org/a', source_context='context', sender_open_id='ou_sender', message_id='om_1')
    reloaded = TeamStore(app.config['LIBRARY_ROOT'])
    assert reloaded.sources_for_rel_path('item.md')[0]['message_id'] == 'om_1'


def test_partial_enqueue_retry_only_remaining(app):
    c = app.test_client()
    p = event({'text': 'https://a.org https://b.org'})
    app.action_queue.submit.side_effect = [None, RuntimeError('failure')]
    with pytest.raises(RuntimeError):
        c.post('/feishu/events', json=p)
    app.action_queue.submit.reset_mock()
    app.action_queue.submit.side_effect = None
    assert c.post('/feishu/events', json=p).status_code == 200
    assert app.action_queue.submit.call_args.kwargs['payload']['raw_targets'] == 'https://b.org'
    assert app.action_queue.submit.call_count == 1
