import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.paper_reader.app import create_app
from src.paper_reader.config import load_config
from src.paper_reader.feedback import FeedbackAgent, parse_triage, reply_text
from src.paper_reader.feishu_events import FeishuClient, enqueue_feedback


def config(tmp_path):
    return load_config(project_dir=tmp_path, cwd=tmp_path, environ={'PAPER_READER_SECRET_KEY':'x', 'PAPER_READER_FEISHU_EVENT_TOKEN':'ev', 'PAPER_READER_INTERNAL_TOKEN':'secret', 'PAPER_READER_DISCOVERY_ENABLED':'false'})


def item(mid='m1', text='移动端书库空白', sender='user', chat='feedback'):
    return {'header': {'event_type':'im.message.receive_v1','event_id':mid,'token':'ev'}, 'event': {'sender': {'sender_type':sender,'sender_id':{'open_id':'ou_x'}},'message':{'message_id':mid,'chat_id':chat,'chat_type':'group','message_type':'text','content':json.dumps({'text':text})}}}


def test_routing_urls_and_loop_guard(tmp_path):
    agent = Mock()
    enqueue_feedback(item(text='https://example.com/x'), agent, frozenset({'feedback'}))
    enqueue_feedback(item(sender='app'), agent, frozenset({'feedback'}))
    enqueue_feedback(item(chat='other'), agent, frozenset({'feedback'}))
    agent.enqueue.assert_not_called()
    enqueue_feedback(item(text='有问题 https://example.com/x'), agent, frozenset({'feedback'}))
    assert agent.enqueue.call_count == 1
    app = create_app(tmp_path / 'lib', tmp_path / 'src', config=config(tmp_path))
    app.testing = True
    app.action_queue.stop()
    app.action_queue.submit = Mock()
    app.config['FEISHU_FEEDBACK_CHAT_IDS'] = frozenset({'feedback'})
    client = app.test_client()
    assert client.post('/feishu/events', json=item(text='有问题 https://example.com/x')).status_code == 200
    assert app.action_queue.submit.call_count == 1
    with app.feedback_agent.connect() as db:
        assert db.execute('select count(*) from jobs').fetchone()[0] == 1
    app.job_queue.stop_all()


def test_parse_triage():
    assert parse_triage('```json\n{"category":"bug","summary_zh":"坏了","confidence":0.8}\n```')['category'] == 'bug'
    assert parse_triage('{"category":"bug","summary_zh":"坏了","confidence":0.4}')['category'] == 'noise'
    with pytest.raises(ValueError): parse_triage('{"category":"other","summary_zh":"x","confidence":1}')
    with pytest.raises(ValueError): parse_triage('{"category":"bug","summary_zh":"x","confidence":NaN}')


def mock_triage(monkeypatch, category='bug'):
    complete = Mock(return_value=json.dumps({'category':category, 'summary_zh':'测试问题', 'severity':'medium', 'confidence':0.95}))
    monkeypatch.setattr('src.paper_reader.feedback.get_llm_client', lambda config: SimpleNamespace(complete=complete))
    return complete


def test_counters_atomic_and_replies(tmp_path, monkeypatch):
    agent = FeedbackAgent(tmp_path / 'f.db', config(tmp_path))
    agent._client = Mock(sender_name=Mock(return_value='陆逸文'))
    mock_triage(monkeypatch)
    for i in range(20): agent.enqueue(str(i), 'chat', 'ou_x', '坏了', '')
    with agent.connect() as db: jobs = [dict(row) for row in db.execute('select * from jobs')]
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(agent.triage, jobs))
    data = agent.pending(0)['entries']
    assert {e['issue'] for e in data} == {f'ISSUE-{i:03}' for i in range(1, 21)}
    assert '状态：修复中' in reply_text(data[0])
    mock_triage(monkeypatch, 'feature_request')
    agent.enqueue('f', 'chat', 'ou_x', '建议收藏夹', '')
    with agent.connect() as db: job = dict(db.execute("select * from jobs where message_id='f'").fetchone())
    agent.triage(job)
    feature = agent.pending(20)['entries'][0]
    assert feature['issue'] == 'FR-001' and '逸文评估' in reply_text(feature)


def test_pending_resolution_and_thread_delivery(tmp_path, monkeypatch):
    app = create_app(tmp_path / 'lib', tmp_path / 'src', config=config(tmp_path))
    app.testing = True
    app.action_queue.stop()
    agent = app.feedback_agent
    agent._client = Mock(sender_name=Mock(return_value='陆逸文'), reply=Mock(return_value='bot-message'))
    mock_triage(monkeypatch)
    agent.enqueue('m1', 'chat', 'ou_x', '坏了', '')
    agent.process_one()
    client = app.test_client()
    assert client.get('/internal/feedback/pending').status_code == 403
    data = client.get('/internal/feedback/pending?token=secret').json
    assert data['entries'][0]['issue'] == 'ISSUE-001' and data['cursor'] == 1
    assert client.get('/internal/feedback/pending?token=secret&cursor=1').json['entries'] == []
    body = {'issue':'ISSUE-001','status':'fixed','resolution_note':'修复空白','commit_ref':'abc'}
    assert client.post('/internal/feedback/resolve?token=secret', json=body).status_code == 200
    agent.process_one()
    assert agent._client.reply.call_args.args[:2] == ('m1', 'ISSUE-001 已修复并上线：修复空白 (commit abc)')
    client.post('/internal/feedback/resolve?token=secret', json=body)
    assert len(client.get('/internal/feedback/pending?token=secret&cursor=1').json['entries']) == 1
    with agent.connect() as db: assert db.execute("select count(*) from bot_replies where message_id='bot-message'").fetchone()[0] == 1
    agent.enqueue('bot-message', 'chat', 'ou_x', 'oops', '')
    with agent.connect() as db: assert db.execute("select count(*) from jobs where message_id='bot-message'").fetchone()[0] == 0
    app.job_queue.stop_all()


def test_feishu_client_caches_token_and_threads():
    client = FeishuClient('https://example.test', 'id', 'secret')
    client._request = Mock(side_effect=[{'tenant_access_token':'t','expire':7200}, {'data':{'message_id':'r1'}}, {'data':{'message_id':'r2'}}])
    assert client.reply('original', '你好', 'one') == 'r1'
    assert client.reply('original', '再见', 'two') == 'r2'
    assert client._request.call_count == 3
    assert client._request.call_args.args[1]['reply_in_thread'] is True
