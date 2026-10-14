from __future__ import annotations

import html
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.paper_reader.app import create_app
from src.paper_reader.config import load_config
from src.paper_reader.discovery import (
    DailyDiscoveryScheduler,
    DiscoveryCandidate,
    DiscoveryService,
    parse_huggingface_daily,
    parse_scoring_output,
)
from src.paper_reader.netutils import FetchedResponse
from src.paper_reader.team_store import TeamStore


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, messages, *, model, session_id=None):  # noqa: ANN001
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if "候选：" not in prompt:
            return "重点关注具身智能、机器人感知和视觉学习，偏好可复现的系统与方法贡献。"
        candidates = json.loads(prompt.split("候选：\n", 1)[1])
        return json.dumps(
            [{"id": item["id"], "score": 8.5, "reason": "与机器人视觉和可复现系统方向直接相关。"} for item in candidates],
            ensure_ascii=False,
        )


def hf_html() -> str:
    payload = {
        "dateString": "2026-10-14",
        "dailyPapers": [
            {
                "title": "Robot Visual Learning",
                "summary": "A paper about vision for robots.",
                "paper": {"id": "2601.00001", "publishedAt": "2026-10-14T00:00:00Z"},
            }
        ],
    }
    return f'<div data-target="DailyPapers" data-props="{html.escape(json.dumps(payload))}"></div>'


def atom_xml(identifier: str, title: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>http://arxiv.org/abs/{identifier}v2</id><title>{title}</title>
      <summary>Detailed abstract.</summary><published>2026-10-14T00:00:00Z</published></entry>
    </feed>"""


class DiscoveryPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.config = load_config(
            project_dir=self.root,
            cwd=self.root,
            environ={
                "PAPER_READER_DATA_DIR": str(self.root / "data"),
                "PAPER_READER_LIBRARY_DIR": str(self.root / "library"),
                "PAPER_READER_DISCOVERY_ENABLED": "false",
                "PAPER_READER_LLM_API_KEY": "test-key",
            },
        )
        self.store = TeamStore(self.config.library_dir)
        self.store.sync_papers([
            SimpleNamespace(
                rel_path="kept.pdf", file_name="kept.pdf", folder="", extension=".pdf",
                title="Embodied Robot Vision", display_title="Embodied Robot Vision",
                preview_text="", extracted_date=None, sort_date="2026-10-01", date_precision=None,
                date_source=None, file_size=1, modified_at="2026-10-01T00:00:00",
                preview_kind="file", resource_kind="paper", source_url=None, source_domain=None,
                fetch_status="ok", fetched_at=None, author=None, site_name=None, is_done=False,
            )
        ])
        self.client = FakeClient()
        self.service = DiscoveryService(store=self.store, config=self.config, client=self.client)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_huggingface_parsing_uses_canonical_arxiv_url(self) -> None:
        candidates = parse_huggingface_daily(hf_html())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].url, "https://arxiv.org/abs/2601.00001")
        self.assertEqual(candidates[0].source, "huggingface")

    def test_run_fetches_each_source_once_dedupes_and_scores(self) -> None:
        def fake_fetch(url: str, **kwargs):  # noqa: ANN001
            body = hf_html() if "huggingface.co" in url else atom_xml(
                "2601.00001" if "cs.RO" in url else f"2601.0000{2 + ['cs.CV', 'cs.LG', 'cs.AI'].index(next(cat for cat in ['cs.CV', 'cs.LG', 'cs.AI'] if cat in url))}",
                "Robot Visual Learning" if "cs.RO" in url else "New Research",
            )
            return FetchedResponse(url, url, 200, {"Content-Type": "application/xml"}, body.encode())

        with patch("src.paper_reader.discovery.fetch_public_url", side_effect=fake_fetch) as fetch:
            result = self.service.run()

        self.assertEqual(fetch.call_count, 5)
        self.assertEqual(result.fetched, 5)
        self.assertEqual(result.eligible, 4)
        self.assertEqual(result.surfaced, 4)
        self.assertEqual(len(self.store.list_discoveries()), 4)
        self.assertEqual(len(self.client.calls), 2)  # one profile + one <=20 candidate score batch
        self.assertEqual(self.service.interest_profile(), result.profile)
        self.assertEqual(len(self.client.calls), 2)  # cached for 24 hours; no second profile call

    def test_eligible_candidates_excludes_library_arxiv_and_prior_rows(self) -> None:
        now = "2026-10-14T00:00:00"
        with self.store._connect() as conn:  # setup a pre-existing canonical library source
            conn.execute(
                """INSERT INTO papers(rel_path,file_name,folder,extension,title,display_title,preview_text,file_size,modified_at,preview_kind,created_at,updated_at)
                VALUES ('x.pdf','x.pdf','','.pdf','X','X','',0,?,'file',?,?)""",
                (now, now, now),
            )
            paper_id = conn.execute("SELECT id FROM papers WHERE rel_path='x.pdf'").fetchone()[0]
            conn.execute(
                "INSERT INTO paper_sources(paper_id,source_type,source_value,created_at) VALUES (?, 'arxiv', '2601.10001', ?)",
                (paper_id, now),
            )
        self.store.add_discoveries([
            {"source": "arxiv:cs.CV", "external_id": "2601.10002", "title": "Old", "url": "https://arxiv.org/abs/2601.10002", "score": 7, "reason": "old", "surfaced_at": now}
        ])
        candidates = [
            DiscoveryCandidate("arxiv:cs.RO", "2601.10001", "In library", "https://arxiv.org/abs/2601.10001", ""),
            DiscoveryCandidate("arxiv:cs.CV", "2601.10002", "Already surfaced", "https://arxiv.org/abs/2601.10002", ""),
            DiscoveryCandidate("arxiv:cs.AI", "2601.10003", "Fresh", "https://arxiv.org/abs/2601.10003", ""),
        ]
        self.assertEqual([item.external_id for item in self.service.eligible_candidates(candidates)], ["2601.10003"])

    def test_scoring_parser_clamps_scores_and_ignores_unknown_ids(self) -> None:
        batch = [DiscoveryCandidate("arxiv:cs.AI", "2601.20001", "One", "https://arxiv.org/abs/2601.20001", "")]
        parsed = parse_scoring_output("```json\n[{\"id\": \"2601.20001\", \"score\": 14, \"reason\": \"很相关\"}, {\"id\": \"bad\", \"score\": 9, \"reason\": \"no\"}]\n```", batch)
        self.assertEqual(parsed, {"2601.20001": (10.0, "很相关")})

    def test_status_transitions_are_one_way(self) -> None:
        self.store.add_discoveries([
            {"source": "arxiv:cs.AI", "external_id": "2601.30001", "title": "Decision", "url": "https://arxiv.org/abs/2601.30001", "score": 8, "reason": "fit", "surfaced_at": "2026-10-14T00:00:00"}
        ])
        user = self.store.bootstrap_default_user("admin", "password")
        item = self.store.list_discoveries()[0]
        decided = self.store.decide_discovery(item.id, status="dismissed", decided_by=user.id)
        self.assertEqual(decided.status, "dismissed")
        with self.assertRaises(ValueError):
            self.store.decide_discovery(item.id, status="imported", decided_by=user.id)

    def test_daily_scheduler_runs_once_per_beijing_day(self) -> None:
        calls: list[int] = []
        scheduler = DailyDiscoveryScheduler(SimpleNamespace(run=lambda: calls.append(1) or "done"), hour=7)
        now = datetime(2026, 10, 14, 7, 5, tzinfo=ZoneInfo("Asia/Shanghai"))
        self.assertEqual(scheduler.run_if_due(now), "done")
        self.assertIsNone(scheduler.run_if_due(now))
        self.assertEqual(calls, [1])


class DiscoveryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        config = load_config(
            project_dir=root,
            cwd=root,
            environ={
                "PAPER_READER_DATA_DIR": str(root / "data"),
                "PAPER_READER_LIBRARY_DIR": str(root / "library"),
                "PAPER_READER_LOGIN_USERNAME": "admin",
                "PAPER_READER_LOGIN_PASSWORD": "password",
                "PAPER_READER_DISCOVERY_ENABLED": "false",
            },
        )
        self.app = create_app(config=config)
        self.app.testing = True
        self.app.config["CSRF_PROTECTION_ENABLED"] = False
        self.client = self.app.test_client()
        self.admin = self.app.team_store.get_user_by_username("admin")
        assert self.admin is not None
        with self.client.session_transaction() as session:
            session.update(authenticated=True, user_id=self.admin.id, username="admin", display_name="admin", role="admin")
        self.app.team_store.add_discoveries([
            {"source": "arxiv:cs.RO", "external_id": "2601.40001", "title": "Import Me", "url": "https://arxiv.org/abs/2601.40001", "score": 9, "reason": "机器人相关", "surfaced_at": "2026-10-14T00:00:00"}
        ])

    def tearDown(self) -> None:
        self.app.action_queue.stop()
        self.app.job_queue.stop_all()
        self.app.discovery_scheduler.stop()
        self.tempdir.cleanup()

    def test_import_action_enqueues_discovery_import_and_keeps_candidate_until_success(self) -> None:
        candidate = self.app.team_store.list_discoveries()[0]
        with patch.object(self.app.action_queue, "submit", return_value=SimpleNamespace(title="发现导入")) as submit:
            response = self.client.post(f"/discover/{candidate.id}/import", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.app.team_store.get_discovery(candidate.id).status, "new")
        self.assertEqual(submit.call_args.kwargs["kind"], "discovery-import")
        self.assertEqual(submit.call_args.kwargs["payload"]["discovery_id"], candidate.id)

    def test_dismiss_action_marks_candidate(self) -> None:
        candidate = self.app.team_store.list_discoveries()[0]
        response = self.client.post(f"/discover/{candidate.id}/dismiss", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.app.team_store.get_discovery(candidate.id).status, "dismissed")
