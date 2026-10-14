from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import ANY, patch

from pypdf import PdfWriter

from src.paper_reader.app import create_app, load_env_file_values, normalize_import_target, parse_import_targets
from src.paper_reader.chat_queue import ChatJob
from src.paper_reader.ai_summary import DEFAULT_USER_PROMPT
from src.paper_reader.chat_ai import answer_question_about_document, build_chat_prompt
from src.paper_reader.markdown_render import render_markdown
from src.paper_reader.netutils import FetchedResponse
from src.paper_reader.prompt_manager import DEFAULT_PROMPT_SLUG, LEGACY_DEFAULT_USER_PROMPT, PromptStore
from src.paper_reader.team_store import TeamStore, generate_auto_tags

paper_reader_app_module = importlib.import_module("src.paper_reader.app")
chat_ai_module = importlib.import_module("src.paper_reader.chat_ai")


DOCX_CONTENT_TYPES = """<?xml version='1.0' encoding='UTF-8'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>
  <Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>
</Types>
"""

DOCX_RELS = """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>
  <Relationship Id='rId2' Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties' Target='docProps/core.xml'/>
</Relationships>
"""

DOCX_DOC = """<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
  <w:body>
    <w:p><w:r><w:t>{title}</w:t></w:r></w:p>
    <w:p><w:r><w:t>{body}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

DOCX_CORE = """<?xml version='1.0' encoding='UTF-8'?>
<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
 xmlns:dc='http://purl.org/dc/elements/1.1/'>
  <dc:title>{title}</dc:title>
</cp:coreProperties>
"""

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class PaperReaderAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.library = Path(self.tempdir.name)
        self.source_tempdir = tempfile.TemporaryDirectory()
        self.extra_apps = []
        self.env_file_path = self.library / ".test-no-env"
        self.env_patch = patch.dict(
            os.environ,
            {
                "PAPER_READER_ENV_FILE": str(self.env_file_path),
                "PAPER_READER_SECRET_KEY": "test-session-secret",
                "PAPER_READER_LOGIN_USERNAME": "admin",
                "PAPER_READER_LOGIN_PASSWORD": "test-admin-password",
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.source_root = Path(self.source_tempdir.name)
        self.app = create_app(self.library, source_archive_root=self.source_root)
        self.app.testing = True
        # Legacy integration tests exercise route behavior unrelated to CSRF; focused
        # tests below enable the production-default protection explicitly.
        self.app.config["CSRF_PROTECTION_ENABLED"] = False
        self.client = self.app.test_client()
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        with self.client.session_transaction() as session:
            session["authenticated"] = True
            session["user_id"] = admin_user.id
            session["username"] = admin_user.username
            session["display_name"] = admin_user.display_name
            session["role"] = admin_user.role

    def tearDown(self) -> None:
        try:
            for extra_app in self.extra_apps:
                extra_app.action_queue.stop()
                extra_app.job_queue.stop_all()
            self.app.action_queue.stop()
            self.app.job_queue.stop_all()
            for _ in range(50):
                snapshot = self.app.job_queue.snapshot(limit=5)
                if snapshot["active_count"] == 0:
                    break
                time.sleep(0.01)
        except Exception:
            pass
        self.source_tempdir.cleanup()
        self.tempdir.cleanup()

    def make_pdf(self, path: Path, title: str) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": title})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            writer.write(handle)

    def make_docx(self, path: Path, title: str, body: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", DOCX_CONTENT_TYPES)
            archive.writestr("_rels/.rels", DOCX_RELS)
            archive.writestr("word/document.xml", DOCX_DOC.format(title=title, body=body))
            archive.writestr("docProps/core.xml", DOCX_CORE.format(title=title))

    def fake_markdown_response(self, body: str):
        payload = body.encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return payload

        return FakeResponse()

    def fake_fetched_response(self, payload: bytes, url: str = "https://example.test/resource", content_type: str = "application/octet-stream") -> FetchedResponse:
        return FetchedResponse(
            requested_url=url,
            final_url=url,
            status=200,
            headers={"Content-Type": content_type, "Content-Length": str(len(payload))},
            body=payload,
        )

    def fake_binary_response(self, payload: bytes):
        class FakeResponse:
            def __init__(self):
                self.offset = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size: int = -1):
                if size is None or size < 0:
                    chunk = payload[self.offset :]
                    self.offset = len(payload)
                    return chunk
                chunk = payload[self.offset : self.offset + size]
                self.offset += len(chunk)
                return chunk

        return FakeResponse()

    def make_feishu_app(self, *, auth_mode: str | None = None, allowed_chat_id: str | None = None):
        env_file = self.library / ".env.feishu"
        settings = [
            "PAPER_READER_SECRET_KEY=test-feishu-secret",
            "PAPER_READER_BASE_URL=https://reader.example.test",
            "PAPER_READER_FEISHU_APP_ID=cli_test_paper_reader",
            "PAPER_READER_FEISHU_APP_SECRET=test-feishu-app-secret",
        ]
        if auth_mode:
            settings.append(f"PAPER_READER_AUTH_MODE={auth_mode}")
        if allowed_chat_id:
            settings.append(f"PAPER_READER_FEISHU_ALLOWED_CHAT_ID={allowed_chat_id}")
        env_file.write_text(
            "\n".join(settings) + "\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"PAPER_READER_ENV_FILE": str(env_file)}, clear=False):
            app = create_app(self.library, source_archive_root=self.source_root)
        app.testing = True
        app.config["CSRF_PROTECTION_ENABLED"] = False
        self.extra_apps.append(app)
        return app

    def login_client_as(self, client, username: str) -> None:
        user = self.app.team_store.get_user_by_username(username)
        assert user is not None
        with client.session_transaction() as session:
            session["authenticated"] = True
            session["user_id"] = user.id
            session["username"] = user.username
            session["display_name"] = user.display_name
            session["role"] = user.role

    def create_prompt(self, slug: str, name: str) -> None:
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name=name,
            slug=slug,
            user_prompt="请直接阅读 `{document_path}`，总结这篇论文。",
            model="gpt-5.4",
            enabled=True,
            auto_run=False,
            created_by_user_id=admin_user.id,
        )

    def create_source_day(self, run_date: str, paper_id: str = "2604.08377", title: str = "SkillClaw") -> Path:
        year, month, day = run_date.split("-")
        day_dir = self.source_root / year / month / day
        pdf_dir = day_dir / "papers"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        self.make_pdf(pdf_path, title)
        manifest = {
            "run_reason": "startup",
            "run_date_beijing": run_date,
            "saved_at_beijing": f"{run_date}T18:30:00+08:00",
            "saved_at_utc": f"{run_date}T10:30:00+00:00",
            "schedule_timezone": "Asia/Shanghai",
            "schedule_time_beijing": "18:30",
            "source": "huggingface_daily_papers",
            "source_url": "https://huggingface.co/papers",
            "snapshot_date": run_date,
            "filter": {"field": "upvotes", "operator": ">=", "value": 5},
            "paper_count": 1,
            "papers": [
                {
                    "paper_id": paper_id,
                    "title": title,
                    "url": f"https://huggingface.co/papers/{paper_id}",
                    "upvotes": 12,
                    "published_at": f"{run_date}T00:00:00.000Z",
                    "authors": ["Alice", "Bob"],
                    "summary": "A source archive test paper.",
                    "comment_count": 2,
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                    "pdf_rel_path": f"papers/{paper_id}.pdf",
                    "pdf_file_name": f"{paper_id}.pdf",
                    "pdf_downloaded": True,
                }
            ],
        }
        (day_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return pdf_path

    def create_insights_history_cache(self) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": "2026-04-12T12:00:00",
            "source_paper_count": 723,
            "theme_count": 1,
            "themes": [
                {
                    "slug": "agent",
                    "label": "Agent",
                    "description": "从工具调用走向长期任务和真实工作流的智能体路线。",
                    "paper_count": 42,
                    "start_date": "2023-01-01",
                    "end_date": "2026-04-10",
                    "year_counts": [
                        {"year": "2023", "count": 6},
                        {"year": "2024", "count": 12},
                        {"year": "2025", "count": 16},
                        {"year": "2026", "count": 8},
                    ],
                    "history_summary": "Agent 这条线先解决会不会用工具，再转向长期任务、真实环境与记忆管理。",
                    "why_it_emerged": "因为静态问答式 benchmark 无法解释真实助理为什么会在长期流程里失效。",
                    "phase_cards": [
                        {
                            "phase": "2023",
                            "label": "工具调用起步",
                            "summary": "开始证明模型能调工具，但环境仍然很玩具。",
                            "main_shift": "从单轮回答走向最基础的 action execution。",
                            "papers": [
                                {
                                    "paper_id": "P01",
                                    "rel_path": "agent-paper.pdf",
                                    "title": "Agent Paper",
                                    "sort_date": "2023-01-01",
                                }
                            ],
                        }
                    ],
                    "milestones": [
                        {
                            "paper_id": "P01",
                            "title": "Agent Paper",
                            "sort_date": "2023-01-01",
                            "role": "foundation",
                            "why": "它把最早期的 agent workflow 定义了出来。",
                            "rel_path": "agent-paper.pdf",
                        },
                        {
                            "paper_id": "P02",
                            "title": "ClawArena",
                            "sort_date": "2026-04-01",
                            "role": "turning_point",
                            "why": "它把评测重点推进到长期环境、动态更新和信念修正。",
                            "rel_path": "clawarena.pdf",
                        },
                    ],
                    "paradigm_shifts": ["从工具调用走向长期任务", "从静态 benchmark 走向真实环境"],
                    "mainstreaming": ["记忆与验证成为标准配件"],
                    "representative_papers": [],
                    "llm_used": True,
                }
            ],
        }
        status = {
            "state": "ready",
            "stage": "complete",
            "updated_at": "2026-04-12T12:00:00",
            "started_at": "2026-04-12T11:58:00",
            "error": None,
            "message": "历史脉络已生成。",
            "progress": 100,
            "current_theme_index": 1,
            "current_theme_slug": "agent",
            "current_theme_label": "Agent",
            "completed_themes": 1,
            "total_themes": 1,
            "scanned_core_papers": 723,
            "total_core_papers": 723,
            "stop_requested": False,
        }
        (cache_dir / "history_timeline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (cache_dir / "history_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_momentum_cache(self) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": "2026-04-12T12:30:00",
            "anchor_date": "2026-04-12",
            "source_paper_count": 723,
            "window_count": 3,
            "windows": [
                {
                    "days": 30,
                    "label": "过去 30 天",
                    "anchor_date": "2026-04-12",
                    "start_date": "2026-03-14",
                    "paper_count": 18,
                    "previous_paper_count": 9,
                    "summary": "最近 30 天里，Agent 和真实环境评测是最明显的升温带。",
                    "hot_topics": [
                        {
                            "name": "Agent",
                            "count": 8,
                            "previous_count": 3,
                            "momentum_score": 23.4,
                            "why": "Agent 话题在最近窗口里明显加速，且更多论文开始把长期任务和真实工作流放进同一个叙事里。",
                            "papers": [
                                {
                                    "paper_id": "",
                                    "rel_path": "agent-paper.pdf",
                                    "title": "Agent Paper",
                                    "sort_date": "2026-04-01",
                                    "summary": "Agent summary",
                                    "methods": [],
                                    "signals": [],
                                }
                            ],
                        }
                    ],
                    "method_routes": [
                        {
                            "name": "Verification/Executable Checks",
                            "count": 6,
                            "previous_count": 2,
                            "momentum_score": 18.0,
                            "why": "越来越多论文开始把 verifier 和 executable checks 当作默认配件。",
                            "papers": [],
                        }
                    ],
                    "benchmark_focus": [
                        {
                            "name": "Workspace/Real-World Eval",
                            "count": 5,
                            "previous_count": 1,
                            "momentum_score": 17.0,
                            "why": "评测焦点开始从静态问答转向真实环境和 workspace 任务。",
                            "papers": [],
                        }
                    ],
                    "emerging_edges": [
                        {
                            "name": "World Model/Simulation",
                            "kind": "topic",
                            "count": 2,
                            "previous_count": 0,
                            "lifetime_count": 4,
                            "momentum_score": 14.0,
                            "why": "虽然总体样本还少，但最近窗口里世界模型与模拟器方向开始连续出现。",
                            "papers": [],
                        }
                    ],
                    "llm_used": True,
                }
            ],
        }
        status = {
            "state": "ready",
            "stage": "complete",
            "updated_at": "2026-04-12T12:30:00",
            "started_at": "2026-04-12T12:20:00",
            "error": None,
            "message": "Momentum Radar 已生成。",
            "progress": 100,
            "completed_windows": 3,
            "total_windows": 3,
            "current_window_days": 90,
            "current_window_label": "过去 90 天",
            "scanned_core_papers": 723,
            "total_core_papers": 723,
            "stop_requested": False,
        }
        (cache_dir / "momentum_dashboard.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (cache_dir / "momentum_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_opportunity_cache(self) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": "2026-04-12T13:00:00",
            "anchor_date": "2026-04-12",
            "source_paper_count": 723,
            "section_count": 2,
            "history_theme_count": 6,
            "momentum_window_count": 3,
            "sections": [
                {
                    "slug": "unresolved_problems",
                    "label": "未解问题",
                    "description": "哪些问题被不断提起，但到今天仍没有被真正解决。",
                    "summary": "当前最值得继续打的未解问题集中在 Agent 方向里的长期任务稳定性与真实闭环。",
                    "candidate_count": 3,
                    "item_count": 1,
                    "anchor_date": "2026-04-12",
                    "llm_used": True,
                    "items": [
                        {
                            "key": "unresolved-agent",
                            "title": "Agent 方向里的关键未解问题",
                            "stats_line": "历史累计 42 篇；最近 90 天 8 篇",
                            "why": "长期任务里的稳定反馈闭环仍然没有被真正解决，所以这条线虽然热，但问题还远没收敛。",
                            "evidence": "历史上 Agent 一直在强调真实工作流，而最近 momentum 里它又持续升温，说明旧问题还在。",
                            "next_bet": "把长期任务里的 verifier、memory 和环境反馈做成一个真正可比较的实验闭环。",
                            "papers": [
                                {
                                    "paper_id": "",
                                    "rel_path": "agent-paper.pdf",
                                    "title": "Agent Paper",
                                    "sort_date": "2026-04-01",
                                    "summary": "Agent summary",
                                    "methods": [],
                                    "signals": [],
                                }
                            ],
                        }
                    ],
                },
                {
                    "slug": "transfer_gaps",
                    "label": "可迁移的方法空白",
                    "description": "哪些不同领域之间存在方法可迁移但尚未迁移的空白。",
                    "summary": "当前最值得注意的迁移空白，是把 Verification/Executable Checks 更系统地带进 Memory 方向。",
                    "candidate_count": 2,
                    "item_count": 1,
                    "anchor_date": "2026-04-12",
                    "llm_used": True,
                    "items": [
                        {
                            "key": "transfer-verification-memory",
                            "title": "把 Verification/Executable Checks 迁移到 Memory",
                            "stats_line": "方法累计 12 篇；Memory 累计 9 篇；当前组合仅 1 篇",
                            "why": "verification 在 agent 和 benchmark 里已经变成标配，但在 memory 这条线上还没有被系统迁移。",
                            "evidence": "Memory 最近仍在升温，可是可执行验证的渗透率很低。",
                            "next_bet": "设计能直接验证 memory update / retrieval / corruption 的 executable checks。",
                            "papers": [],
                        }
                    ],
                },
            ],
        }
        status = {
            "state": "ready",
            "stage": "complete",
            "updated_at": "2026-04-12T13:00:00",
            "started_at": "2026-04-12T12:50:00",
            "error": None,
            "message": "Opportunity Map 已生成。",
            "progress": 100,
            "completed_sections": 5,
            "total_sections": 5,
            "current_section_slug": "transfer_gaps",
            "current_section_label": "可迁移的方法空白",
            "scanned_core_papers": 723,
            "total_core_papers": 723,
            "stop_requested": False,
        }
        (cache_dir / "opportunity_map.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (cache_dir / "opportunity_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_insights_status(self, payload: dict[str, object]) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "history_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_momentum_status(self, payload: dict[str, object]) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "momentum_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_opportunity_status(self, payload: dict[str, object]) -> None:
        cache_dir = self.library / ".paper-reader-insights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "opportunity_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_login_required_for_index(self) -> None:
        client = self.app.test_client()
        response = client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_login_allows_access_with_correct_credentials(self) -> None:
        client = self.app.test_client()
        response = client.post(
            "/login",
            data={
                "username": "admin",
                "password": "test-admin-password",
                "next": "/",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")

    def test_login_locks_for_five_minutes_after_three_failures(self) -> None:
        client = self.app.test_client()
        guard = self.app.login_guard
        original_now = guard._now
        timeline = {"value": 1000.0}
        guard._now = lambda: timeline["value"]
        self.addCleanup(setattr, guard, "_now", original_now)

        for _ in range(3):
            response = client.post(
                "/login",
                data={"username": "admin", "password": "wrong", "next": "/"},
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("已锁定 5 分钟", html)

        blocked = client.post(
            "/login",
            data={"username": "admin", "password": "test-admin-password", "next": "/"},
            follow_redirects=True,
        )
        self.assertIn("当前已锁定", blocked.get_data(as_text=True))

        timeline["value"] += 301
        success = client.post(
            "/login",
            data={"username": "admin", "password": "test-admin-password", "next": "/"},
            follow_redirects=False,
        )
        self.assertEqual(success.status_code, 302)
        self.assertEqual(success.headers["Location"], "/")

    def test_login_credentials_can_be_overridden_by_env_file(self) -> None:
        env_file = self.library / ".env.custom"
        env_file.write_text(
            "PAPER_READER_LOGIN_USERNAME=reader\nPAPER_READER_LOGIN_PASSWORD=custom-secret-456\n",
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {
                "PAPER_READER_ENV_FILE": str(env_file),
                "PAPER_READER_LOGIN_USERNAME": "reader",
                "PAPER_READER_LOGIN_PASSWORD": "custom-secret-456",
            },
            clear=False,
        ):
            app = create_app(self.library)
        app.testing = True
        app.config["CSRF_PROTECTION_ENABLED"] = False
        client = app.test_client()

        custom_login = client.post(
            "/login",
            data={"username": "reader", "password": "custom-secret-456", "next": "/"},
            follow_redirects=False,
        )
        self.assertEqual(custom_login.status_code, 302)
        self.assertEqual(custom_login.headers["Location"], "/")

    def test_feishu_login_redirect_uses_configured_callback_url(self) -> None:
        app = self.make_feishu_app()
        client = app.test_client()

        response = client.get("/login/feishu?next=/sources", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        location = response.headers["Location"]
        parsed = urlparse(location)
        query = parse_qs(parsed.query)

        self.assertEqual(f"{parsed.scheme}://{parsed.netloc}{parsed.path}", paper_reader_app_module.FEISHU_AUTH_PAGE_URL)
        self.assertEqual(query["app_id"][0], "cli_test_paper_reader")
        self.assertEqual(query["redirect_uri"][0], "https://reader.example.test/auth/feishu/callback")
        self.assertTrue(query["state"][0])

        with client.session_transaction() as session:
            self.assertEqual(session["feishu_oauth_next"], "/sources")
            self.assertEqual(session["feishu_oauth_state"], query["state"][0])

    def test_feishu_callback_creates_member_and_logs_in(self) -> None:
        app = self.make_feishu_app()
        client = app.test_client()

        start = client.get("/login/feishu?next=/sources", follow_redirects=False)
        self.assertEqual(start.status_code, 302)
        with client.session_transaction() as session:
            state = session["feishu_oauth_state"]

        def fake_urlopen(request_obj, timeout=30):
            url = request_obj.full_url
            headers = dict(request_obj.header_items())
            if url == paper_reader_app_module.FEISHU_APP_ACCESS_TOKEN_URL:
                payload = json.loads((request_obj.data or b"{}").decode("utf-8"))
                self.assertEqual(payload["app_id"], "cli_test_paper_reader")
                return self.fake_markdown_response(
                    json.dumps({"code": 0, "msg": "ok", "app_access_token": "app-access-token"})
                )
            if url == paper_reader_app_module.FEISHU_USER_ACCESS_TOKEN_URL:
                payload = json.loads((request_obj.data or b"{}").decode("utf-8"))
                self.assertEqual(headers.get("Authorization"), "Bearer app-access-token")
                self.assertEqual(payload["grant_type"], "authorization_code")
                self.assertEqual(payload["code"], "code-123")
                self.assertEqual(payload["app_id"], "cli_test_paper_reader")
                return self.fake_markdown_response(
                    json.dumps({"code": 0, "msg": "ok", "data": {"access_token": "user-access-token"}})
                )
            if url == paper_reader_app_module.FEISHU_USER_INFO_URL:
                self.assertEqual(headers.get("Authorization"), "Bearer user-access-token")
                return self.fake_markdown_response(
                    json.dumps(
                        {
                            "code": 0,
                            "msg": "ok",
                            "data": {
                                "open_id": "ou_test_feishu",
                                "name": "Feishu Test User",
                            },
                        }
                    )
                )
            raise AssertionError(url)

        with patch.object(paper_reader_app_module, "urlopen", side_effect=fake_urlopen):
            response = client.get(f"/auth/feishu/callback?code=code-123&state={state}", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/sources")

        username = paper_reader_app_module.build_feishu_username("ou_test_feishu")
        user = app.team_store.get_user_by_username(username)
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.display_name, "Feishu Test User")
        self.assertEqual(user.role, "member")

        with client.session_transaction() as session:
            self.assertTrue(session["authenticated"])
            self.assertEqual(session["username"], username)

    def test_auth_mode_configuration_matrix(self) -> None:
        common = {"PAPER_READER_SECRET_KEY": "config-test-secret"}
        local = paper_reader_app_module.load_config(project_dir=self.library, cwd=self.library, environ=common)
        feishu = paper_reader_app_module.load_config(
            project_dir=self.library,
            cwd=self.library,
            environ={
                **common,
                "PAPER_READER_BASE_URL": "https://reader.example.test",
                "PAPER_READER_FEISHU_APP_ID": "cli_test",
                "PAPER_READER_FEISHU_APP_SECRET": "test-secret",
            },
        )
        mixed = paper_reader_app_module.load_config(
            project_dir=self.library,
            cwd=self.library,
            environ={
                **common,
                "PAPER_READER_BASE_URL": "https://reader.example.test",
                "PAPER_READER_FEISHU_APP_ID": "cli_test",
                "PAPER_READER_FEISHU_APP_SECRET": "test-secret",
                "PAPER_READER_AUTH_MODE": "mixed",
                "PAPER_READER_SESSION_COOKIE_SECURE": "false",
            },
        )
        partial = paper_reader_app_module.load_config(
            project_dir=self.library,
            cwd=self.library,
            environ={**common, "PAPER_READER_BASE_URL": "https://reader.example.test"},
        )
        self.assertEqual(local.auth_mode, "local")
        self.assertEqual(feishu.auth_mode, "feishu")
        self.assertEqual(mixed.auth_mode, "mixed")
        self.assertFalse(mixed.session_cookie_secure)
        with self.assertRaises(ValueError):
            create_app(self.library / "partial-oauth", config=partial)
        self.assertNotIn("使用飞书登录", self.client.get("/login").get_data(as_text=True))
        mixed_app = self.make_feishu_app(auth_mode="mixed")
        mixed_page = mixed_app.test_client().get("/login").get_data(as_text=True)
        self.assertIn("使用飞书登录", mixed_page)
        self.assertIn("账号密码登录", mixed_page)

    def test_feishu_mode_hides_local_login_and_disables_registration(self) -> None:
        app = self.make_feishu_app(auth_mode="feishu")
        client = app.test_client()

        page = client.get("/login")
        body = page.get_data(as_text=True)
        self.assertEqual(page.status_code, 200)
        self.assertIn("使用飞书登录", body)
        self.assertNotIn("账号密码登录", body)
        self.assertEqual(client.get("/register").status_code, 404)
        self.assertEqual(client.post("/register").status_code, 404)
        self.assertEqual(client.post("/login", data={"username": "admin"}).status_code, 404)

    def test_first_feishu_user_becomes_admin_in_feishu_mode(self) -> None:
        isolated = tempfile.TemporaryDirectory()
        self.addCleanup(isolated.cleanup)
        root = Path(isolated.name)
        config = paper_reader_app_module.load_config(
            project_dir=root,
            cwd=root,
            environ={
                "PAPER_READER_SECRET_KEY": "first-feishu-admin-secret",
                "PAPER_READER_BASE_URL": "https://reader.example.test",
                "PAPER_READER_FEISHU_APP_ID": "cli_test",
                "PAPER_READER_FEISHU_APP_SECRET": "test-secret",
                "PAPER_READER_AUTH_MODE": "feishu",
            },
        )
        app = create_app(root / "library", source_archive_root=root / "sources", config=config)
        app.testing = True
        app.config["CSRF_PROTECTION_ENABLED"] = False
        self.extra_apps.append(app)
        client = app.test_client()
        state = parse_qs(urlparse(client.get("/login/feishu").headers["Location"]).query)["state"][0]
        with patch.object(paper_reader_app_module, "fetch_feishu_app_access_token", return_value="app-token"), patch.object(
            paper_reader_app_module, "fetch_feishu_user_access_token", return_value="user-token"
        ), patch.object(
            paper_reader_app_module,
            "fetch_feishu_user_profile",
            return_value={"open_id": "ou_first", "name": "First DexGEM Admin"},
        ):
            response = client.get(f"/auth/feishu/callback?code=code-123&state={state}")
        self.assertEqual(response.status_code, 302)
        user = app.team_store.get_user_by_username(paper_reader_app_module.build_feishu_username("ou_first"))
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.role, "admin")

    def test_feishu_group_membership_allows_before_creating_user(self) -> None:
        app = self.make_feishu_app(allowed_chat_id="oc_required")
        client = app.test_client()
        start = client.get("/login/feishu", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["Location"]).query)["state"][0]

        responses = {
            paper_reader_app_module.FEISHU_APP_ACCESS_TOKEN_URL: {"code": 0, "app_access_token": "app-token"},
            paper_reader_app_module.FEISHU_USER_ACCESS_TOKEN_URL: {"code": 0, "data": {"access_token": "user-token"}},
            paper_reader_app_module.FEISHU_USER_INFO_URL: {"code": 0, "data": {"open_id": "ou_allowed", "name": "Allowed"}},
            "https://open.feishu.cn/open-apis/im/v1/chats/oc_required/members/is_in_chat": {"code": 0, "data": {"is_in_chat": True}},
        }

        def fake_open_json_request(url, **kwargs):
            if url.endswith("/is_in_chat"):
                self.assertEqual(kwargs["headers"]["Authorization"], "Bearer user-token")
            return responses[url]

        with patch.object(paper_reader_app_module, "open_json_request", side_effect=fake_open_json_request):
            response = client.get(f"/auth/feishu/callback?code=code-123&state={state}", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIsNotNone(app.team_store.get_user_by_username(paper_reader_app_module.build_feishu_username("ou_allowed")))

    def test_feishu_group_membership_denial_and_failure_create_no_user(self) -> None:
        for membership_response in (
            {"code": 0, "data": {"is_in_chat": False}},
            {"code": 99991679, "msg": "permission denied"},
        ):
            app = self.make_feishu_app(allowed_chat_id="oc_required")
            client = app.test_client()
            start = client.get("/login/feishu", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["Location"]).query)["state"][0]

            def fake_open_json_request(url, **kwargs):
                responses = {
                    paper_reader_app_module.FEISHU_APP_ACCESS_TOKEN_URL: {"code": 0, "app_access_token": "app-token"},
                    paper_reader_app_module.FEISHU_USER_ACCESS_TOKEN_URL: {"code": 0, "data": {"access_token": "user-token"}},
                    paper_reader_app_module.FEISHU_USER_INFO_URL: {"code": 0, "data": {"open_id": "ou_denied", "name": "Denied"}},
                }
                return membership_response if url.endswith("/is_in_chat") else responses[url]

            with patch.object(paper_reader_app_module, "open_json_request", side_effect=fake_open_json_request):
                response = client.get(f"/auth/feishu/callback?code=code-123&state={state}", follow_redirects=False)

            self.assertEqual(response.status_code, 403)
            self.assertIn("不在必需的 DexGEM 飞书群", response.get_data(as_text=True))
            self.assertIsNone(app.team_store.get_user_by_username(paper_reader_app_module.build_feishu_username("ou_denied")))

    def test_csrf_rejects_missing_token_and_accepts_session_token(self) -> None:
        self.app.config["CSRF_PROTECTION_ENABLED"] = True
        rejected = self.client.post("/folders", data={"new_folder": "Blocked"})
        self.assertEqual(rejected.status_code, 403)
        with self.client.session_transaction() as sess:
            sess["csrf_token"] = "csrf-test-token"
        accepted = self.client.post("/folders", data={"new_folder": "Allowed", "csrf_token": "csrf-test-token"})
        self.assertEqual(accepted.status_code, 302)
        self.assertTrue((self.library / "Allowed").exists())

    def test_login_rotates_session_and_secure_cookie_follows_https_base_url(self) -> None:
        app = self.make_feishu_app()
        self.assertTrue(app.config["SESSION_COOKIE_SECURE"])
        self.assertEqual(app.config["PERMANENT_SESSION_LIFETIME"].total_seconds(), 12 * 60 * 60)

        self.app.config["CSRF_PROTECTION_ENABLED"] = True
        client = self.app.test_client()
        page = client.get("/login")
        self.assertEqual(page.status_code, 200)
        with client.session_transaction() as sess:
            csrf = sess["csrf_token"]
            sess["untrusted_pre_login_value"] = "discard-me"
        response = client.post(
            "/login",
            data={"username": "admin", "password": "test-admin-password", "csrf_token": csrf},
        )
        self.assertEqual(response.status_code, 302)
        with client.session_transaction() as sess:
            self.assertNotIn("untrusted_pre_login_value", sess)
            self.assertNotEqual(sess["csrf_token"], csrf)
            self.assertTrue(sess.permanent)

    def test_register_creates_member_and_logs_in(self) -> None:
        client = self.app.test_client()
        response = client.post(
            "/register",
            data={
                "username": "new-user",
                "display_name": "New User",
                "password": "pw-123",
                "confirm_password": "pw-123",
                "next": "/",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        user = self.app.team_store.get_user_by_username("new-user")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.role, "member")

    def test_team_store_promotes_configured_admins_on_startup(self) -> None:
        self.app.team_store.create_user("configured-admin", "Configured Admin", "secret", "member")
        refreshed = TeamStore(self.library, admin_usernames=("configured-admin",))
        user = refreshed.get_user_by_username("configured-admin")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.role, "admin")

    def test_find_paper_by_source_dedupes_arxiv_versions_by_base_id(self) -> None:
        pdf_path = self.library / "Imports" / "arXiv" / "2601.07155.pdf"
        self.make_pdf(pdf_path, "Stable On-Policy Distillation through Adaptive Target Reformulation")
        record = self.app.library.build_record_for_rel_path("Imports/arXiv/2601.07155.pdf", [])
        self.app.team_store.sync_papers([record])
        import sqlite3
        with sqlite3.connect(self.app.team_store.db_path) as conn:
            paper_id = int(conn.execute("SELECT id FROM papers WHERE rel_path = ?", ("Imports/arXiv/2601.07155.pdf",)).fetchone()[0])
            conn.execute(
                """
                INSERT INTO paper_sources(paper_id, source_type, source_value, source_url, imported_by_user_id, created_at)
                VALUES (?, 'arxiv', '2601.07155v1', 'https://arxiv.org/abs/2601.07155v1', NULL, '2026-04-24T00:00:00')
                """,
                (paper_id,),
            )

        found = self.app.team_store.find_paper_by_source("arxiv", "2601.07155")
        self.assertEqual(found, "Imports/arXiv/2601.07155.pdf")

    def test_remote_import_overwrites_existing_arxiv_when_newer_version_arrives(self) -> None:
        existing_rel_path = "Imports/arXiv/2601.07155.pdf"
        existing_path = self.library / existing_rel_path
        self.make_pdf(existing_path, "Stable On-Policy Distillation v1")
        record = self.app.library.build_record_for_rel_path(existing_rel_path, [])
        self.app.team_store.sync_papers([record])
        self.app.team_store.add_source(
            existing_rel_path,
            source_type="arxiv",
            source_value="2601.07155v1",
            source_url="https://arxiv.org/abs/2601.07155v1",
        )

        upgraded_pdf = self.library / "tmp-upgrade.pdf"
        self.make_pdf(upgraded_pdf, "Stable On-Policy Distillation v2")
        upgraded_bytes = upgraded_pdf.read_bytes()

        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        action = paper_reader_app_module.ActionRecord(
            id="upgrade001",
            kind="remote-import",
            title="远程导入：2601.07155v2",
            source="remote-import",
            requested_by_user_id=admin_user.id,
            requested_by_display_name=admin_user.display_name,
            status="running",
            progress=1,
            message="任务开始执行。",
            error=None,
            result={},
            payload={"raw_targets": "2601.07155v2", "recommendation_reason": ""},
            created_at="2026-04-24T00:00:00",
            updated_at="2026-04-24T00:00:00",
            started_at="2026-04-24T00:00:00",
            finished_at=None,
        )
        handler = self.app.action_queue._handlers["remote-import"]

        with patch.object(paper_reader_app_module, "fetch_public_url", return_value=self.fake_fetched_response(upgraded_bytes, content_type="application/pdf")):
            with patch.object(
                self.app.job_queue,
                "submit",
                return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []},
            ) as mocked_submit:
                result = handler(action, lambda progress, message, payload=None: None, lambda: False)

        self.assertEqual(result["summary"]["saved"], 1)
        refreshed_bytes = existing_path.read_bytes()
        self.assertEqual(refreshed_bytes, upgraded_bytes)
        latest_sources = self.app.team_store.sources_for_rel_path(existing_rel_path)
        self.assertEqual(latest_sources[0]["source_value"], "2601.07155")
        self.assertEqual(latest_sources[0]["source_url"], "https://arxiv.org/abs/2601.07155v2")
        self.assertEqual(mocked_submit.call_args.kwargs["force"], True)

    def test_remote_import_retries_slow_pdf_downloads(self) -> None:
        remote_pdf = self.library / "remote.pdf"
        self.make_pdf(remote_pdf, "Retry Remote Import")
        remote_bytes = remote_pdf.read_bytes()
        remote_pdf.unlink()
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        action = paper_reader_app_module.ActionRecord(
            id="retry001",
            kind="remote-import",
            title="远程导入：2604.02268",
            source="remote-import",
            requested_by_user_id=admin_user.id,
            requested_by_display_name=admin_user.display_name,
            status="running",
            progress=1,
            message="任务开始执行。",
            error=None,
            result={},
            payload={"raw_targets": "2604.02268", "recommendation_reason": ""},
            created_at="2026-04-29T00:00:00",
            updated_at="2026-04-29T00:00:00",
            started_at="2026-04-29T00:00:00",
            finished_at=None,
        )
        handler = self.app.action_queue._handlers["remote-import"]
        calls = {"count": 0}

        def flaky_fetch(url, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("The read operation timed out")
            self.assertEqual(kwargs["timeout"], paper_reader_app_module.REMOTE_PDF_DOWNLOAD_TIMEOUT)
            self.assertEqual(kwargs["max_bytes"], 100 * 1024 * 1024)
            self.assertEqual(kwargs["max_redirects"], 10)
            return self.fake_fetched_response(remote_bytes, url=url, content_type="application/pdf")

        with patch.object(paper_reader_app_module, "fetch_public_url", side_effect=flaky_fetch):
            with patch.object(paper_reader_app_module.time, "sleep") as mocked_sleep:
                with patch.object(
                    self.app.job_queue,
                    "submit",
                    return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []},
                ):
                    result = handler(action, lambda progress, message, payload=None: None, lambda: False)

        self.assertEqual(calls["count"], 2)
        mocked_sleep.assert_called_once()
        self.assertEqual(result["summary"]["saved"], 1)
        saved_path = self.library / "Imports" / "arXiv" / "2604.02268.pdf"
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.read_bytes(), remote_bytes)
        self.assertEqual(saved_path.stat().st_mode & 0o444, 0o444)

    def test_profile_avatar_upload_updates_current_user_and_serves_file(self) -> None:
        response = self.client.post(
            "/profile/avatar",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "paper": "",
                "tab": "source",
                "avatar": (io.BytesIO(PNG_BYTES), "avatar.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        html = response.get_data(as_text=True)
        user = self.app.team_store.get_user_by_username("admin")
        assert user is not None
        assert user.avatar_rel_path is not None
        avatar_path = self.library / ".paper-reader-avatars" / user.avatar_rel_path

        self.assertEqual(response.status_code, 200)
        self.assertTrue(avatar_path.exists())
        self.assertIn("/avatars/", html)

        avatar_response = self.client.get(f"/avatars/{user.avatar_rel_path}")
        self.assertEqual(avatar_response.status_code, 200)
        self.assertEqual(avatar_response.data, PNG_BYTES)

    def test_load_env_file_values_parses_simple_dotenv(self) -> None:
        env_file = self.library / ".env.parse"
        env_file.write_text(
            "# comment\nexport PAPER_READER_LOGIN_USERNAME='reader'\nPAPER_READER_LOGIN_PASSWORD=\"secret\"\n",
            encoding="utf-8",
        )

        values = load_env_file_values(env_file)

        self.assertEqual(values["PAPER_READER_LOGIN_USERNAME"], "reader")
        self.assertEqual(values["PAPER_READER_LOGIN_PASSWORD"], "secret")

    def test_admin_can_create_member_user(self) -> None:
        response = self.client.post(
            "/team/users/save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "username": "alice",
                "display_name": "Alice",
                "password": "alice-pass-123",
                "role": "member",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        created_user = self.app.team_store.get_user_by_username("alice")
        self.assertIsNotNone(created_user)
        self.assertEqual(created_user.role, "member")

        login_client = self.app.test_client()
        login = login_client.post(
            "/login",
            data={"username": "alice", "password": "alice-pass-123", "next": "/"},
            follow_redirects=False,
        )
        self.assertEqual(login.status_code, 302)
        self.assertEqual(login.headers["Location"], "/")

    def test_admin_can_update_member_password_role_and_status(self) -> None:
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        alice = self.app.team_store.get_user_by_username("alice")
        assert alice is not None

        response = self.client.post(
            "/team/users/update",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "user_id": str(alice.id),
                "display_name": "Alice Chen",
                "role": "admin",
                "is_active": "on",
                "new_password": "new-pass-456",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        updated = self.app.team_store.get_user_by_username("alice")
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.display_name, "Alice Chen")
        self.assertEqual(updated.role, "admin")
        self.assertIsNone(self.app.team_store.authenticate_user("alice", "alice-pass-123"))
        self.assertIsNotNone(self.app.team_store.authenticate_user("alice", "new-pass-456"))

        deactivate = self.client.post(
            "/team/users/update",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "user_id": str(alice.id),
                "display_name": "Alice Chen",
                "role": "admin",
            },
            follow_redirects=True,
        )

        self.assertEqual(deactivate.status_code, 200)
        deactivated = self.app.team_store.get_user_by_username("alice")
        self.assertIsNotNone(deactivated)
        assert deactivated is not None
        self.assertFalse(deactivated.is_active)
        self.assertIsNone(self.app.team_store.authenticate_user("alice", "new-pass-456"))

    def test_member_cannot_edit_prompts(self) -> None:
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        client = self.app.test_client()
        self.login_client_as(client, "alice")

        response = client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "方法拆解",
                "slug": "method-breakdown",
                "user_prompt": "请解释方法。",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("只有管理员能处理", response.get_data(as_text=True))
        self.assertIsNone(self.app.prompt_store.get_prompt("method-breakdown"))

    def test_member_upload_uses_managed_inbox_and_cannot_rename_files(self) -> None:
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        client = self.app.test_client()
        self.login_client_as(client, "alice")

        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        uploaded_content = upload_bytes.getvalue()
        upload_bytes.seek(0)

        fake_task = paper_reader_app_module.ActionRecord(
            id="memberupload1",
            kind="upload-import",
            title="上传整理：single.pdf",
            source="upload",
            requested_by_user_id=1,
            requested_by_display_name="Alice",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-19T00:00:00",
            updated_at="2026-04-19T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = client.post(
                "/upload-file",
                data={
                    "target_folder": "secret/admin-only",
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "file": (upload_bytes, "single.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "queued")
        self.assertIsNone(response.json["saved_rel_path"])
        mocked.assert_called_once()
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["target_folder"], "TeamInbox/alice")
        self.assertEqual(payload["staged_files"][0]["original_name"], "single.pdf")

        member_file = self.library / "TeamInbox" / "alice" / "single.pdf"
        member_file.parent.mkdir(parents=True, exist_ok=True)
        member_file.write_bytes(uploaded_content)

        rename_response = client.post(
            "/rename",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "tab": "source",
                "rel_path": "TeamInbox/alice/single.pdf",
                "new_name": "renamed.pdf",
            },
            follow_redirects=True,
        )
        html = rename_response.get_data(as_text=True)

        self.assertEqual(rename_response.status_code, 200)
        self.assertIn("只有管理员能处理", html)
        self.assertTrue((self.library / "TeamInbox" / "alice" / "single.pdf").exists())
        self.assertFalse((self.library / "TeamInbox" / "alice" / "renamed.pdf").exists())

    def test_normalize_import_target_accepts_common_arxiv_shortcuts(self) -> None:
        self.assertEqual(normalize_import_target("arxiv:2501.12948"), "2501.12948")
        self.assertEqual(normalize_import_target("arxiv 2501.12948v2"), "2501.12948v2")
        self.assertEqual(normalize_import_target("arxiv.org/abs/2501.12948"), "https://arxiv.org/abs/2501.12948")
        self.assertEqual(normalize_import_target("abs/2501.12948"), "https://arxiv.org/abs/2501.12948")

    def test_parse_import_targets_accepts_json_array(self) -> None:
        self.assertEqual(
            parse_import_targets('["2501.12948", "https://arxiv.org/abs/2501.12949"]'),
            ["2501.12948", "https://arxiv.org/abs/2501.12949"],
        )

    def test_parse_import_targets_accepts_comma_and_space_separated_items(self) -> None:
        self.assertEqual(
            parse_import_targets("2501.12948, arxiv:2501.12949 https://arxiv.org/abs/2501.12950"),
            ["2501.12948", "2501.12949", "https://arxiv.org/abs/2501.12950"],
        )

    def test_generate_auto_tags_prefers_specialized_ai_terms(self) -> None:
        tags = generate_auto_tags(
            title="DeepSeek LoRA for Finance Coding via On-Policy Distillation",
            preview_text=(
                "Researchers from Tsinghua University and NVIDIA study diffusion-style training, "
                "on-policy distillation, and coding benchmarks for financial reasoning."
            ),
            folder="research/agents",
            extension="pdf",
        )

        self.assertIn("lora", tags)
        self.assertIn("on-policy distillation", tags)
        self.assertIn("finance", tags)
        self.assertIn("coding", tags)
        self.assertTrue(any(tag in tags for tag in ["deepseek", "thu", "nvidia"]))
        self.assertNotIn("distillation", tags)
        self.assertNotIn("models", tags)
        self.assertNotIn("training", tags)
        self.assertNotIn("pdf", tags)

    def test_team_metadata_is_visible_and_searchable(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Robot Policy")
        response = self.client.post(
            "/recommend",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "mode": "save",
                "reason": "",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/tags/add",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "tag_name": "robotics",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/recommend",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "mode": "save",
                "reason": "",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/comments",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "body": "这个结果很适合组会分享。",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("1 人", html)
        self.assertIn("#robotics", html)
        self.assertIn("1 评论", html)
        self.assertIn("协作记录", html)
        self.assertIn("这个结果很适合组会分享。", html)
        self.assertIn("评论", html)

        search_response = self.client.get("/?q=robotics")
        self.assertIn("Robot Policy", search_response.get_data(as_text=True))

    def test_like_toggle_route_now_behaves_like_recommend(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Merged Recommend")

        response = self.client.post(
            "/like-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
            },
            follow_redirects=True,
        )

        html = response.get_data(as_text=True)
        feed = self.app.team_store.recent_recommendation_feed(limit=10)
        feed_paths = [item["rel_path"] for item in feed]

        self.assertIn("点赞已经并入推荐", html)
        self.assertIn("paper.pdf", feed_paths)

    def test_recommend_save_mode_persists_membership_without_reason(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Recommendation Save")

        response = self.client.post(
            "/recommend",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "mode": "save",
                "reason": "",
            },
            follow_redirects=True,
        )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("推荐信息已经保存", html)
        self.assertIn("1 人", html)

    def test_shared_and_private_chat_threads_are_persisted_separately(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Chat Paper")
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        alice_client = self.app.test_client()
        self.login_client_as(alice_client, "alice")

        replies = iter(["团队共享回答", "私人回答"])

        def complete_chat(**kwargs):
            self.app.team_store.update_chat_message(
                kwargs["assistant_message_id"],
                body=next(replies),
                status="completed",
                model="gpt-5.4",
            )

        with patch.object(self.app.chat_queue, "submit", side_effect=complete_chat):
            shared_response = self.client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "shared",
                    "body": "这篇论文的重点是什么？",
                },
                headers={"X-Requested-With": "fetch"},
            )
            private_response = self.client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "private",
                    "body": "只给我的复现建议是什么？",
                },
                headers={"X-Requested-With": "fetch"},
            )

        self.assertEqual(shared_response.status_code, 200)
        self.assertEqual(private_response.status_code, 200)
        shared_payload = shared_response.get_json()
        private_payload = private_response.get_json()
        assert shared_payload is not None
        assert private_payload is not None
        self.assertIn("团队共享回答", str(shared_payload["context"]))
        self.assertIn("私人回答", str(private_payload["context"]))

        admin_html = self.client.get("/?paper=paper.pdf&tab=source").get_data(as_text=True)
        self.assertIn("团队共享回答", admin_html)
        self.assertIn("私人回答", admin_html)
        self.assertIn("一起学", admin_html)
        self.assertIn("我爱学", admin_html)

        alice_html = alice_client.get("/?paper=paper.pdf&tab=source").get_data(as_text=True)
        self.assertIn("团队共享回答", alice_html)
        self.assertNotIn("私人回答", alice_html)

        context_response = self.client.get("/chat/context?paper=paper.pdf")
        self.assertEqual(context_response.status_code, 200)
        context_payload = context_response.get_json()
        assert context_payload is not None
        self.assertIn("团队共享回答", str(context_payload["shared"]))

    def test_team_shared_chat_renders_markdown_for_multiple_users(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Team Chat Paper")
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        self.app.team_store.create_user("bob", "Bob", "bob-pass-123", "member")
        alice_client = self.app.test_client()
        bob_client = self.app.test_client()
        self.login_client_as(alice_client, "alice")
        self.login_client_as(bob_client, "bob")

        def complete_chat(**kwargs):
            self.app.team_store.update_chat_message(
                kwargs["assistant_message_id"],
                body=f"**Paper Bot 回复**\n- 面向 {kwargs['display_name']}\n- 渠道 {kwargs['visibility']}",
                status="completed",
                model="gpt-5.4",
            )

        with patch.object(self.app.chat_queue, "submit", side_effect=complete_chat):
            alice_shared = alice_client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "shared",
                    "body": "Alice 的共享问题",
                },
                headers={"X-Requested-With": "fetch"},
            )
            bob_shared = bob_client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "shared",
                    "body": "Bob 的共享问题",
                },
                headers={"X-Requested-With": "fetch"},
            )
            alice_private = alice_client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "private",
                    "body": "Alice 的私有问题",
                },
                headers={"X-Requested-With": "fetch"},
            )

        self.assertEqual(alice_shared.status_code, 200)
        self.assertEqual(bob_shared.status_code, 200)
        self.assertEqual(alice_private.status_code, 200)

        alice_html = alice_client.get("/?paper=paper.pdf&tab=source").get_data(as_text=True)
        bob_html = bob_client.get("/?paper=paper.pdf&tab=source").get_data(as_text=True)
        alice_context = alice_client.get("/chat/context?paper=paper.pdf").get_json()
        bob_context = bob_client.get("/chat/context?paper=paper.pdf").get_json()

        assert alice_context is not None
        assert bob_context is not None
        self.assertIn("Alice 的共享问题", bob_html)
        self.assertIn("Bob 的共享问题", bob_html)
        self.assertIn("<strong>Paper Bot 回复</strong>", bob_html)
        self.assertIn("Alice 的私有问题", alice_html)
        self.assertNotIn("Alice 的私有问题", bob_html)
        self.assertIn("body_html", str(bob_context["shared"]))
        self.assertIn("<strong>Paper Bot 回复</strong>", str(bob_context["shared"]))
        self.assertIn("Alice 的私有问题", str(alice_context["private"]))
        self.assertNotIn("Alice 的私有问题", str(bob_context["private"]))

    def test_chat_send_route_returns_pending_context_for_async_polling(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Async Chat Paper")

        with patch.object(self.app.chat_queue, "submit", return_value=None) as mocked:
            response = self.client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "private",
                    "body": "请总结这篇论文的核心贡献",
                },
                headers={"X-Requested-With": "fetch"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        assert payload is not None
        self.assertTrue(payload["ok"])
        self.assertIn("Paper Bot 正在思考...", str(payload["context"]["private"]))
        self.assertIn("pending", str(payload["context"]["private"]))
        mocked.assert_called_once()

        context_response = self.client.get("/chat/context?paper=paper.pdf")
        self.assertEqual(context_response.status_code, 200)
        context_payload = context_response.get_json()
        assert context_payload is not None
        self.assertIn("Paper Bot 正在思考...", str(context_payload["private"]))
        self.assertIn("pending", str(context_payload["private"]))

    def test_chat_send_stream_route_streams_markdown_partials(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Streaming Chat Paper")

        with patch.object(
            paper_reader_app_module,
            "stream_answer_question_about_document",
            return_value=iter(["**流式回答**\n\n", "公式 $x^2$。"]),
        ):
            response = self.client.post(
                "/chat/send-stream",
                data={
                    "rel_path": "paper.pdf",
                    "visibility": "private",
                    "body": "请流式回答",
                },
            )
            payload = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/x-ndjson")
        self.assertIn('"event": "init"', payload)
        self.assertIn('"event": "partial"', payload)
        self.assertIn('"event": "done"', payload)
        self.assertIn("<strong>流式回答</strong>", payload)

        context_response = self.client.get("/chat/context?paper=paper.pdf")
        self.assertEqual(context_response.status_code, 200)
        context_payload = context_response.get_json()
        assert context_payload is not None
        self.assertIn("completed", str(context_payload["private"]))
        self.assertIn("<strong>流式回答</strong>", str(context_payload["private"]))

    def test_chat_context_eventually_shows_background_reply(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Background Chat Paper")

        def delayed_answer(*args, **kwargs):
            time.sleep(0.15)
            return "后台线程回答完成"

        with patch("src.paper_reader.chat_queue.answer_question_about_document", side_effect=delayed_answer):
            response = self.client.post(
                "/chat/send",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                    "visibility": "shared",
                    "body": "请给我一个简短总结",
                },
                headers={"X-Requested-With": "fetch"},
            )

            self.assertEqual(response.status_code, 200)
            initial_payload = response.get_json()
            assert initial_payload is not None
            self.assertIn("pending", str(initial_payload["context"]["shared"]))

            completed_payload = None
            for _ in range(40):
                poll = self.client.get("/chat/context?paper=paper.pdf")
                self.assertEqual(poll.status_code, 200)
                completed_payload = poll.get_json()
                assert completed_payload is not None
                if "后台线程回答完成" in str(completed_payload["shared"]):
                    break
                time.sleep(0.05)

        assert completed_payload is not None
        self.assertIn("后台线程回答完成", str(completed_payload["shared"]))
        self.assertIn("completed", str(completed_payload["shared"]))

    def test_index_lists_existing_pdf_docx_and_default_prompt(self) -> None:
        self.make_pdf(self.library / "2501.12948.pdf", "DeepSeek-R1")
        self.make_docx(self.library / "notes.docx", "RL Notes", "submitted on 25 Jan 2025")

        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("DeepSeek-R1", html)
        self.assertIn("RL Notes", html)
        self.assertIn("2025-01", html)
        self.assertIn("核心解读", html)

    def test_upload_submits_background_action(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        fake_task = paper_reader_app_module.ActionRecord(
            id="upload001",
            kind="upload-import",
            title="上传整理（1 个文件）",
            source="upload",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-19T00:00:00",
            updated_at="2026-04-19T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = self.client.post(
                "/upload",
                data={
                    "target_folder": "arxiv/2025",
                    "files": (upload_bytes, "paper.pdf"),
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台上传整理任务已开始：上传整理（1 个文件）", html)
        mocked.assert_called_once()

    def test_upload_file_endpoint_returns_json_for_background_queue(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        fake_task = paper_reader_app_module.ActionRecord(
            id="upload002",
            kind="upload-import",
            title="上传整理：single.pdf",
            source="upload",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-19T00:00:00",
            updated_at="2026-04-19T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = self.client.post(
                "/upload-file",
                data={
                    "target_folder": "incoming",
                    "folder": "incoming",
                    "q": "",
                    "sort": "date_desc",
                    "file": (upload_bytes, "single.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "queued")
        self.assertEqual(response.json["task_id"], "upload002")
        self.assertIn("后台正在查重、入库并安排分析", response.json["message"])
        mocked.assert_called_once()

    def test_upload_file_endpoint_queues_even_if_duplicate_will_be_checked_later(self) -> None:
        original = io.BytesIO(b"same-content")
        duplicate = io.BytesIO(b"same-content")
        (self.library / "existing.pdf").write_bytes(original.getvalue())

        fake_task = paper_reader_app_module.ActionRecord(
            id="upload003",
            kind="upload-import",
            title="上传整理：renamed.pdf",
            source="upload",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-19T00:00:00",
            updated_at="2026-04-19T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = self.client.post(
                "/upload-file",
                data={
                    "target_folder": "",
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "file": (duplicate, "renamed.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "queued")
        self.assertFalse((self.library / "renamed.pdf").exists())
        mocked.assert_called_once()

    def test_upload_route_reports_invalid_files_but_queues_valid_ones(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        fake_task = paper_reader_app_module.ActionRecord(
            id="upload004",
            kind="upload-import",
            title="上传整理（1 个文件）",
            source="upload",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-19T00:00:00",
            updated_at="2026-04-19T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = self.client.post(
                "/upload",
                data={
                    "target_folder": "mixed",
                    "folder": "mixed",
                    "q": "",
                    "sort": "date_desc",
                    "files": [
                        (upload_bytes, "ok.pdf"),
                        (io.BytesIO(b"bad"), "bad.txt"),
                    ],
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台上传整理任务已开始：上传整理（1 个文件）", html)
        self.assertIn("有 1 个文件因为格式问题没有进入后台整理", html)
        self.assertFalse((self.library / "mixed" / "bad.txt").exists())
        mocked.assert_called_once()

    def test_import_link_route_accepts_arxiv_shortcuts(self) -> None:
        fake_task = paper_reader_app_module.ActionRecord(
            id="task123",
            kind="remote-import",
            title="远程导入：2501.12948",
            source="remote-import",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-18T00:00:00",
            updated_at="2026-04-18T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked_submit:
            response = self.client.post(
                "/import-link",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "import_target": "arxiv:2501.12948",
                    "recommendation_reason": "",
                },
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台导入任务已开始：远程导入：2501.12948", html)
        mocked_submit.assert_called_once()

    def test_import_link_route_supports_batch_arxiv_imports(self) -> None:
        fake_task = paper_reader_app_module.ActionRecord(
            id="task456",
            kind="remote-import",
            title="远程导入（2 项）",
            source="remote-import",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-18T00:00:00",
            updated_at="2026-04-18T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked_submit:
            response = self.client.post(
                "/import-link",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "import_target": "2501.12948, arxiv:2501.12949",
                    "recommendation_reason": "适合组会",
                },
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台导入任务已开始：远程导入（2 项）", html)
        mocked_submit.assert_called_once()

    def test_import_link_route_shows_per_item_status_for_mixed_batch_inputs(self) -> None:
        fake_task = paper_reader_app_module.ActionRecord(
            id="task789",
            kind="remote-import",
            title="远程导入（3 项）",
            source="remote-import",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-18T00:00:00",
            updated_at="2026-04-18T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task):
            response = self.client.post(
                "/import-link",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "import_target": (
                        "2501.12948, "
                        "https://openreview.net/pdf?id=test-openreview "
                        "https://example.com/papers/test-paper.pdf"
                    ),
                    "recommendation_reason": "",
                },
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台导入任务已开始：远程导入（3 项）", html)

    def test_prompt_save_route_creates_custom_prompt(self) -> None:
        response = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "方法拆解",
                "slug": "method-breakdown",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，从实现角度解释这篇论文。",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("方法拆解", html)
        self.assertIsNotNone(self.app.prompt_store.get_prompt("method-breakdown"))

    def test_prompt_save_route_auto_generates_slug_for_chinese_name(self) -> None:
        first = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "实验摘要",
                "slug": "",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，总结实验部分。",
            },
            follow_redirects=True,
        )
        second = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "复现建议",
                "slug": "",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，给出复现建议。",
            },
            follow_redirects=True,
        )

        prompts = self.app.prompt_store.list_prompts()
        slugs = [prompt.slug for prompt in prompts]

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("prompt", slugs)
        self.assertIn("prompt-2", slugs)
        self.assertEqual(len(slugs), len(set(slugs)))

    def test_prompt_store_can_manage_multiple_prompts(self) -> None:
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name="方法拆解",
            slug="method-breakdown",
            user_prompt="请直接阅读 `{document_path}`，解释方法。",
            model="gpt-5.4",
            enabled=True,
            auto_run=True,
        )
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name="实验摘要",
            slug="experiment-summary",
            user_prompt="请直接阅读 `{document_path}`，解释实验。",
            model="gpt-5.4",
            enabled=False,
            auto_run=False,
        )

        prompts = self.app.prompt_store.list_prompts()
        names = [prompt.name for prompt in prompts]

        self.assertIn("核心解读", names)
        self.assertIn("方法拆解", names)
        self.assertIn("实验摘要", names)
        self.assertEqual(len(prompts), 3)

    def test_prompt_manager_panel_shows_tag_prompt_controls(self) -> None:
        response = self.client.get("/tool-panels/prompt-manager")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("标签生成 Prompt", html)
        self.assertIn("保存标签 Prompt", html)
        self.assertIn("核心解读完成后自动刷新 AI 标签", html)

    def test_tag_prompt_save_route_updates_config(self) -> None:
        response = self.client.post(
            "/tag-prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "model": "gpt-5.4-mini",
                "enabled": "on",
                "user_prompt": "请直接阅读 `{document_path}`，只输出 JSON 标签数组。",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)
        tag_prompt = self.app.prompt_store.get_tag_prompt()

        self.assertEqual(response.status_code, 200)
        self.assertIn("标签生成 Prompt 已保存", html)
        self.assertTrue(tag_prompt.enabled)
        self.assertEqual(tag_prompt.model, "gpt-5.4-mini")
        self.assertIn("JSON 标签数组", tag_prompt.user_prompt)

    def test_tag_generate_ai_route_replaces_ai_tags(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Tag Prompt Paper")
        visible_slugs = [prompt.slug for prompt in self.app.prompt_store.active_prompts()]
        paper = self.app.library.build_record_for_rel_path("paper.pdf", visible_slugs)
        self.app.team_store.sync_papers([paper])
        self.app.team_store.replace_generated_tags("paper.pdf", ["legacy-tag"], source_type="ai")
        self.app.prompt_store.save_tag_prompt(
            user_prompt="请直接阅读 `{document_path}`，只输出 JSON 标签数组。",
            model="gpt-5.4",
            enabled=True,
        )

        fake_task = paper_reader_app_module.ActionRecord(
            id="tags001",
            kind="ai-tags",
            title="AI 标签刷新：paper.pdf",
            source="ai-tags",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-18T00:00:00",
            updated_at="2026-04-18T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task):
            response = self.client.post(
                "/tags/generate-ai",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "show_done": "",
                    "rel_path": "paper.pdf",
                    "tab": "source",
                },
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI 标签刷新任务已开始：AI 标签刷新：paper.pdf", html)

    def test_generate_prompt_result_refreshes_ai_tags_after_core_prompt(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Auto Tag Paper")
        visible_slugs = [prompt.slug for prompt in self.app.prompt_store.active_prompts()]
        paper = self.app.library.build_record_for_rel_path("paper.pdf", visible_slugs)
        self.app.team_store.sync_papers([paper])
        self.app.prompt_store.save_tag_prompt(
            user_prompt="请直接阅读 `{document_path}`，只输出 JSON 标签数组。",
            model="gpt-5.4",
            enabled=True,
        )
        core_prompt = self.app.prompt_store.get_prompt(DEFAULT_PROMPT_SLUG)
        assert core_prompt is not None

        with patch.object(
            paper_reader_app_module,
            "run_prompt_on_document",
            side_effect=["# 这是一份核心解读", '["robotics", "rag"]'],
        ):
            result_path, generated = self.app.library.generate_prompt_result("paper.pdf", core_prompt, force=True)

        context = self.app.team_store.paper_context("paper.pdf")
        ai_tags = [tag.name for tag in context["tags"] if tag.source_type == "ai"]

        self.assertTrue(generated)
        self.assertTrue(result_path.exists())
        self.assertEqual(ai_tags, ["rag", "robotics"])

    def test_prompt_missing_tab_shows_empty_state(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        self.create_prompt("experiment-read", "实验摘要")

        response = self.client.get("/?paper=paper.pdf&tab=experiment-read")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("这份解读还没生成，所以这里暂时还是空的。", html)
        self.assertIn("实验摘要", html)

    def test_index_renders_compact_workspace_controls(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Compact Workspace")

        response = self.client.get("/?paper=paper.pdf&tab=source")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("mobile-tab-bar", html)
        self.assertIn('data-mobile-tab="library"', html)
        self.assertIn('data-mobile-tab="reader"', html)
        self.assertIn('data-mobile-tab="chat"', html)
        self.assertIn('aria-controls="settings-drawer"', html)
        self.assertIn('id="settings-open-button"', html)
        self.assertIn('role="dialog"', html)
        self.assertIn('aria-labelledby="settings-drawer-title"', html)
        self.assertIn('tabindex="-1"', html)
        self.assertIn('aria-hidden="true"', html)
        self.assertIn('aria-label="折叠阅读区"', html)
        self.assertIn("settings-user-card", html)
        self.assertIn('data-collapsible-toggle="tag-editor"', html)
        self.assertIn("data-chat-form", html)
        self.assertIn("item-header", html)
        self.assertIn("item-details", html)
        self.assertIn("item-actions", html)
        self.assertIn('data-sidebar-switch="recommendations"', html)
        self.assertIn('data-sidebar-switch="library"', html)
        self.assertIn('data-sidebar-panel="library"', html)
        self.assertIn("讨论与笔记", html)
        self.assertIn('data-chat-switch="activity"', html)
        self.assertIn('data-chat-switch="private"', html)
        self.assertIn('data-chat-panel="activity"', html)
        self.assertIn('data-chat-panel="private"', html)
        self.assertIn('data-chat-panel="shared"', html)
        self.assertIn('id="chat-shared" hidden', html)
        self.assertIn("workspace-recommend-button", html)
        self.assertIn("发到一起学", html)
        self.assertIn("围绕这篇论文的交流统一留在评论区", html)
        self.assertIn("compact-tag-toggle", html)
        self.assertIn("添标签", html)
        self.assertIn("原文阅读", html)
        self.assertIn("评论", html)
        self.assertNotIn("workspace-orbit-dock", html)
        self.assertNotIn("workspace-like-button", html)
        self.assertNotIn("viewer-meta-grid", html)
        self.assertNotIn("settings-orb", html)
        self.assertNotIn("推荐和原来的点赞已经合并；这里只显示团队明确想继续读的论文", html)

    def test_workspace_styles_support_dragging_and_mobile_collapses(self) -> None:
        css = (Path(self.app.static_folder) / "style.css").read_text(encoding="utf-8")

        self.assertIn("touch-action: none;", css)
        self.assertIn(
            ".viewer-pane {\n  display: flex;\n  flex-direction: column;",
            css,
        )
        self.assertIn(
            ".pane-center {\n  display: flex;\n  flex-direction: column;",
            css,
        )
        self.assertIn(
            ".workspace-shell.center-collapsed .viewer-context-panel,\n.workspace-shell.center-collapsed .viewer-inline-tags-panel,\n.workspace-shell.center-collapsed .viewer-tag-editor,\n.workspace-shell.center-collapsed .viewer-tabs,\n.workspace-shell.center-collapsed [data-center-stage] {\n  display: none;",
            css,
        )
        self.assertIn(
            ".workspace-shell.left-collapsed .pane-left,\n  .workspace-shell.center-collapsed .pane-center,\n  .workspace-shell.right-collapsed .pane-right {\n    display: none;",
            css,
        )
        self.assertIn(
            "body.settings-open .settings-orb:not(.dragging) {\n  transform: rotate(8deg) scale(0.96);\n  z-index: 54;\n}",
            css,
        )

    def test_workspace_template_hides_collapsed_regions_from_keyboard_navigation(self) -> None:
        template = (Path(self.app.template_folder) / "index.html").read_text(encoding="utf-8")

        self.assertIn('pane.toggleAttribute("inert", collapsed);', template)
        self.assertIn('pane.setAttribute("aria-hidden", collapsed ? "true" : "false");', template)
        self.assertIn("initializeSidebarPanelSwitch();", template)
        self.assertIn('settingsDrawer.toggleAttribute("inert", !layoutState.settingsOpen);', template)
        self.assertIn('settingsDrawer.setAttribute("aria-hidden", layoutState.settingsOpen ? "false" : "true");', template)

    def test_sidebar_groups_by_year_month_and_done_toggle(self) -> None:
        self.make_pdf(self.library / "2025-paper.pdf", "2025 Paper")
        self.make_docx(self.library / "2024-notes.docx", "2024 Notes", "submitted on 11 Dec 2024")

        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("paper-year-group", html)
        self.assertIn("2025", html)
        self.assertIn("2024-12", html)
        self.assertIn("含已读", html)

    def test_done_toggle_marks_current_user_only_and_hides_it_by_default(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Done Me")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# Cached\n", encoding="utf-8")
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        alice_client = self.app.test_client()
        self.login_client_as(alice_client, "alice")

        response = self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "tab": "source",
                "rel_path": "paper.pdf",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.library / "paper.pdf").exists())
        self.assertTrue(result_path.exists())
        self.assertNotIn("Done Me", html)
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.assertTrue(self.app.team_store.is_done_for_user("paper.pdf", admin_user.id))

        show_done_response = self.client.get("/?show_done=1")
        self.assertIn("Done Me", show_done_response.get_data(as_text=True))

        alice_response = alice_client.get("/")
        self.assertIn("Done Me", alice_response.get_data(as_text=True))

    def test_done_toggle_can_restore_paper(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Restore Me")
        response = self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "tab": "source",
                "rel_path": "paper.pdf",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        restore_response = self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "1",
                "tab": "source",
                "rel_path": "paper.pdf",
            },
            follow_redirects=True,
        )

        self.assertEqual(restore_response.status_code, 200)
        self.assertTrue((self.library / "paper.pdf").exists())
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.assertFalse(self.app.team_store.is_done_for_user("paper.pdf", admin_user.id))

    def test_recommendation_feed_hides_my_read_papers_but_not_other_users(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Feed Done State")
        self.app.team_store.create_user("alice", "Alice", "alice-pass-123", "member")
        alice_client = self.app.test_client()
        self.login_client_as(alice_client, "alice")

        self.client.post(
            "/recommend",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "rel_path": "paper.pdf",
                "tab": "source",
                "mode": "save",
                "reason": "",
            },
            follow_redirects=True,
        )
        self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "tab": "source",
                "rel_path": "paper.pdf",
            },
            follow_redirects=True,
        )

        admin_html = self.client.get("/").get_data(as_text=True)
        alice_html = alice_client.get("/").get_data(as_text=True)

        self.assertNotIn("Feed Done State", admin_html)
        self.assertIn("Feed Done State", alice_html)

    def test_index_reads_from_persisted_active_index_without_rescanning_tree(self) -> None:
        self.make_pdf(self.library / "cached-paper.pdf", "Cached Paper")
        self.app.library.rebuild_active_index(lightweight=True)

        fresh_app = create_app(self.library)
        fresh_app.testing = True
        client = fresh_app.test_client()
        with client.session_transaction() as session:
            session["authenticated"] = True
            session["username"] = "admin"

        with patch.object(fresh_app.library, "iter_documents", side_effect=AssertionError("should not rescan tree")):
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("cached-paper.pdf", response.get_data(as_text=True))

    def test_personal_done_state_follows_rename_and_delete(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Done Index")
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.app.library.rebuild_active_index(lightweight=True)
        self.app.team_store.sync_papers(self.app.library.scan(force=True).papers)
        self.app.team_store.set_done_state("paper.pdf", admin_user.id, is_done=True)

        renamed_rel_path = self.app.library.rename_file("paper.pdf", "renamed.pdf")
        active_prompt_slugs = [prompt.slug for prompt in self.app.prompt_store.active_prompts()]
        moved_paper = self.app.library.build_record_for_rel_path(renamed_rel_path, active_prompt_slugs)
        self.app.team_store.rename_paper("paper.pdf", moved_paper)

        self.assertTrue(self.app.team_store.is_done_for_user("renamed.pdf", admin_user.id))

        self.app.library.delete_file(renamed_rel_path)
        self.app.team_store.delete_paper(renamed_rel_path)
        self.assertFalse(self.app.team_store.is_done_for_user("renamed.pdf", admin_user.id))

    def test_batch_section_hides_done_papers_even_when_show_done_enabled(self) -> None:
        self.make_pdf(self.library / "todo.pdf", "Todo Paper")
        self.make_pdf(self.library / "done.pdf", "Done Paper")
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.app.library.rebuild_active_index(lightweight=True)
        self.app.team_store.sync_papers(self.app.library.scan(force=True).papers)
        self.app.team_store.set_done_state("done.pdf", admin_user.id, is_done=True)

        response = self.client.get("/tool-panels/batch-run?show_done=1")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Todo Paper", html)
        self.assertNotIn("Done Paper", html)

    def test_batch_section_shows_select_all_controls(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Prompt Target")

        response = self.client.get("/tool-panels/batch-run")
        html = response.get_data(as_text=True)

        self.assertIn('data-check-action="all" data-check-group="prompt"', html)
        self.assertIn('data-check-action="all" data-check-group="paper"', html)
        self.assertIn('data-check-action="none" data-check-group="prompt"', html)
        self.assertIn('data-check-action="none" data-check-group="paper"', html)
        self.assertIn('data-batch-select-all', html)
        self.assertIn("全选全部匹配结果", html)
        self.assertIn("选择分析模板", html)
        self.assertIn("批量生成所选结果", html)

    def test_batch_section_can_optionally_include_done_papers(self) -> None:
        self.make_pdf(self.library / "todo.pdf", "Todo Paper")
        self.make_pdf(self.library / "done.pdf", "Done Paper")
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.app.library.rebuild_active_index(lightweight=True)
        self.app.team_store.sync_papers(self.app.library.scan(force=True).papers)
        self.app.team_store.set_done_state("done.pdf", admin_user.id, is_done=True)

        default_response = self.client.get("/tool-panels/batch-run")
        include_response = self.client.get("/tool-panels/batch-run?batch_show_done=1")

        default_html = default_response.get_data(as_text=True)
        include_html = include_response.get_data(as_text=True)

        self.assertEqual(default_response.status_code, 200)
        self.assertEqual(include_response.status_code, 200)
        self.assertIn("Todo Paper", default_html)
        self.assertNotIn("Done Paper", default_html)
        self.assertIn("Done Paper", include_html)

    def test_batch_section_paginates_large_result_sets(self) -> None:
        for index in range(55):
            self.make_pdf(self.library / f"paper-{index:02d}.pdf", f"Paper {index:02d}")

        first_response = self.client.get("/tool-panels/batch-run")
        second_response = self.client.get("/tool-panels/batch-run?batch_page=2")
        first_html = first_response.get_data(as_text=True)
        second_html = second_response.get_data(as_text=True)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertIn("共 55 篇", first_html)
        self.assertIn("第 1/2 页", first_html)
        self.assertIn("Paper 54", first_html)
        self.assertNotIn("Paper 04", first_html)
        self.assertIn("Paper 04", second_html)
        self.assertNotIn("Paper 54", second_html)

    def test_prompt_batch_route_can_select_all_filtered_papers_across_pages(self) -> None:
        for index in range(53):
            self.make_pdf(self.library / f"paper-{index:02d}.pdf", f"Paper {index:02d}")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 53, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-batch-run",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "paper-52.pdf",
                    "tab": "source",
                    "batch_page": "2",
                    "select_all_filtered": "1",
                    "prompt_slugs": ["core-zh"],
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        called_rel_paths = mocked.call_args.args[0]
        self.assertEqual(len(called_rel_paths), 53)
        self.assertIn("paper-00.pdf", called_rel_paths)
        self.assertIn("paper-52.pdf", called_rel_paths)

    def test_prompt_batch_route_select_all_filtered_includes_subfolders(self) -> None:
        for index in range(17):
            self.make_pdf(self.library / "root" / f"paper-{index:02d}.pdf", f"Root {index:02d}")
        for index in range(9):
            self.make_pdf(self.library / "root" / "nested" / f"nested-{index:02d}.pdf", f"Nested {index:02d}")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 26, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-batch-run",
                data={
                    "folder": "root",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "root/paper-16.pdf",
                    "tab": "source",
                    "batch_page": "1",
                    "select_all_filtered": "1",
                    "prompt_slugs": ["core-zh"],
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        called_rel_paths = mocked.call_args.args[0]
        self.assertEqual(len(called_rel_paths), 26)
        self.assertIn("root/paper-00.pdf", called_rel_paths)
        self.assertIn("root/nested/nested-08.pdf", called_rel_paths)

    def test_prompt_tab_renders_markdown_html(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            "# 标题\n\n## 小节\n\n- 列表项\n\n包含 **加粗**、`code` 和 $E=mc^2$。\n\n<script>alert('x')</script>\n",
            encoding="utf-8",
        )

        response = self.client.get("/?paper=paper.pdf&tab=core-zh")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('<article class="markdown-viewer markdown-render">', html)
        self.assertIn("<h1>标题</h1>", html)
        self.assertIn("<h2>小节</h2>", html)
        self.assertIn("<strong>加粗</strong>", html)
        self.assertIn("<code>code</code>", html)
        self.assertIn("$E=mc^2$", html)
        self.assertIn("vendor/mathjax/tex-svg.js", html)
        self.assertIn("&lt;script&gt;alert", html)
        self.assertNotIn("<script>alert", html)

    def test_generate_prompt_result_skips_existing_markdown(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        existing_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("# Existing\n", encoding="utf-8")

        with patch("src.paper_reader.ai_summary.run_prompt_on_document", side_effect=AssertionError("should not run")):
            path, generated = self.app.library.generate_prompt_result("paper.pdf", prompt, force=False)

        self.assertFalse(generated)
        self.assertEqual(path, existing_path)

    def test_rename_and_delete_file_move_prompt_results(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# Cached\n", encoding="utf-8")

        rename_response = self.client.post(
            "/rename",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "tab": prompt.slug,
                "rel_path": "paper.pdf",
                "new_name": "paper-renamed.pdf",
            },
            follow_redirects=True,
        )
        self.assertEqual(rename_response.status_code, 200)
        moved_result = self.app.library.prompt_result_path_for("paper-renamed.pdf", prompt.slug)
        self.assertTrue((self.library / "paper-renamed.pdf").exists())
        self.assertTrue(moved_result.exists())
        self.assertFalse(result_path.exists())

        delete_response = self.client.post(
            "/delete",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "rel_path": "paper-renamed.pdf",
            },
            follow_redirects=True,
        )
        self.assertEqual(delete_response.status_code, 200)
        self.assertFalse((self.library / "paper-renamed.pdf").exists())
        self.assertFalse(moved_result.exists())

    def test_prompt_batch_route_runs_selected_prompts(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        self.create_prompt("method-breakdown", "方法拆解")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 2, "existing": 0, "skipped": 1, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-batch-run",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "paper.pdf",
                    "tab": "source",
                    "rel_paths": ["paper.pdf"],
                    "prompt_slugs": ["core-zh", "method-breakdown"],
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            ["paper.pdf"],
            ["core-zh", "method-breakdown"],
            force=False,
            source="batch",
            requested_by_user_id=ANY,
            requested_by_display_name="admin",
        )

    def test_prompt_run_route_submits_background_job(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-run",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "rel_path": "paper.pdf",
                    "prompt_slug": "core-zh",
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            ["paper.pdf"],
            ["core-zh"],
            force=False,
            source="manual",
            requested_by_user_id=ANY,
            requested_by_display_name="admin",
        )

    def test_serve_file_tolerates_zoom_fragment_encoded_in_path(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Zoom Safe")

        response = self.client.get("/files/paper.pdf%23zoom=page-width")
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")

    def test_offline_package_route_builds_complete_zip_bundle(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Offline PDF")
        self.make_docx(self.library / "notes.docx", "Offline Notes", "docx body for offline preview")
        self.create_prompt("method-breakdown", "方法拆解")

        core_prompt = self.app.prompt_store.get_prompt("core-zh")
        assert core_prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", core_prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# 离线摘要\n\n- 要点一\n", encoding="utf-8")

        response = self.client.post(
            "/offline-package",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "paper.pdf",
                "tab": "source",
                "rel_paths": ["paper.pdf", "notes.docx"],
            },
        )
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")

        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())

        self.assertIn("index.html", names)
        self.assertIn("manifest.json", names)
        self.assertIn("assets/style.css", names)
        self.assertIn("assets/offline-reader.css", names)
        self.assertIn("assets/offline-reader.js", names)
        self.assertIn("assets/vendor/mathjax/tex-svg.js", names)
        self.assertIn("papers/paper.pdf", names)
        self.assertIn("papers/notes.docx", names)
        self.assertIn("prompt-results/paper.pdf/core-zh.md", names)

        index_html = archive.read("index.html").decode("utf-8")
        manifest_payload = archive.read("manifest.json").decode("utf-8")

        self.assertIn("论文离线阅读包", index_html)
        self.assertIn("offline-manifest", index_html)
        self.assertIn("assets/vendor/mathjax/tex-svg.js", index_html)
        self.assertIn("Offline PDF", manifest_payload)
        self.assertIn("Offline Notes", manifest_payload)

    def test_offline_package_route_ignores_done_papers(self) -> None:
        self.make_pdf(self.library / "todo.pdf", "Todo Export")
        self.make_pdf(self.library / "done.pdf", "Done Export")
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.app.library.rebuild_active_index(lightweight=True)
        self.app.team_store.sync_papers(self.app.library.scan(force=True).papers)
        self.app.team_store.set_done_state("done.pdf", admin_user.id, is_done=True)

        response = self.client.post(
            "/offline-package",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "todo.pdf",
                "tab": "source",
                "rel_paths": ["todo.pdf", "done.pdf"],
            },
        )
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())

        self.assertIn("papers/todo.pdf", names)
        self.assertNotIn("papers/done.pdf", names)

    def test_jobs_status_endpoint_returns_snapshot(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        with patch.object(self.app.job_queue, "snapshot", return_value={"jobs": [{"id": "job-1", "status": "queued"}], "active_count": 1, "queued_count": 1, "running_count": 0, "average_duration_seconds": 12.5, "max_concurrency": 32, "active_executions": 0}) as mocked:
            response = self.client.get("/jobs/status?paper=paper.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["active_count"], 1)
        self.assertEqual(response.json["max_concurrency"], 32)
        mocked.assert_called_once_with(rel_path="paper.pdf")

    def test_job_snapshot_counts_all_queued_jobs_even_when_list_is_limited(self) -> None:
        rel_paths = []
        for index in range(65):
            rel_path = f"paper-{index:03d}.pdf"
            self.make_pdf(self.library / rel_path, f"Paper {index:03d}")
            rel_paths.append(rel_path)

        release = threading.Event()

        def slow_generate(rel_path, prompt, *, force=False, progress_callback=None, should_abort=None, process_callback=None):
            while not release.is_set():
                if should_abort and should_abort():
                    raise InterruptedError("Job interrupted.")
                time.sleep(0.02)
            return self.app.library.prompt_result_path_for(rel_path, prompt.slug), True

        with patch.object(self.app.library, "generate_prompt_result", side_effect=slow_generate):
            self.app.job_queue.submit(rel_paths, ["core-zh"], force=True, source="batch")
            for _ in range(100):
                snapshot = self.app.job_queue.snapshot(limit=32)
                if snapshot["active_count"] == 65:
                    break
                time.sleep(0.02)
            release.set()

        self.assertEqual(snapshot["active_count"], 65)
        self.assertEqual(snapshot["queued_count"] + snapshot["running_count"], 65)
        self.assertEqual(len(snapshot["jobs"]), 32)

    def test_job_queue_defaults_to_12_workers(self) -> None:
        self.assertEqual(self.app.job_queue.max_concurrency, 12)
        self.assertEqual(len(self.app.job_queue._workers), 32)

    def test_jobs_config_route_updates_max_concurrency(self) -> None:
        response = self.client.post(
            "/jobs/config",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "max_concurrency": "7",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.app.job_queue.max_concurrency, 7)
        self.assertEqual(self.app.settings_store.max_concurrency(), 7)

    def test_jobs_stop_all_route_interrupts_running_and_queued_jobs(self) -> None:
        self.make_pdf(self.library / "paper-a.pdf", "Paper A")
        self.make_pdf(self.library / "paper-b.pdf", "Paper B")
        self.app.job_queue.update_max_concurrency(1)

        started = threading.Event()
        release = threading.Event()

        def slow_generate(rel_path, prompt, *, force=False, progress_callback=None, should_abort=None, process_callback=None):
            started.set()
            while not release.is_set():
                if should_abort and should_abort():
                    raise InterruptedError("Job interrupted.")
                time.sleep(0.02)
            return self.app.library.prompt_result_path_for(rel_path, prompt.slug), True

        with patch.object(self.app.library, "generate_prompt_result", side_effect=slow_generate):
            self.app.job_queue.submit(["paper-a.pdf", "paper-b.pdf"], ["core-zh"], force=True, source="batch")
            self.assertTrue(started.wait(timeout=2))

            for _ in range(100):
                snapshot = self.app.job_queue.snapshot()
                if snapshot["running_count"] == 1 and snapshot["queued_count"] >= 1:
                    break
                time.sleep(0.02)

            response = self.client.post(
                "/jobs/stop-all",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "",
                    "tab": "source",
                },
                follow_redirects=True,
            )
            release.set()

        self.assertEqual(response.status_code, 200)
        for _ in range(100):
            snapshot = self.app.job_queue.snapshot()
            if snapshot["active_count"] == 0:
                break
            time.sleep(0.02)
        statuses = {job.status for job in self.app.job_queue.list_jobs(limit=10)}
        self.assertIn("stopped", statuses)

    def test_reindex_route_uses_lightweight_scan_without_metadata_extraction(self) -> None:
        (self.library / "manual-added.pdf").write_bytes(b"%PDF-1.4\n")
        done_dir = self.library / "DONE"
        done_dir.mkdir(parents=True, exist_ok=True)
        (done_dir / "manual-done.pdf").write_bytes(b"%PDF-1.4\n")

        before = self.app.library.scan(force=True, include_done=True)
        self.assertNotIn("manual-done.pdf", [paper.file_name for paper in before.papers])

        with patch("src.paper_reader.document_utils.extract_document_metadata", side_effect=AssertionError("should not extract")):
            response = self.client.post(
                "/reindex",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "",
                    "tab": "source",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        cached = self.app.library.scan()
        names = [paper.file_name for paper in cached.papers]
        self.assertIn("manual-added.pdf", names)
        self.assertIn("manual-done.pdf", names)
        self.assertTrue((self.library / "manual-done.pdf").exists())
        self.assertFalse((done_dir / "manual-done.pdf").exists())
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        self.assertTrue(self.app.team_store.is_done_for_user("manual-done.pdf", admin_user.id))

    def test_index_shows_sources_button_in_bottom_tools(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("查看来源归档", html)
        self.assertIn("/sources", html)
        self.assertIn("打开 Insights 页面", html)
        self.assertIn("/insights", html)

    def test_insights_page_renders_history_visualization(self) -> None:
        self.create_insights_history_cache()
        self.create_momentum_cache()
        self.create_opportunity_cache()

        response = self.client.get("/insights")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("历史脉络", html)
        self.assertIn("Agent", html)
        self.assertIn("工具调用起步", html)
        self.assertIn("ClawArena", html)
        self.assertIn("重新生成", html)
        self.assertIn("核心解读扫描：723/723 篇", html)
        self.assertIn("Trend Radar", html)
        self.assertIn("Workspace/Real-World Eval", html)
        self.assertIn("World Model/Simulation", html)
        self.assertIn("Opportunity Map", html)
        self.assertIn("Agent 方向里的关键未解问题", html)
        self.assertIn("把 Verification/Executable Checks 迁移到 Memory", html)

    def test_insights_rebuild_route_starts_background_generation(self) -> None:
        with patch.object(self.app.history_store, "start_or_resume", return_value=True) as mocked:
            response = self.client.post("/insights/history/rebuild", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_insights_page_shows_continue_for_stopped_history_job(self) -> None:
        self.create_insights_history_cache()
        self.write_insights_status(
            {
                "state": "stopped",
                "stage": "stopped",
                "updated_at": "2026-04-12T12:10:00",
                "started_at": "2026-04-12T11:58:00",
                "error": None,
                "message": "历史脉络任务已停止，可稍后继续。",
                "progress": 64,
                "current_theme_index": 1,
                "current_theme_slug": "agent",
                "current_theme_label": "Agent",
                "completed_themes": 1,
                "total_themes": 3,
                "scanned_core_papers": 412,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        response = self.client.get("/insights")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("历史脉络生成已暂停", html)
        self.assertIn(">继续<", html)
        self.assertIn("64%", html)
        self.assertIn("核心解读扫描：412/723 篇", html)

    def test_insights_status_marks_stale_running_job_as_stopped(self) -> None:
        self.write_insights_status(
            {
                "state": "running",
                "stage": "theme_generation",
                "updated_at": "2026-04-12T12:05:00",
                "started_at": "2026-04-12T11:58:00",
                "error": None,
                "message": "正在生成第 2/3 条历史主线：Agent",
                "progress": 57,
                "current_theme_index": 2,
                "current_theme_slug": "agent",
                "current_theme_label": "Agent",
                "completed_themes": 1,
                "total_themes": 3,
                "scanned_core_papers": 301,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        snapshot = self.app.history_store.status_snapshot()

        self.assertEqual(snapshot["state"], "stopped")
        self.assertTrue(snapshot["can_continue"])
        self.assertFalse(snapshot["can_stop"])
        self.assertEqual(snapshot["scanned_core_papers"], 301)
        self.assertEqual(snapshot["total_core_papers"], 723)

    def test_insights_stop_route_requests_background_stop(self) -> None:
        with patch.object(self.app.history_store, "request_stop", return_value=True) as mocked:
            response = self.client.post("/insights/history/stop", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_momentum_rebuild_route_starts_background_generation(self) -> None:
        with patch.object(self.app.momentum_store, "start_or_resume", return_value=True) as mocked:
            response = self.client.post("/insights/momentum/rebuild", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_momentum_stop_route_requests_background_stop(self) -> None:
        with patch.object(self.app.momentum_store, "request_stop", return_value=True) as mocked:
            response = self.client.post("/insights/momentum/stop", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_insights_page_shows_continue_for_stopped_momentum_job(self) -> None:
        self.create_momentum_cache()
        self.write_momentum_status(
            {
                "state": "stopped",
                "stage": "stopped",
                "updated_at": "2026-04-12T12:36:00",
                "started_at": "2026-04-12T12:20:00",
                "error": None,
                "message": "Momentum Radar 任务已停止，可稍后继续。",
                "progress": 58,
                "completed_windows": 1,
                "total_windows": 3,
                "current_window_days": 60,
                "current_window_label": "过去 60 天",
                "scanned_core_papers": 512,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        response = self.client.get("/insights")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Momentum Radar 生成已暂停", html)
        self.assertIn("研究热区窗口：1/3 个", html)
        self.assertIn("核心解读扫描：512/723 篇", html)

    def test_momentum_status_marks_stale_running_job_as_stopped(self) -> None:
        self.write_momentum_status(
            {
                "state": "running",
                "stage": "window_generation",
                "updated_at": "2026-04-12T12:35:00",
                "started_at": "2026-04-12T12:20:00",
                "error": None,
                "message": "正在生成过去 60 天的 Momentum Radar。",
                "progress": 61,
                "completed_windows": 1,
                "total_windows": 3,
                "current_window_days": 60,
                "current_window_label": "过去 60 天",
                "scanned_core_papers": 480,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        snapshot = self.app.momentum_store.status_snapshot()

        self.assertEqual(snapshot["state"], "stopped")
        self.assertTrue(snapshot["can_continue"])
        self.assertFalse(snapshot["can_stop"])
        self.assertEqual(snapshot["scanned_core_papers"], 480)
        self.assertEqual(snapshot["total_core_papers"], 723)

    def test_opportunity_rebuild_route_starts_background_generation(self) -> None:
        with patch.object(self.app.opportunity_store, "start_or_resume", return_value=True) as mocked:
            response = self.client.post("/insights/opportunity/rebuild", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_opportunity_stop_route_requests_background_stop(self) -> None:
        with patch.object(self.app.opportunity_store, "request_stop", return_value=True) as mocked:
            response = self.client.post("/insights/opportunity/stop", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/insights", response.headers["Location"])
        mocked.assert_called_once_with()

    def test_insights_page_shows_continue_for_stopped_opportunity_job(self) -> None:
        self.create_opportunity_cache()
        self.write_opportunity_status(
            {
                "state": "stopped",
                "stage": "stopped",
                "updated_at": "2026-04-12T13:06:00",
                "started_at": "2026-04-12T12:50:00",
                "error": None,
                "message": "Opportunity Map 任务已停止，可稍后继续。",
                "progress": 72,
                "completed_sections": 3,
                "total_sections": 5,
                "current_section_slug": "crowded_spaces",
                "current_section_label": "开始拥挤的方向",
                "scanned_core_papers": 620,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        response = self.client.get("/insights")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Opportunity Map 生成已暂停", html)
        self.assertIn("机会模块：3/5 个", html)
        self.assertIn("核心解读扫描：620/723 篇", html)

    def test_opportunity_status_marks_stale_running_job_as_stopped(self) -> None:
        self.write_opportunity_status(
            {
                "state": "running",
                "stage": "section_generation",
                "updated_at": "2026-04-12T13:04:00",
                "started_at": "2026-04-12T12:50:00",
                "error": None,
                "message": "正在生成机会模块：高回报低密度",
                "progress": 66,
                "completed_sections": 2,
                "total_sections": 5,
                "current_section_slug": "high_upside_edges",
                "current_section_label": "高回报低密度",
                "scanned_core_papers": 540,
                "total_core_papers": 723,
                "stop_requested": False,
            }
        )

        snapshot = self.app.opportunity_store.status_snapshot()

        self.assertEqual(snapshot["state"], "stopped")
        self.assertTrue(snapshot["can_continue"])
        self.assertFalse(snapshot["can_stop"])
        self.assertEqual(snapshot["scanned_core_papers"], 540)
        self.assertEqual(snapshot["total_core_papers"], 723)

    def test_sources_page_lists_archived_day_and_paper(self) -> None:
        self.create_source_day("2026-04-11", paper_id="2604.08377", title="SkillClaw")

        response = self.client.get("/sources")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("每日论文来源", html)
        self.assertIn("2026-04-11", html)
        self.assertIn("SkillClaw", html)
        self.assertIn("2604.08377", html)
        self.assertIn("打包下载选中论文", html)
        self.assertIn("导入到阅读器", html)

    def test_sources_download_zip_packages_selected_papers(self) -> None:
        self.create_source_day("2026-04-11", paper_id="2604.08377", title="SkillClaw")

        response = self.client.post(
            "/sources/download-zip",
            data={"run_date": "2026-04-11", "paper_ids": ["2604.08377"]},
        )
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())
        self.assertIn("manifest.json", names)
        self.assertTrue(any(name.endswith(".pdf") for name in names))

    def test_sources_import_submits_background_action(self) -> None:
        self.create_source_day("2026-04-11", paper_id="2604.08377", title="SkillClaw")

        fake_task = paper_reader_app_module.ActionRecord(
            id="source001",
            kind="source-import",
            title="来源归档导入：2026-04-11（1 篇）",
            source="source-import",
            requested_by_user_id=1,
            requested_by_display_name="admin",
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload={},
            created_at="2026-04-18T00:00:00",
            updated_at="2026-04-18T00:00:00",
            started_at=None,
            finished_at=None,
        )
        with patch.object(self.app.action_queue, "submit", return_value=fake_task) as mocked:
            response = self.client.post(
                "/sources/import",
                data={"run_date": "2026-04-11", "paper_ids": ["2604.08377"]},
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("后台导入任务已开始：来源归档导入：2026-04-11（1 篇）", html)
        mocked.assert_called_once()

    def test_generate_prompt_result_prefers_cached_arxiv_markdown(self) -> None:
        pdf_path = self.library / "arxiv" / "2501.12948v2.pdf"
        self.make_pdf(pdf_path, "Cached Markdown Paper")
        rel_path = "arxiv/2501.12948v2.pdf"
        scan = self.app.library.scan(force=True)
        self.app.team_store.sync_papers(scan.papers)
        self.app.team_store.add_source(
            rel_path,
            source_type="arxiv",
            source_value="2501.12948v2",
            source_url="https://arxiv.org/abs/2501.12948v2",
        )
        prompt = self.app.prompt_store.get_prompt(DEFAULT_PROMPT_SLUG)
        assert prompt is not None

        with patch("src.paper_reader.arxiv_markdown.fetch_public_url", return_value=self.fake_fetched_response(b"# arXiv markdown", content_type="text/markdown")):
            with patch.object(paper_reader_app_module, "run_prompt_on_document", return_value="解读完成") as mocked_run:
                with patch.object(self.app.library, "_maybe_refresh_ai_tags", return_value=None):
                    result_path, generated = self.app.library.generate_prompt_result(rel_path, prompt, force=True)

        cache_path = self.app.library.arxiv_markdown_path_for(rel_path)
        metadata_path = self.app.library.arxiv_markdown_metadata_path_for(rel_path)
        self.assertTrue(generated)
        self.assertTrue(result_path.exists())
        self.assertTrue(cache_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(cache_path.read_text(encoding="utf-8"), "# arXiv markdown\n")
        self.assertEqual(mocked_run.call_args.kwargs["source_markdown_path"], cache_path)

    def test_chat_queue_uses_cached_arxiv_markdown_for_answer(self) -> None:
        pdf_path = self.library / "chat" / "2501.12948.pdf"
        self.make_pdf(pdf_path, "Chat Markdown Paper")
        rel_path = "chat/2501.12948.pdf"
        scan = self.app.library.scan(force=True)
        self.app.team_store.sync_papers(scan.papers)
        self.app.team_store.add_source(
            rel_path,
            source_type="arxiv",
            source_value="2501.12948",
            source_url="https://arxiv.org/abs/2501.12948",
        )
        assistant_message_id = self.app.team_store.add_chat_message(
            rel_path,
            visibility="shared",
            body="Paper Bot 正在思考...",
            user_id=None,
            role="assistant",
            status="pending",
        )
        job = ChatJob(
            rel_path=rel_path,
            visibility="shared",
            user_id=1,
            display_name="admin",
            question="这篇论文在解决什么问题？",
            history=[],
            assistant_message_id=assistant_message_id,
            model="gpt-5.4",
        )

        with patch("src.paper_reader.arxiv_markdown.fetch_public_url", return_value=self.fake_fetched_response(b"# cached chat markdown", content_type="text/markdown")):
            with patch("src.paper_reader.chat_queue.answer_question_about_document", return_value="回答完成") as mocked_answer:
                self.app.chat_queue._process_job(job)

        cache_path = self.app.library.arxiv_markdown_path_for(rel_path)
        self.assertTrue(cache_path.exists())
        self.assertEqual(mocked_answer.call_args.kwargs["arxiv_markdown_path"], cache_path)
        context = self.app.team_store.chat_context(rel_path, current_user_id=1)
        shared_messages = context["shared"]["messages"]
        self.assertEqual(shared_messages[-1].body, "回答完成")
        self.assertEqual(shared_messages[-1].status, "completed")

    def test_upload_import_action_prefetches_arxiv_markdown_cache(self) -> None:
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        staging_dir = Path(self.app.config["UPLOAD_STAGING_ROOT"]) / "case-upload"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staging_dir / "2501.12948v2.pdf"
        self.make_pdf(staged_path, "Uploaded Markdown Paper")
        action = paper_reader_app_module.ActionRecord(
            id="uploadcase1",
            kind="upload-import",
            title="上传整理：2501.12948v2.pdf",
            source="upload",
            requested_by_user_id=admin_user.id,
            requested_by_display_name=admin_user.display_name,
            status="running",
            progress=1,
            message="任务开始执行。",
            error=None,
            result={},
            payload={
                "target_folder": "Uploads",
                "staged_files": [
                    {
                        "staged_rel_path": staged_path.relative_to(self.library).as_posix(),
                        "original_name": "2501.12948v2.pdf",
                    }
                ],
            },
            created_at="2026-04-20T00:00:00",
            updated_at="2026-04-20T00:00:00",
            started_at="2026-04-20T00:00:00",
            finished_at=None,
        )
        handler = self.app.action_queue._handlers["upload-import"]

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}):
            with patch("src.paper_reader.arxiv_markdown.fetch_public_url", return_value=self.fake_fetched_response(b"# prefetched upload markdown", content_type="text/markdown")):
                result = handler(action, lambda progress, message, payload=None: None, lambda: False)

        self.assertEqual(result["summary"]["saved"], 1)
        rel_path = "Uploads/2501.12948v2.pdf"
        cache_path = self.app.library.arxiv_markdown_path_for(rel_path)
        self.assertTrue(cache_path.exists())
        self.assertEqual(cache_path.read_text(encoding="utf-8"), "# prefetched upload markdown\n")

    def test_source_import_action_prefetches_arxiv_markdown_cache(self) -> None:
        self.create_source_day("2026-04-11", paper_id="2604.08377", title="SkillClaw")
        admin_user = self.app.team_store.get_user_by_username("admin")
        assert admin_user is not None
        action = paper_reader_app_module.ActionRecord(
            id="sourcecase1",
            kind="source-import",
            title="来源归档导入：2026-04-11（1 篇）",
            source="source-import",
            requested_by_user_id=admin_user.id,
            requested_by_display_name=admin_user.display_name,
            status="running",
            progress=1,
            message="任务开始执行。",
            error=None,
            result={},
            payload={"run_date": "2026-04-11", "paper_ids": ["2604.08377"]},
            created_at="2026-04-20T00:00:00",
            updated_at="2026-04-20T00:00:00",
            started_at="2026-04-20T00:00:00",
            finished_at=None,
        )
        handler = self.app.action_queue._handlers["source-import"]

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}):
            with patch("src.paper_reader.arxiv_markdown.fetch_public_url", return_value=self.fake_fetched_response(b"# prefetched source markdown", content_type="text/markdown")):
                result = handler(action, lambda progress, message, payload=None: None, lambda: False)

        self.assertEqual(result["summary"]["saved"], 1)
        rel_path = "Sources/HuggingFace/2026/04/11/2604.08377.pdf"
        cache_path = self.app.library.arxiv_markdown_path_for(rel_path)
        self.assertTrue(cache_path.exists())
        self.assertEqual(cache_path.read_text(encoding="utf-8"), "# prefetched source markdown\n")

    def test_index_shows_arxiv_markdown_cache_status(self) -> None:
        pdf_path = self.library / "status" / "2501.12948.pdf"
        self.make_pdf(pdf_path, "Status Markdown Paper")
        rel_path = "status/2501.12948.pdf"
        scan = self.app.library.scan(force=True)
        self.app.team_store.sync_papers(scan.papers)
        self.app.team_store.add_source(
            rel_path,
            source_type="arxiv",
            source_value="2501.12948",
            source_url="https://arxiv.org/abs/2501.12948",
        )
        with patch("src.paper_reader.arxiv_markdown.fetch_public_url", return_value=self.fake_fetched_response(b"# cached status markdown", content_type="text/markdown")):
            self.app.library.ensure_arxiv_markdown_for_rel_path(rel_path)

        response = self.client.get("/", query_string={"paper": rel_path, "tab": "source"})
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("arXiv Markdown", html)
        self.assertIn("已缓存", html)
        self.assertIn("2501.12948", html)
        self.assertIn("markdown.new", html)

    def test_build_chat_prompt_requests_markdown_math_instead_of_code_blocks(self) -> None:
        pdf_path = self.library / "formula" / "paper.pdf"
        self.make_pdf(pdf_path, "Formula Paper")

        prompt = build_chat_prompt(
            pdf_path,
            question="请推导文中的损失函数。",
            visibility="private",
            history=[],
            prompt_contexts=[],
        )

        self.assertIn("使用 Markdown 输出", prompt)
        self.assertIn("行内公式使用 `$...$`", prompt)
        self.assertIn("独立公式使用 `$$...$$`", prompt)
        self.assertIn("不要用 ``` 代码块包裹数学公式", prompt)
        self.assertIn("必须写出关键公式", prompt)

    def test_answer_question_about_document_uses_litellm_client(self) -> None:
        pdf_path = self.library / "formula" / "reasoning.pdf"
        self.make_pdf(pdf_path, "Reasoning Paper")

        class FakeClient:
            def complete(self, messages, *, model, session_id=None):
                self.messages = messages
                self.model = model
                self.session_id = session_id
                return "使用 $$E=mc^2$$ 作答"

        client = FakeClient()
        with patch.object(chat_ai_module, "get_llm_client", return_value=client):
            answer = answer_question_about_document(
                pdf_path,
                question="给我关键公式。",
                visibility="private",
                history=[],
                prompt_contexts=[],
                session_id="test-chat",
            )

        self.assertEqual(answer, "使用 $$E=mc^2$$ 作答")
        self.assertEqual(client.model, "glm-5.3")
        self.assertEqual(client.session_id, "test-chat")
        self.assertIn("论文正文", client.messages[0]["content"])

    def test_render_markdown_supports_rule_and_blockquote(self) -> None:
        rendered = render_markdown("# 标题\n\n> 引用内容\n\n---\n\n1. 第一项\n2. 第二项")

        self.assertIn("<h1>标题</h1>", rendered)
        self.assertIn("<blockquote><p>引用内容</p></blockquote>", rendered)
        self.assertIn("<hr>", rendered)
        self.assertIn("<ol><li>第一项</li><li>第二项</li></ol>", rendered)

    def test_render_markdown_preserves_math_blocks_and_strips_answer_wrapper(self) -> None:
        rendered = render_markdown("<answer>\n\n$$\nE = mc^2\n$$\n\n行内公式 $a^2+b^2=c^2$。\n\n</answer>")

        self.assertNotIn("&lt;answer&gt;", rendered)
        self.assertIn('<div class="math-block">$$\nE = mc^2\n$$</div>', rendered)
        self.assertIn("<p>行内公式 $a^2+b^2=c^2$。</p>", rendered)

    def test_render_markdown_keeps_html_like_inline_code_readable(self) -> None:
        rendered = render_markdown("答案放在 `<answer></answer>` 里。")

        self.assertIn("<p>答案放在 <code>&lt;answer&gt;&lt;/answer&gt;</code> 里。</p>", rendered)
        self.assertNotIn("&amp;lt;answer&amp;gt;", rendered)

    def test_render_markdown_supports_language_code_blocks_and_inline_math(self) -> None:
        rendered = render_markdown(
            "行内公式 $a_i=b_i+c_i$。\n\n```python\nprint('hi')\n```\n\n\\begin{aligned}\na&=b+c\n\\end{aligned}"
        )

        self.assertIn("<p>行内公式 $a_i=b_i+c_i$。</p>", rendered)
        self.assertIn('<div class="code-block"><div class="code-block-head"><span class="code-block-language">python</span></div><pre><code class="language-python">print(&#x27;hi&#x27;)</code></pre></div>', rendered)
        self.assertIn('<div class="math-block">\\begin{aligned}\na&amp;=b+c\n\\end{aligned}</div>', rendered)

    def test_render_markdown_supports_tables_images_and_task_lists(self) -> None:
        rendered = render_markdown(
            "| 指标 | 数值 |\n"
            "| :-- | --: |\n"
            "| F1 | 91.2 |\n\n"
            "- [x] 已完成\n"
            "- [ ] 待验证\n\n"
            "![架构图](https://example.com/arch.png)\n\n"
            "~~旧结论~~"
        )

        self.assertIn('<div class="table-wrapper"><table><thead><tr><th style="text-align:left">指标</th><th style="text-align:right">数值</th></tr></thead><tbody><tr><td style="text-align:left">F1</td><td style="text-align:right">91.2</td></tr></tbody></table></div>', rendered)
        self.assertIn('<label class="task-list-item"><input type="checkbox" disabled checked><span>已完成</span></label>', rendered)
        self.assertIn('<img src="https://example.com/arch.png" alt="架构图" loading="lazy">', rendered)
        self.assertIn("<p><del>旧结论</del></p>", rendered)

    def test_default_summary_prompt_mentions_markdown_math_and_code_format(self) -> None:
        self.assertIn("行内公式使用 `$...$`", DEFAULT_USER_PROMPT)
        self.assertIn("三反引号代码块，并显式标注语言", DEFAULT_USER_PROMPT)

    def test_chat_prompt_mentions_code_fences_and_math_format(self) -> None:
        prompt = build_chat_prompt(
            self.library / "paper.pdf",
            question="请给我关键公式和伪代码",
            visibility="shared",
            history=[],
            prompt_contexts=[],
        )

        self.assertIn("行内公式使用 `$...$`", prompt)
        self.assertIn("三反引号代码块，并标注语言", prompt)

    def test_prompt_store_upgrades_legacy_default_core_prompt(self) -> None:
        prompt = self.app.prompt_store.get_prompt(DEFAULT_PROMPT_SLUG)
        self.assertIsNotNone(prompt)
        assert prompt is not None

        import sqlite3

        with sqlite3.connect(self.app.team_store.db_path) as conn:
            conn.execute(
                "UPDATE prompt_versions SET user_prompt = ? WHERE id = ?",
                (LEGACY_DEFAULT_USER_PROMPT, prompt.version_id),
            )

        refreshed = PromptStore(self.library, self.app.team_store.db_path).get_prompt(DEFAULT_PROMPT_SLUG)
        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertGreaterEqual(refreshed.version, 2)
        self.assertIn("全文使用标准 Markdown 组织内容", refreshed.user_prompt)


if __name__ == "__main__":
    unittest.main()
