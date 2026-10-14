from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.paper_reader.app import PaperRecord, create_app
from src.paper_reader.config import load_config
from src.paper_reader.screening import candidate_packet, delete_library_item, screen_packets


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    def complete(self, messages, *, model: str, session_id=None):  # noqa: ANN001, ANN201
        self.calls.append((messages, model))
        return self.response


class ScreeningDecisionTests(unittest.TestCase):
    def test_batch_screening_returns_keep_and_drop(self) -> None:
        client = FakeLLM(
            '[{"url":"https://github.com/lab/project","decision":"keep","reason":"research repository",'
            '"corrected_kind":"project"},'
            '{"url":"https://pan.baidu.com/s/abc","decision":"drop","reason":"netdisk share extraction",'
            '"corrected_kind":null}]'
        )
        packets = [
            candidate_packet(url="https://github.com/lab/project", context_text="release code for a robotics paper"),
            candidate_packet(url="https://pan.baidu.com/s/abc", context_text="提取码在这里"),
        ]

        results = screen_packets(packets, client, model="deepseek-flash")

        self.assertEqual([result.decision for result in results], ["keep", "drop"])
        self.assertEqual(results[0].corrected_kind, "project")
        self.assertTrue(all(result.actionable for result in results))
        self.assertEqual(len(client.calls), 1)
        self.assertIn("netdisk", client.calls[0][0][0]["content"])

    def test_malformed_llm_output_is_error_not_drop(self) -> None:
        packets = [candidate_packet(url="https://example.com/useful", context_text="research article")]
        results = screen_packets(packets, FakeLLM("I would keep it."))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decision, "error")
        self.assertFalse(results[0].actionable)
        self.assertIn("JSON", results[0].error or "")


class ScreeningDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        config = load_config(
            project_dir=self.root,
            cwd=self.root,
            environ={
                "PAPER_READER_ENV_FILE": str(self.root / "missing.env"),
                "PAPER_READER_DATA_DIR": str(self.root / "data"),
                "PAPER_READER_LIBRARY_DIR": str(self.root / "library"),
                "PAPER_READER_LOGIN_USERNAME": "admin",
                "PAPER_READER_LOGIN_PASSWORD": "pass-12345",
            },
        )
        self.app = create_app(config.library_dir, config=config)
        self.rel_path = "Web/rejected.md"
        path = config.library_dir / self.rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Referral page\n", encoding="utf-8")
        self.app.team_store.sync_papers(
            [
                PaperRecord(
                    rel_path=self.rel_path,
                    file_name="rejected.md",
                    folder="Web",
                    extension=".md",
                    title="Referral page",
                    display_title="Referral page",
                    preview_text="Referral page",
                    extracted_date=None,
                    date_precision=None,
                    date_source=None,
                    sort_date=None,
                    file_size=15,
                    modified_at="2026-01-01T00:00:00",
                    preview_kind="web",
                    prompt_result_count=0,
                    prompt_result_slugs=[],
                    is_done=False,
                    resource_kind="article",
                    source_url="https://example.com/referral",
                    source_domain="example.com",
                    fetch_status="ok",
                )
            ]
        )
        self.user = self.app.team_store.get_user_by_username("admin")
        assert self.user is not None

    def tearDown(self) -> None:
        self.app.action_queue.stop()
        self.app.job_queue.stop_all()
        self.temp_dir.cleanup()

    def test_deletion_cascades_all_paper_dependents(self) -> None:
        with sqlite3.connect(self.app.team_store.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            paper_id = conn.execute("SELECT id FROM papers WHERE rel_path = ?", (self.rel_path,)).fetchone()[0]
            now = "2026-01-01T00:00:00"
            conn.execute("INSERT INTO paper_sources(paper_id, source_type, source_value, created_at) VALUES (?, 'feishu', 'Research', ?)", (paper_id, now))
            conn.execute("INSERT INTO paper_recommendations(paper_id, user_id, reason, created_at, updated_at) VALUES (?, ?, '', ?, ?)", (paper_id, self.user.id, now, now))
            conn.execute("INSERT INTO paper_user_states(paper_id, user_id, is_done, updated_at) VALUES (?, ?, 0, ?)", (paper_id, self.user.id, now))
            conn.execute("INSERT INTO paper_likes(paper_id, user_id, created_at) VALUES (?, ?, ?)", (paper_id, self.user.id, now))
            conn.execute("INSERT INTO comments(paper_id, user_id, body, created_at, updated_at) VALUES (?, ?, 'junk', ?, ?)", (paper_id, self.user.id, now, now))
            tag_id = conn.execute("INSERT INTO tags(name, slug, created_at, updated_at) VALUES ('junk', 'junk', ?, ?)", (now, now)).lastrowid
            conn.execute("INSERT INTO paper_tags(paper_id, tag_id, source_type, created_at) VALUES (?, ?, 'ai', ?)", (paper_id, tag_id, now))
            conn.execute("INSERT INTO prompt_runs(paper_id, prompt_slug, prompt_name, prompt_version, model, status, shared, updated_at) VALUES (?, 'core', 'Core', 1, 'deepseek-flash', 'completed', 1, ?)", (paper_id, now))
            thread_id = conn.execute("INSERT INTO chat_threads(paper_id, visibility, thread_key, created_at, updated_at) VALUES (?, 'shared', 'thread-rejected', ?, ?)", (paper_id, now, now)).lastrowid
            conn.execute("INSERT INTO chat_messages(thread_id, role, body, created_at, updated_at) VALUES (?, 'user', 'why', ?, ?)", (thread_id, now, now))
            conn.execute("INSERT INTO paper_date_overrides(paper_id, display_date, precision, sort_date, updated_at) VALUES (?, '2026', 'year', '2026-01-01', ?)", (paper_id, now))

        delete_library_item(self.app, self.rel_path)

        self.assertFalse((self.app.config["LIBRARY_ROOT"] / self.rel_path).exists())
        with sqlite3.connect(self.app.team_store.db_path) as conn:
            for table in (
                "papers",
                "paper_sources",
                "paper_recommendations",
                "paper_user_states",
                "paper_likes",
                "comments",
                "paper_tags",
                "prompt_runs",
                "chat_threads",
                "chat_messages",
                "paper_date_overrides",
            ):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)


if __name__ == "__main__":
    unittest.main()
