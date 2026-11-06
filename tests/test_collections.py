from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.paper_reader.app import create_app
from src.paper_reader.team_store import TeamStore


class CollectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.store = TeamStore(self.root)
        self.alice = self.store.create_user("alice", "Alice", "password123", "member")
        self.bob = self.store.create_user("bob", "Bob", "password123", "member")
        self.admin = self.store.create_user("root", "Root", "password123", "admin")
        self.store.sync_papers([self.paper("a.pdf", "A"), self.paper("b.pdf", "B")])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def paper(rel_path: str, title: str, *, fetch_status: str = "ok") -> SimpleNamespace:
        return SimpleNamespace(
            rel_path=rel_path,
            file_name=Path(rel_path).name,
            folder="",
            extension=".pdf",
            title=title,
            display_title=title,
            preview_text=f"Preview for {title}",
            extracted_date="2026-01",
            sort_date="2026-01-01",
            date_precision="month",
            date_source="metadata",
            file_size=10,
            modified_at="2026-01-01T00:00:00",
            preview_kind="file",
            resource_kind="paper",
            source_url=None,
            source_domain=None,
            fetch_status=fetch_status,
            fetched_at=None,
            author=None,
            site_name=None,
            is_done=False,
        )

    def test_schema_is_idempotent_and_has_required_cascades(self) -> None:
        TeamStore(self.root)
        with sqlite3.connect(self.store.db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"collections", "collection_items", "collection_events"}.issubset(tables))
            foreign_keys = conn.execute("PRAGMA foreign_key_list(collection_items)").fetchall()
            self.assertTrue(any(row[2] == "collections" and row[6] == "CASCADE" for row in foreign_keys))
            self.assertTrue(any(row[2] == "papers" and row[6] == "CASCADE" for row in foreign_keys))

    def test_lazy_default_is_renamable_but_not_deletable(self) -> None:
        default_id = self.store.ensure_default_collection(self.alice.id)
        self.assertEqual(default_id, self.store.ensure_default_collection(self.alice.id))
        self.store.rename_collection(default_id, user_id=self.alice.id, name="待读")
        collection = self.store.collection_for_user(default_id, self.alice.id)
        self.assertEqual(collection.name, "待读")
        with self.assertRaisesRegex(ValueError, "不能删除"):
            self.store.delete_collection(default_id, user_id=self.alice.id)

    def test_name_collisions_get_stable_slug_suffixes(self) -> None:
        first = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="VLA")
        second = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="VLA")
        groups = self.store.list_collections_for_user(self.alice.id)
        slugs = {item.id: item.slug for item in groups["personal"]}
        self.assertEqual(slugs[first], "vla")
        self.assertEqual(slugs[second], "vla-2")
        bob_collection = self.store.create_collection(owner_type="user", user_id=self.bob.id, name="VLA")
        self.assertEqual(self.store.collection_for_user(bob_collection, self.bob.id).slug, "vla")

    def test_personal_is_private_and_team_membership_is_shared(self) -> None:
        personal = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="Private")
        team = self.store.create_collection(owner_type="team", user_id=self.alice.id, name="Shared")
        with self.assertRaises(PermissionError):
            self.store.set_collection_membership(
                collection_id=personal, rel_path="a.pdf", user_id=self.bob.id, present=True
            )
        with self.assertRaises(PermissionError):
            self.store.rename_collection(personal, user_id=self.admin.id, name="Admin peek", is_admin=True)
        self.assertTrue(
            self.store.set_collection_membership(
                collection_id=team, rel_path="a.pdf", user_id=self.bob.id, present=True
            )
        )
        with self.assertRaises(PermissionError):
            self.store.rename_collection(team, user_id=self.bob.id, name="Nope")
        self.store.rename_collection(team, user_id=self.admin.id, name="Admin rename", is_admin=True)
        self.assertEqual(self.store.collection_for_user(team, self.bob.id).name, "Admin rename")
        self.assertFalse(
            self.store.set_collection_membership(
                collection_id=team, rel_path="a.pdf", user_id=self.bob.id, present=False
            )
        )

    def test_membership_is_idempotent_and_delete_cascades(self) -> None:
        collection = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="Papers")
        for _ in range(2):
            self.assertTrue(
                self.store.set_collection_membership(
                    collection_id=collection, rel_path="a.pdf", user_id=self.alice.id, present=True
                )
            )
        self.assertEqual(len(self.store.collection_items_for_user(collection, user_id=self.alice.id)), 1)
        self.store.delete_paper("a.pdf")
        self.assertEqual(self.store.collection_items_for_user(collection, user_id=self.alice.id), [])

    def test_counts_and_lists_use_scanned_set_without_deleting_stale_rows(self) -> None:
        collection = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="Papers")
        for rel_path in ("a.pdf", "b.pdf"):
            self.store.set_collection_membership(
                collection_id=collection, rel_path=rel_path, user_id=self.alice.id, present=True
            )
        record = next(
            item
            for item in self.store.list_collections_for_user(
                self.alice.id, scanned_rel_paths={"a.pdf"}
            )["personal"]
            if item.id == collection
        )
        self.assertEqual(record.item_count, 1)
        self.assertEqual(record.unavailable_count, 1)
        self.assertEqual(
            [item["rel_path"] for item in self.store.collection_items_for_user(
                collection, user_id=self.alice.id, scanned_rel_paths={"a.pdf"}
            )],
            ["a.pdf"],
        )
        self.assertEqual(len(self.store.collection_items_for_user(collection, user_id=self.alice.id)), 2)

    def test_digest_persists_and_distinct_savers_feed_interest_rows(self) -> None:
        personal = self.store.create_collection(owner_type="user", user_id=self.alice.id, name="Mine")
        team = self.store.create_collection(owner_type="team", user_id=self.alice.id, name="Team")
        self.store.set_collection_membership(
            collection_id=personal, rel_path="a.pdf", user_id=self.alice.id, present=True
        )
        self.store.set_collection_membership(
            collection_id=team, rel_path="a.pdf", user_id=self.bob.id, present=True
        )
        self.store.save_collection_digest(
            personal, user_id=self.alice.id, markdown="# 摘要\n内容", item_count=1
        )
        persisted = self.store.collection_for_user(personal, self.alice.id)
        self.assertEqual(persisted.digest_markdown, "# 摘要\n内容")
        self.assertIsNotNone(persisted.digest_generated_at)
        row = next(item for item in self.store.library_interest_rows() if item["rel_path"] == "a.pdf")
        self.assertEqual(row["saver_count"], 2)


class CollectionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.env_patch = patch.dict(
            os.environ,
            {
                "PAPER_READER_ENV_FILE": str(self.root / ".no-env"),
                "PAPER_READER_SECRET_KEY": "collection-test-secret",
                "PAPER_READER_LOGIN_USERNAME": "admin",
                "PAPER_READER_LOGIN_PASSWORD": "test-admin-password",
            },
            clear=False,
        )
        self.env_patch.start()
        self.app = create_app(self.root, source_archive_root=self.root / "sources")
        self.app.testing = True
        self.app.config["CSRF_PROTECTION_ENABLED"] = False
        self.client = self.app.test_client()
        self.admin = self.app.team_store.get_user_by_username("admin")
        assert self.admin is not None
        self.login(self.admin)
        (self.root / "sample.md").write_text("# Sample paper\n\nA useful robotics method.", encoding="utf-8")
        self.client.get("/")

    def tearDown(self) -> None:
        self.app.action_queue.stop()
        self.app.job_queue.stop_all()
        self.env_patch.stop()
        self.tempdir.cleanup()

    def login(self, user) -> None:
        with self.client.session_transaction() as session:
            session["authenticated"] = True
            session["user_id"] = user.id
            session["username"] = user.username
            session["display_name"] = user.display_name
            session["role"] = user.role

    def test_json_create_and_idempotent_toggle(self) -> None:
        created = self.client.post(
            "/collections/create", json={"owner_type": "user", "name": "VLA"}
        )
        self.assertEqual(created.status_code, 201)
        collection_id = created.json["collection"]["id"]
        for _ in range(2):
            response = self.client.post(
                "/collections/toggle",
                json={"collection_id": collection_id, "rel_path": "sample.md", "present": True},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["present"])
        self.assertEqual(
            len(self.app.team_store.collection_items_for_user(collection_id, user_id=self.admin.id)),
            1,
        )

    def test_star_lazily_creates_default_and_collection_param_round_trips(self) -> None:
        response = self.client.post(
            "/collections/toggle",
            data={
                "rel_path": "sample.md",
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "tab": "source",
                "collection": "all",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("collection=all", response.location)
        groups = self.app.team_store.list_collections_for_user(self.admin.id)
        self.assertEqual(len(groups["personal"]), 1)
        self.assertTrue(groups["personal"][0].is_default)

    def test_team_creator_management_is_enforced_by_routes(self) -> None:
        team_id = self.app.team_store.create_collection(
            owner_type="team", user_id=self.admin.id, name="Shared"
        )
        member = self.app.team_store.create_user("member", "Member", "password123", "member")
        self.login(member)
        add = self.client.post(
            "/collections/toggle",
            json={"collection_id": team_id, "rel_path": "sample.md", "present": True},
        )
        self.assertEqual(add.status_code, 200)
        rename = self.client.post(f"/collections/{team_id}/rename", data={"name": "Hijack"})
        self.assertEqual(rename.status_code, 403)

    def test_digest_uses_default_model_and_persists_result(self) -> None:
        collection_id = self.app.team_store.create_collection(
            owner_type="user", user_id=self.admin.id, name="Digest me"
        )
        self.app.team_store.set_collection_membership(
            collection_id=collection_id,
            rel_path="sample.md",
            user_id=self.admin.id,
            present=True,
        )
        fake_client = SimpleNamespace(complete=lambda messages, **kwargs: "## 条目\n- Sample\n\n## 综合判断\n很好")
        with patch("src.paper_reader.app.get_llm_client", return_value=fake_client) as mocked:
            response = self.client.post(f"/collections/{collection_id}/digest/generate")
        self.assertEqual(response.status_code, 302)
        mocked.assert_called_once()
        saved = self.app.team_store.collection_for_user(collection_id, self.admin.id)
        self.assertIn("综合判断", saved.digest_markdown or "")
        page = self.client.get(f"/?collection={collection_id}")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn("collection-digest-panel", html)
        self.assertIn("综合判断", html)
        self.assertIn("重新生成", html)

    def test_sidebar_sheet_chips_flat_rows_and_undo_render(self) -> None:
        personal = self.app.team_store.create_collection(
            owner_type="user", user_id=self.admin.id, name="Reading"
        )
        team = self.app.team_store.create_collection(
            owner_type="team", user_id=self.admin.id, name="Lab picks"
        )
        for collection_id in (personal, team):
            self.app.team_store.set_collection_membership(
                collection_id=collection_id,
                rel_path="sample.md",
                user_id=self.admin.id,
                present=True,
            )
        response = self.client.get(f"/?collection={personal}&paper=sample.md")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-sidebar-switch="collections"', html)
        self.assertIn('data-sidebar-panel="collections"', html)
        self.assertIn("paper-row-managed", html)
        self.assertIn("collection-membership-sheet", html)
        self.assertIn("★ Reading", html)
        self.assertIn("★ Lab picks", html)
        self.assertIn(f'name="collection" value="{personal}"', html)

        removed = self.client.post(
            "/collections/toggle",
            data={
                "collection_id": personal,
                "collection": personal,
                "rel_path": "sample.md",
                "present": "0",
            },
            follow_redirects=True,
        )
        removed_html = removed.get_data(as_text=True)
        self.assertIn("撤销", removed_html)
        self.assertIn('name="present" value="1"', removed_html)

    def test_retrieval_event_requires_explicit_paper_selection(self) -> None:
        collection_id = self.app.team_store.create_collection(
            owner_type="user", user_id=self.admin.id, name="Metrics"
        )
        self.app.team_store.set_collection_membership(
            collection_id=collection_id,
            rel_path="sample.md",
            user_id=self.admin.id,
            present=True,
        )
        self.client.get(f"/?collection={collection_id}")
        with self.app.team_store._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM collection_events").fetchone()[0], 0)
        self.client.get(f"/?collection={collection_id}&paper=sample.md")
        with self.app.team_store._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM collection_events").fetchone()[0], 1)

    def test_csrf_protects_collection_mutations(self) -> None:
        self.app.config["CSRF_PROTECTION_ENABLED"] = True
        rejected = self.client.post(
            "/collections/create", json={"owner_type": "user", "name": "No token"}
        )
        self.assertEqual(rejected.status_code, 403)
        with self.client.session_transaction() as session:
            session["csrf_token"] = "collection-csrf"
        accepted = self.client.post(
            "/collections/create",
            json={"owner_type": "user", "name": "Token"},
            headers={"X-CSRFToken": "collection-csrf"},
        )
        self.assertEqual(accepted.status_code, 201)


if __name__ == "__main__":
    unittest.main()
