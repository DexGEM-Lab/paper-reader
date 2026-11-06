from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
