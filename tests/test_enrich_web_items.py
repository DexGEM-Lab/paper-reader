from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.enrich_web_items import enrich_web_items


class BackfillResumeTests(unittest.TestCase):
    def test_completed_items_are_skipped_and_limit_counts_pending_only(self) -> None:
        records = [
            SimpleNamespace(rel_path="Web/done.md", resource_kind="article", source_url="https://example.com/done"),
            SimpleNamespace(rel_path="Web/pending.md", resource_kind="project", source_url="https://example.com/pending"),
            SimpleNamespace(rel_path="Web/later.md", resource_kind="news", source_url="https://example.com/later"),
        ]

        class WebStore:
            fetched: list[str] = []

            def is_enriched(self, rel_path: str) -> bool:
                return rel_path == "Web/done.md"

            def metadata_for_rel_path(self, rel_path: str):  # noqa: ANN001
                return {"canonical_url": f"https://example.com/{rel_path.split('/')[-1][:-3]}", "resource_kind": "project", "assets": []}

            def fetch_and_store(self, target, *, existing_rel_path=None):  # noqa: ANN001
                self.fetched.append(existing_rel_path)
                return SimpleNamespace(rel_path=existing_rel_path)

            def discover_attachments(self, rel_path: str, *, client, model):  # noqa: ANN001
                return []

        class TeamStore:
            synced: list[str] = []
            replaced: list[str] = []

            def sync_papers(self, items):  # noqa: ANN001
                self.synced.extend(item.rel_path for item in items)

            def replace_attachments(self, rel_path, attachments):  # noqa: ANN001
                self.replaced.append(rel_path)

        web_store = WebStore()
        team_store = TeamStore()
        library = SimpleNamespace(
            scan=lambda include_done=True: SimpleNamespace(papers=records),
            web_resources=web_store,
            build_record_for_rel_path=lambda rel_path, prompts: SimpleNamespace(rel_path=rel_path),
        )
        app = SimpleNamespace(library=library, team_store=team_store)

        stats = enrich_web_items(
            app,
            client=object(),
            model="cheap",
            limit=1,
            qps=10,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["processed"], 1)
        self.assertEqual(web_store.fetched, ["Web/pending.md"])
        self.assertEqual(team_store.replaced, ["Web/pending.md"])


if __name__ == "__main__":
    unittest.main()
