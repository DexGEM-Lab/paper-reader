from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
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

            def fetch_and_store(self, target, *, existing_rel_path=None, preserve_attachment_state=False):  # noqa: ANN001
                self.fetched.append(existing_rel_path)
                return SimpleNamespace(rel_path=existing_rel_path, fetch_status="ok", error=None)

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

    def test_reextract_processes_only_legacy_snapshots_and_preserves_attachments(self) -> None:
        records = [
            SimpleNamespace(rel_path="Web/legacy.md", resource_kind="project", source_url="https://example.com/legacy"),
            SimpleNamespace(rel_path="Web/current.md", resource_kind="article", source_url="https://example.com/current"),
        ]

        class WebStore:
            fetched: list[tuple[str, bool]] = []

            def metadata_for_rel_path(self, rel_path: str):  # noqa: ANN001
                version = 3 if rel_path.endswith("current.md") else 0
                return {"canonical_url": f"https://example.com/{rel_path.split('/')[-1][:-3]}", "resource_kind": "project", "reader_mode_version": version}

            def is_enriched(self, rel_path: str) -> bool:
                return True

            def fetch_and_store(self, target, *, existing_rel_path=None, preserve_attachment_state=False):  # noqa: ANN001
                self.fetched.append((existing_rel_path, preserve_attachment_state))
                return SimpleNamespace(rel_path=existing_rel_path, fetch_status="ok", error=None)

        class TeamStore:
            replaced: list[str] = []
            attachments = ["existing supplement"]
            sources = ["existing Feishu provenance"]

            def sync_papers(self, items):  # noqa: ANN001
                return None

            def replace_attachments(self, rel_path, attachments):  # noqa: ANN001
                self.replaced.append(rel_path)

        web_store = WebStore()
        team_store = TeamStore()
        app = SimpleNamespace(
            library=SimpleNamespace(
                scan=lambda include_done=True: SimpleNamespace(papers=records),
                web_resources=web_store,
                build_record_for_rel_path=lambda rel_path, prompts: SimpleNamespace(rel_path=rel_path),
            ),
            team_store=team_store,
        )

        stats = enrich_web_items(
            app,
            client=object(),
            model="cheap",
            limit=None,
            qps=10,
            reextract=True,
            sleep_fn=lambda seconds: None,
        )

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(web_store.fetched, [("Web/legacy.md", True)])
        self.assertEqual(team_store.replaced, [])
        self.assertEqual(team_store.attachments, ["existing supplement"])
        self.assertEqual(team_store.sources, ["existing Feishu provenance"])

    def test_failed_reextract_restores_snapshot_metadata_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            snapshot_path = root / "Web" / "legacy.md"
            metadata_path = snapshot_path.with_suffix(".json")
            asset_dir = snapshot_path.with_name("legacy.assets")
            snapshot_path.parent.mkdir(parents=True)
            asset_dir.mkdir()
            snapshot_path.write_text("original reader snapshot", encoding="utf-8")
            metadata_path.write_text(json.dumps({"canonical_url": "https://example.com/legacy", "resource_kind": "project"}), encoding="utf-8")
            (asset_dir / "original.png").write_bytes(b"original asset")

            class WebStore:
                library_root = root

                def metadata_for_rel_path(self, rel_path: str):  # noqa: ANN001
                    return json.loads(metadata_path.read_text(encoding="utf-8"))

                def metadata_path_for(self, path: Path) -> Path:
                    return path.with_suffix(".json")

                def _asset_dir_for(self, path: Path) -> Path:
                    return path.with_name(f"{path.stem}.assets")

                def fetch_and_store(self, target, *, existing_rel_path=None, preserve_attachment_state=False):  # noqa: ANN001
                    snapshot_path.write_text("failed replacement", encoding="utf-8")
                    metadata_path.write_text('{"fetch_status":"failed"}', encoding="utf-8")
                    for child in asset_dir.iterdir():
                        child.unlink()
                    (asset_dir / "partial.png").write_bytes(b"partial")
                    return SimpleNamespace(rel_path=existing_rel_path, fetch_status="failed", error="upstream timeout")

            record = SimpleNamespace(rel_path="Web/legacy.md", resource_kind="project", source_url="https://example.com/legacy")
            app = SimpleNamespace(
                library=SimpleNamespace(
                    scan=lambda include_done=True: SimpleNamespace(papers=[record]),
                    web_resources=WebStore(),
                    build_record_for_rel_path=lambda rel_path, prompts: SimpleNamespace(rel_path=rel_path),
                ),
                team_store=SimpleNamespace(),
            )
            stats = enrich_web_items(
                app,
                client=object(),
                model="cheap",
                limit=None,
                qps=10,
                reextract=True,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(stats["failed"], 1)
            self.assertEqual(snapshot_path.read_text(encoding="utf-8"), "original reader snapshot")
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["canonical_url"], "https://example.com/legacy")
            self.assertEqual([path.name for path in asset_dir.iterdir()], ["original.png"])
            self.assertEqual((asset_dir / "original.png").read_bytes(), b"original asset")


if __name__ == "__main__":
    unittest.main()
