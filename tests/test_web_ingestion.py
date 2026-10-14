from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

from src.paper_reader.action_queue import ActionRecord
from src.paper_reader.ai_summary import extract_document_context
from src.paper_reader.config import load_config
from src.paper_reader.app import create_app
from src.paper_reader.markdown_render import render_markdown
from src.paper_reader.netutils import (
    DEFAULT_MAX_REDIRECTS,
    FetchedResponse,
    ResponseTooLargeError,
    UnsafeURLError,
    _SafeRedirectHandler,
    _read_bounded,
    fetch_public_url,
    quote_url_path,
    validate_public_url,
)
from src.paper_reader.web_resources import (
    MAX_ASSETS_PER_ITEM,
    DiscoveredAttachment,
    WebResourceStore,
    classify_import_target,
    extract_web_content,
    judge_attachment_candidates,
)


PUBLIC_DNS = [(2, 1, 6, "", ("93.184.216.34", 443))]
ARTICLE_HTML = b"""<!doctype html>
<html><head><title>Research systems in practice</title>
<meta name="author" content="Ada Researcher">
<meta property="og:site_name" content="Example Lab">
<meta property="article:published_time" content="2026-04-20">
</head><body><nav>Navigation</nav><article><h1>Research systems in practice</h1>
<p>This technical article explains a concrete research system, its design, evaluation protocol, and measured results.</p>
<p>The extracted snapshot must remain useful for reading, prompting, comments, and team discussion.</p>
<p>Related work: https://arxiv.org/abs/2501.12948.</p>
</article></body></html>"""


def fetched(payload: bytes, *, url: str = "https://example.com/research/system", content_type: str = "text/html") -> FetchedResponse:
    return FetchedResponse(
        requested_url=url,
        final_url=url,
        status=200,
        headers={"Content-Type": content_type, "Content-Length": str(len(payload))},
        body=payload,
    )


class NetutilsTests(unittest.TestCase):
    def test_private_and_reserved_addresses_are_rejected(self) -> None:
        private_resolver = lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 80))]
        with self.assertRaises(UnsafeURLError):
            validate_public_url("http://internal.example/resource", resolver=private_resolver)
        with self.assertRaises(UnsafeURLError):
            validate_public_url("http://169.254.169.254/latest/meta-data", resolver=lambda *args, **kwargs: [])

    def test_redirect_to_private_address_is_rejected_before_following(self) -> None:
        handler = _SafeRedirectHandler(resolver=lambda *args, **kwargs: PUBLIC_DNS, max_redirects=5)
        request = Request("https://example.com/start")
        with self.assertRaises(UnsafeURLError):
            handler.redirect_request(request, None, 302, "Found", {}, "http://127.0.0.1/admin")

    def test_response_size_cap_is_enforced_while_streaming(self) -> None:
        class Response:
            headers = {}

            def __init__(self) -> None:
                self.parts = [b"1234", b"5678", b""]

            def read(self, size: int) -> bytes:
                return self.parts.pop(0)

        with self.assertRaises(ResponseTooLargeError):
            _read_bounded(Response(), max_bytes=6)

    def test_fetch_limit_configuration_uses_separate_document_and_page_bounds(self) -> None:
        config = load_config(
            project_dir=Path("/tmp/paper-reader-config"),
            cwd=Path("/tmp/paper-reader-config"),
            environ={
                "PAPER_READER_ENV_FILE": "/tmp/paper-reader-config/missing.env",
                "PAPER_READER_FETCH_MAX_BYTES": str(120 * 1024 * 1024),
                "PAPER_READER_PAGE_FETCH_MAX_BYTES": str(20 * 1024 * 1024),
                "PAPER_READER_FETCH_MAX_REDIRECTS": "12",
            },
        )
        self.assertEqual(config.fetch_max_bytes, 120 * 1024 * 1024)
        self.assertEqual(config.fetch_page_max_bytes, 20 * 1024 * 1024)
        self.assertEqual(config.fetch_max_redirects, 12)
        self.assertTrue(config.screen_imports)

    def test_document_fetch_accepts_over_25mib_when_configured(self) -> None:
        class Response:
            status = 200
            headers = {"Content-Type": "application/pdf"}

            def __init__(self, size: int) -> None:
                self.remaining = size

            def __enter__(self):
                return self

            def __exit__(self, *args):  # noqa: ANN002, ANN003
                return None

            def geturl(self) -> str:
                return "https://example.com/paper.pdf"

            def read(self, size: int) -> bytes:
                if not self.remaining:
                    return b""
                chunk_size = min(size, self.remaining)
                self.remaining -= chunk_size
                return b"x" * chunk_size

        class Opener:
            def open(self, request, timeout):  # noqa: ANN001
                return Response(26 * 1024 * 1024)

        with patch("src.paper_reader.netutils.build_opener", return_value=Opener()):
            response = fetch_public_url(
                "https://example.com/paper.pdf",
                max_bytes=27 * 1024 * 1024,
                prefer_https=False,
                resolver=lambda *args, **kwargs: PUBLIC_DNS,
            )

        self.assertEqual(len(response.body), 26 * 1024 * 1024)

    def test_non_ascii_paths_are_quoted_idempotently_before_fetching(self) -> None:
        raw_url = "https://github.com/Rawmantic/Dataset3D高质量三维仿真数据集"
        encoded_url = "https://github.com/Rawmantic/Dataset3D%E9%AB%98%E8%B4%A8%E9%87%8F%E4%B8%89%E7%BB%B4%E4%BB%BF%E7%9C%9F%E6%95%B0%E6%8D%AE%E9%9B%86"
        self.assertEqual(quote_url_path(raw_url), encoded_url)
        self.assertEqual(quote_url_path(encoded_url), encoded_url)

        class Response:
            status = 200
            headers = {"Content-Type": "text/html"}

            def __enter__(self):
                return self

            def __exit__(self, *args):  # noqa: ANN002, ANN003
                return None

            def geturl(self) -> str:
                return encoded_url

            def read(self, size: int) -> bytes:
                return b""

        class Opener:
            request_url = ""

            def open(self, request, timeout):  # noqa: ANN001
                self.request_url = request.full_url
                return Response()

        opener = Opener()
        with patch("src.paper_reader.netutils.build_opener", return_value=opener):
            response = fetch_public_url(raw_url, prefer_https=False, resolver=lambda *args, **kwargs: PUBLIC_DNS)

        self.assertEqual(opener.request_url, encoded_url)
        self.assertEqual(response.final_url, encoded_url)

    def test_six_redirects_are_accepted_by_default_bound(self) -> None:
        self.assertEqual(DEFAULT_MAX_REDIRECTS, 10)
        handler = _SafeRedirectHandler(
            resolver=lambda *args, **kwargs: PUBLIC_DNS,
            max_redirects=DEFAULT_MAX_REDIRECTS,
        )
        request = Request("https://example.com/start")
        for hop in range(7):
            request = handler.redirect_request(request, None, 302, "Found", {}, f"/hop-{hop}")
            assert request is not None
        self.assertEqual(getattr(request, "_paper_reader_redirect_count"), 7)


class WebClassificationTests(unittest.TestCase):
    def test_arxiv_forms_share_one_canonical_source(self) -> None:
        targets = [
            classify_import_target("2501.12948"),
            classify_import_target("https://arxiv.org/abs/2501.12948v2"),
            classify_import_target("https://arxiv.org/html/2501.12948"),
            classify_import_target("https://arxiv.org/pdf/2501.12948.pdf"),
        ]
        self.assertEqual({target.source_value for target in targets}, {"2501.12948"})
        self.assertEqual({target.canonical_url for target in targets}, {"https://arxiv.org/abs/2501.12948"})
        self.assertTrue(all(target.route == "pdf" for target in targets))

    def test_openreview_landing_resolves_to_pdf(self) -> None:
        target = classify_import_target("https://openreview.net/forum?id=paper123")
        self.assertEqual(target.source_type, "openreview")
        self.assertEqual(target.source_value, "paper123")
        self.assertEqual(target.download_url, "https://openreview.net/pdf?id=paper123")

    def test_generic_links_are_classified_as_first_class_web_resources(self) -> None:
        article = classify_import_target("https://example.com/blog/post?utm_source=chat")
        project = classify_import_target("https://github.com/dexgem/research-project")
        news = classify_import_target("https://reuters.com/technology/research-news")
        self.assertEqual(article.resource_kind, "article")
        self.assertEqual(article.canonical_url, "https://example.com/blog/post")
        self.assertEqual(project.resource_kind, "project")
        self.assertEqual(news.resource_kind, "news")

    def test_canonical_url_folds_host_and_path_case_and_trailing_slashes(self) -> None:
        upper = classify_import_target("HTTPS://ROBODOJO-BENCHMARK.COM/LeaderBoard///")
        lower = classify_import_target("https://robodojo-benchmark.com/leaderboard")
        self.assertEqual(upper.canonical_url, "https://robodojo-benchmark.com/leaderboard")
        self.assertEqual(upper.source_value, lower.source_value)

    def test_trafilatura_extracts_markdown_and_metadata(self) -> None:
        markdown, metadata = extract_web_content(fetched(ARTICLE_HTML), resource_kind="article")
        self.assertIn("research system", markdown.casefold())
        self.assertEqual(metadata["title"], "Research systems in practice")
        self.assertIn("Ada Researcher", str(metadata["author"]))

    def test_extraction_sanitizes_html_titles_and_rejects_absurd_dates(self) -> None:
        html = b"""<html><head><title>&lt;em&gt;Robot &amp;amp; Agent&lt;/em&gt;</title>
        <meta property=\"article:published_time\" content=\"2046-06-01\"></head><body><article>
        <p>This is a substantive technical research article with enough extracted body text to be retained in a snapshot.</p>
        </article></body></html>"""
        markdown, metadata = extract_web_content(fetched(html), resource_kind="article")
        self.assertIn("substantive technical", markdown)
        self.assertEqual(metadata["title"], "Robot & Agent")
        with tempfile.TemporaryDirectory() as tempdir:
            store = WebResourceStore(Path(tempdir))
            with patch("src.paper_reader.web_resources.fetch_public_url", return_value=fetched(html)):
                snapshot = store.fetch_and_store(classify_import_target("https://example.com/future"))
            stored = store.metadata_for_rel_path(snapshot.rel_path)
            assert stored is not None
            self.assertIsNone(stored["published_date"])
            self.assertIsNone(stored["sort_date"])


class RichContentTests(unittest.TestCase):
    def test_media_assets_are_bounded_downloaded_and_rewritten_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            images = "".join(
                f'<img src="/media/figure-{index}.png" alt="Figure {index}" width="800" height="600">'
                for index in range(25)
            )
            page = (
                '<html><head><title>Visual research project</title>'
                '<meta property="og:image" content="/hero.jpg"></head>'
                '<body><article><h1>Visual research project</h1>'
                '<p>This project documents its research method, evaluation protocol, experiments, and results in detail.</p>'
                f'{images}</article></body></html>'
            ).encode()

            def fake_fetch(url: str, **kwargs):  # noqa: ANN003
                if url == "https://example.com/project":
                    return fetched(page, url=url)
                return fetched(b"image-bytes", url=url, content_type="image/png")

            store = WebResourceStore(root)
            with patch("src.paper_reader.web_resources.fetch_public_url", side_effect=fake_fetch):
                snapshot = store.fetch_and_store(classify_import_target("https://example.com/project"))

            metadata = store.metadata_for_rel_path(snapshot.rel_path)
            assert metadata is not None
            self.assertEqual(len(metadata["assets"]), MAX_ASSETS_PER_ITEM)
            markdown = (root / snapshot.rel_path).read_text(encoding="utf-8")
            self.assertIn("/files/Web/", markdown)
            self.assertNotIn("https://example.com/hero.jpg", markdown)
            asset_dir = (root / snapshot.rel_path).with_name(f"{Path(snapshot.rel_path).stem}.assets")
            self.assertEqual(len(list(asset_dir.glob("*"))), MAX_ASSETS_PER_ITEM)
            store.delete_snapshot(snapshot.rel_path)
            self.assertFalse((root / snapshot.rel_path).exists())
            self.assertFalse(asset_dir.exists())

    def test_attachment_judge_parsing_enforces_three_item_cap(self) -> None:
        candidates = [{"url": f"https://example.com/file-{index}.pdf", "text": f"Paper {index}"} for index in range(5)]

        class Client:
            def complete(self, messages, *, model, session_id=None):  # noqa: ANN001
                return json.dumps([
                    {"id": index, "kind": "paper-pdf", "title": f"Selected {index}"}
                    for index in range(5)
                ])

        selected = judge_attachment_candidates(candidates, page_title="Project", client=Client(), model="cheap")
        self.assertEqual(len(selected), 3)
        self.assertEqual([item.title for item in selected], ["Selected 0", "Selected 1", "Selected 2"])

    def test_attachment_discovery_downloads_at_most_three_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            links = "".join(f'<a href="/paper-{index}.pdf">Paper PDF {index}</a>' for index in range(5))
            page = (
                '<html><head><title>Attachment project</title></head><body><article>'
                '<p>This research project explains the method, experiments, evaluation, and concrete results.</p>'
                f'{links}</article></body></html>'
            ).encode()
            store = WebResourceStore(root)
            with patch("src.paper_reader.web_resources.fetch_public_url", return_value=fetched(page, url="https://example.com/project")):
                snapshot = store.fetch_and_store(classify_import_target("https://example.com/project"))

            class Client:
                def complete(self, messages, *, model, session_id=None):  # noqa: ANN001
                    return json.dumps([
                        {"id": index, "kind": "paper-pdf", "title": f"Paper {index}"}
                        for index in range(5)
                    ])

            def fetch_pdf(url: str, **kwargs):  # noqa: ANN003
                return fetched(b"%PDF-1.7 test", url=url, content_type="application/pdf")

            with patch("src.paper_reader.web_resources.fetch_public_url", side_effect=fetch_pdf):
                attachments = store.discover_attachments(snapshot.rel_path, client=Client(), model="cheap")

            self.assertEqual(len(attachments), 3)
            self.assertTrue(all(item.local_path and (root / item.local_path).exists() for item in attachments))
            self.assertTrue(store.is_enriched(snapshot.rel_path))


class WebImportIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.source_tempdir = tempfile.TemporaryDirectory()
        self.library = Path(self.tempdir.name)
        self.env_patch = patch.dict(
            os.environ,
            {
                "PAPER_READER_ENV_FILE": str(self.library / ".missing-env"),
                "PAPER_READER_SECRET_KEY": "test-secret",
                "PAPER_READER_LOGIN_USERNAME": "admin",
                "PAPER_READER_LOGIN_PASSWORD": "test-password",
                "PAPER_READER_SCREEN_IMPORTS": "false",
            },
            clear=False,
        )
        self.env_patch.start()
        self.app = create_app(self.library, source_archive_root=Path(self.source_tempdir.name))
        self.app.testing = True
        self.client = self.app.test_client()
        self.admin = self.app.team_store.get_user_by_username("admin")
        assert self.admin is not None
        with self.client.session_transaction() as session:
            session.update(
                authenticated=True,
                user_id=self.admin.id,
                username=self.admin.username,
                display_name=self.admin.display_name,
                role=self.admin.role,
            )

    def tearDown(self) -> None:
        self.app.action_queue.stop()
        self.app.job_queue.stop_all()
        self.env_patch.stop()
        self.source_tempdir.cleanup()
        self.tempdir.cleanup()

    def action(self, url: str, *, provenance=None) -> ActionRecord:  # noqa: ANN001
        return ActionRecord(
            id="webimport001",
            kind="remote-import",
            title=f"Import {url}",
            source="test",
            requested_by_user_id=self.admin.id,
            requested_by_display_name=self.admin.display_name,
            status="running",
            progress=1,
            message="running",
            error=None,
            result={},
            payload={"raw_targets": url, "recommendation_reason": "", "provenance": provenance or []},
            created_at="2026-04-20T00:00:00",
            updated_at="2026-04-20T00:00:00",
            started_at="2026-04-20T00:00:00",
            finished_at=None,
        )

    def run_import(self, url: str, *, response=None, error=None, provenance=None):  # noqa: ANN001
        handler = self.app.action_queue._handlers["remote-import"]
        effect = error if error is not None else (response or fetched(ARTICLE_HTML, url=url))
        with patch("src.paper_reader.web_resources.fetch_public_url", side_effect=effect if isinstance(effect, Exception) else None, return_value=None if isinstance(effect, Exception) else effect):
            with patch.object(
                self.app.job_queue,
                "submit",
                return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []},
            ):
                return handler(self.action(url, provenance=provenance), lambda *args, **kwargs: None, lambda: False)

    def test_web_import_creates_snapshot_team_record_and_reader_view(self) -> None:
        url = "https://example.com/research/system?utm_source=feishu"
        provenance = [
            {
                "url": url,
                "chat_name": "DexGEM Research",
                "msg_time_iso": "2026-04-20T01:02:03Z",
                "context_text": "Useful systems paper",
                "sender_open_id": "ou_test",
            }
        ]
        result = self.run_import(url, provenance=provenance)
        item = result["items"][0]
        rel_path = item["rel_path"]
        self.assertEqual(item["status"], "saved")
        self.assertTrue((self.library / rel_path).exists())
        self.assertTrue((self.library / rel_path).with_suffix(".json").exists())
        record = self.app.library.build_record_for_rel_path(rel_path, [])
        self.assertEqual(record.resource_kind, "article")
        self.assertEqual(record.fetch_status, "ok")
        self.assertEqual(record.source_domain, "example.com")
        title, prompt_text, _ = extract_document_context(self.library / rel_path)
        self.assertEqual(title, "Research systems in practice")
        self.assertIn("evaluation protocol", prompt_text)
        self.assertIsNone(self.app.library.resolve_arxiv_id_for(rel_path))
        sources = self.app.team_store.sources_for_rel_path(rel_path)
        self.assertTrue(any(source["source_type"] == "canonical_url" for source in sources))
        feishu = next(source for source in sources if source["source_type"] == "feishu")
        self.assertEqual(feishu["source_value"], "DexGEM Research")
        self.assertEqual(feishu["source_context"], "Useful systems paper")

        html = self.client.get(f"/?paper={rel_path}&tab=source").get_data(as_text=True)
        self.assertIn("ARTICLE", html)
        self.assertIn("example.com", html)
        self.assertIn("Open original", html)
        self.assertIn("Research systems in practice", html)
        self.assertIn("Useful systems paper", html)

    def test_reader_lists_locally_persisted_attachments(self) -> None:
        result = self.run_import("https://example.com/research/attachments")
        rel_path = result["items"][0]["rel_path"]
        attachment_path = self.library / "Web" / "example-com" / "attachments" / "supplement.pdf"
        attachment_path.parent.mkdir(parents=True, exist_ok=True)
        attachment_path.write_bytes(b"%PDF-1.7 test")
        attachment_rel_path = attachment_path.relative_to(self.library).as_posix()
        self.app.team_store.replace_attachments(
            rel_path,
            [
                DiscoveredAttachment(
                    kind="supplementary",
                    title="Supplementary experiments",
                    source_url="https://example.com/supplement.pdf",
                    local_path=attachment_rel_path,
                )
            ],
        )

        html = self.client.get(f"/?paper={rel_path}&tab=source").get_data(as_text=True)
        self.assertIn("附件", html)
        self.assertIn("Supplementary experiments", html)
        self.assertIn(f"/files/{attachment_rel_path}", html)

    def test_reimport_refreshes_same_canonical_item_instead_of_duplicating(self) -> None:
        url = "https://example.com/research/system"
        first = self.run_import(url)
        first_path = first["items"][0]["rel_path"]
        updated_html = ARTICLE_HTML.replace(b"Research systems in practice", b"Updated research system")
        second = self.run_import(url, response=fetched(updated_html, url=url))
        self.assertEqual(second["items"][0]["status"], "existing")
        self.assertEqual(second["items"][0]["rel_path"], first_path)
        self.assertEqual(len(list((self.library / "Web").rglob("*.md"))), 1)
        self.assertIn("Updated research system", (self.library / first_path).read_text(encoding="utf-8"))

    def test_failed_web_extraction_is_dropped_even_when_screening_is_disabled(self) -> None:
        url = "https://example.com/paywalled"
        result = self.run_import(url, error=RuntimeError("HTTP 403 for source"))
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("网页正文提取失败", item["message"])
        self.assertIsNone(self.app.team_store.find_paper_by_source("canonical_url", url))
        self.assertEqual(list((self.library / "Web").rglob("*.md")), [])

    def test_notion_style_javascript_shell_is_dropped_before_llm_screening(self) -> None:
        url = "https://excessive-share-4ad.notion.site/"
        shell = b"""<html><head><title>Notion | Where teams and agents work together</title></head>
        <body><main><p>You need to enable JavaScript to run this app. Notion is where teams and agents
        work together to plan, build, and share knowledge across the organization.</p></main></body></html>"""
        result = self.run_import(url, response=fetched(shell, url=url))
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("JavaScript 的壳页面", item["message"])
        self.assertIsNone(self.app.team_store.find_paper_by_source("canonical_url", url))
        self.assertEqual(list((self.library / "Web").rglob("*.md")), [])

    def test_private_url_import_is_blocked_without_network_access(self) -> None:
        handler = self.app.action_queue._handlers["remote-import"]
        result = handler(
            self.action("http://127.0.0.1/research"),
            lambda *args, **kwargs: None,
            lambda: False,
        )
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("网页正文提取失败", item["message"])
        self.assertIsNone(self.app.team_store.find_paper_by_source("canonical_url", "http://127.0.0.1/research"))
        self.assertEqual(list((self.library / "Web").rglob("*.md")), [])

    def test_live_import_screening_allows_and_flags_borderline_web_item(self) -> None:
        url = "https://example.com/research/borderline"
        self.app.config["SCREEN_IMPORTS"] = True

        class Client:
            def complete(self, messages, *, model, session_id=None):  # noqa: ANN001
                return json.dumps([{
                    "url": url,
                    "decision": "borderline",
                    "reason": "Research-adjacent project with limited evaluation detail.",
                    "corrected_kind": "project",
                }])

        with patch("src.paper_reader.app.get_llm_client", return_value=Client()):
            result = self.run_import(url)
        item = result["items"][0]
        self.assertEqual(item["status"], "saved")
        metadata = self.app.library.web_resources.metadata_for_rel_path(item["rel_path"])
        assert metadata is not None
        self.assertEqual(metadata["screening"]["decision"], "borderline")
        html = self.client.get(f"/?paper={item['rel_path']}&tab=source").get_data(as_text=True)
        self.assertIn("待复核", html)

    def test_live_import_screening_drops_without_creating_item(self) -> None:
        url = "https://example.com/pricing"
        self.app.config["SCREEN_IMPORTS"] = True

        class Client:
            def complete(self, messages, *, model, session_id=None):  # noqa: ANN001
                return json.dumps([{
                    "url": url,
                    "decision": "drop",
                    "reason": "Marketing and pricing page without research content.",
                    "corrected_kind": None,
                }])

        with patch("src.paper_reader.app.get_llm_client", return_value=Client()):
            result = self.run_import(url)
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("链接未通过内容筛选", item["message"])
        self.assertIsNone(self.app.team_store.find_paper_by_source("canonical_url", url))
        self.assertEqual(list((self.library / "Web").rglob("*.md")), [])

    def test_upload_limit_is_configured_and_markdown_links_are_scheme_safe(self) -> None:
        self.assertEqual(self.app.config["MAX_CONTENT_LENGTH"], 100 * 1024 * 1024)
        rendered = render_markdown("[bad](javascript:alert(1)) [good](https://example.com)")
        self.assertNotIn("javascript:", rendered)
        self.assertIn('href="https://example.com"', rendered)


if __name__ == "__main__":
    unittest.main()
