"""Real-wheel regression for source documents and their trailing attachments."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from werkzeug.serving import make_server

from src.paper_reader.app import create_app
from src.paper_reader.web_resources import DiscoveredAttachment


@unittest.skipUnless(shutil.which("agent-browser"), "agent-browser is not installed")
class AttachmentScrollBrowserTests(unittest.TestCase):
    def test_last_attachment_is_reachable_after_markdown_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "PAPER_READER_ENV_FILE": f"{tmp}/missing.env",
            "PAPER_READER_DATA_DIR": f"{tmp}/data",
            "PAPER_READER_AUTH_MODE": "local",
            "PAPER_READER_LOGIN_USERNAME": "admin",
            "PAPER_READER_LOGIN_PASSWORD": "test-password",
            "PAPER_READER_SECRET_KEY": "test-secret",
            "PAPER_READER_DISCOVERY_ENABLED": "false",
        }):
            library = Path(tmp) / "library"
            library.mkdir()
            (library / "fixture.md").write_text(
                "# Attachment scroll fixture\n\n" + "A long source paragraph.\n\n" * 150,
                encoding="utf-8",
            )
            pdf = PdfWriter()
            pdf.add_blank_page(width=72, height=72)
            pdf.add_metadata({"/Title": "PDF attachment scroll fixture"})
            with (library / "fixture.pdf").open("wb") as stream:
                pdf.write(stream)
            app = create_app(library, source_archive_root=Path(tmp) / "sources")
            app.team_store.sync_papers(app.library.scan().papers)
            for source in ("fixture.md", "fixture.pdf"):
                app.team_store.replace_attachments(source, [
                    DiscoveredAttachment(
                        kind="supplementary",
                        title=f"Supplement {n:02d}",
                        source_url=f"https://example.com/supplement-{n}.pdf",
                        local_path="",
                    ) for n in range(3)
                ])
            server = make_server("127.0.0.1", 0, app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            browser = ["agent-browser", "--session", f"paper-reader-attachments-{uuid.uuid4().hex}"]

            def run(*args: str) -> str:
                return subprocess.run(browser + list(args), check=True, text=True,
                                      capture_output=True, timeout=30).stdout

            try:
                base = f"http://127.0.0.1:{server.server_port}"
                run("open", f"{base}/login")
                run("fill", 'input[name="username"]', "admin")
                run("fill", 'input[name="password"]', "test-password")
                run("click", 'button[type="submit"]')
                for source in ("fixture.md", "fixture.pdf"):
                    for width, height in ((1442, 800), (1274, 627), (823, 463), (390, 844)):
                        with self.subTest(source=source, viewport=(width, height)):
                            run("set", "viewport", str(width), str(height))
                            run("open", f"{base}/?paper={source}&tab=source")
                            # A PDF keeps its own document scroller and fills the source
                            # viewport; the stage must still scroll past it to attachments.
                            if source.endswith(".pdf"):
                                sizing = json.loads(run("eval", """(() => {
                                    const stage = document.querySelector('.reader-stage');
                                    const pdf = document.querySelector('.main-viewer');
                                    return {stage: stage.clientHeight, pdf: pdf.clientHeight};
                                })()""", "--json"))["data"]["result"]
                                self.assertGreater(sizing["pdf"], sizing["stage"] - 40)
                            # Wheel over the source gutter (outside the native PDF
                            # frame) so this tests actual input, not scrollTop assignment.
                            point = json.loads(run("eval", """(() => {
                                const box = document.querySelector('.reader-stage').getBoundingClientRect();
                                return {x: box.left + 3, y: box.top + box.height / 2};
                            })()""", "--json"))["data"]["result"]
                            run("mouse", "move", str(round(point["x"])), str(round(point["y"])))
                            run("mouse", "wheel", "100000")
                            run("wait", "200")
                            result = json.loads(run("eval", """(() => {
                                const stage = document.querySelector('.reader-stage');
                                const attachments = document.querySelector('.paper-attachments');
                                const last = attachments.querySelector('li:last-child a');
                                const box = last.getBoundingClientRect();
                                const bounds = stage.getBoundingClientRect();
                                const source = document.querySelector('.source-viewer');
                                return {
                                    lastVisible: box.top >= bounds.top && box.bottom <= bounds.bottom
                                        && box.bottom <= innerHeight,
                                    lastText: last.textContent,
                                    lastHit: last.contains(document.elementFromPoint(
                                        box.left + box.width / 2, box.top + box.height / 2)),
                                    scrolled: stage.scrollTop,
                                    sourceHasNestedScroll: source.scrollHeight > source.clientHeight + 1,
                                };
                            })()""", "--json"))["data"]["result"]
                            self.assertEqual(result["lastText"], "Supplement 02")
                            self.assertTrue(result["lastVisible"], result)
                            self.assertTrue(result["lastHit"], result)
                            self.assertGreater(result["scrolled"], 0)
                            self.assertFalse(result["sourceHasNestedScroll"], result)
            finally:
                subprocess.run(browser + ["close"], capture_output=True, timeout=30)
                server.shutdown()
                thread.join(timeout=5)
                app.action_queue.stop()
                app.job_queue.stop_all()
