"""Browser regression for the mobile library's independently scrollable pane.

Requires the optional agent-browser CLI and its Chromium installation.
"""
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


@unittest.skipUnless(shutil.which("agent-browser"), "agent-browser is not installed")
class MobileLibraryBrowserTests(unittest.TestCase):
    def test_library_rows_and_filters_have_room_at_mobile_and_desktop(self) -> None:
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
            pdf = PdfWriter()
            pdf.add_blank_page(width=72, height=72)
            pdf.add_metadata({"/Title": "Mobile library fixture"})
            with (library / "fixture.pdf").open("wb") as stream:
                pdf.write(stream)
            app = create_app(library, source_archive_root=Path(tmp) / "sources")
            server = make_server("127.0.0.1", 0, app, threaded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            browser = ["agent-browser", "--session", f"paper-reader-test-{uuid.uuid4().hex}"]

            def run(*args: str) -> str:
                return subprocess.run(browser + list(args), check=True, text=True, capture_output=True, timeout=30).stdout

            try:
                run("set", "viewport", "390", "844")
                run("open", f"http://127.0.0.1:{server.server_port}/login")
                run("fill", 'input[name="username"]', "admin")
                run("fill", 'input[name="password"]', "test-password")
                run("click", 'button[type="submit"]')
                run("click", '[data-mobile-tab="library"]')
                for width, height in ((390, 844), (1920, 1080)):
                    run("set", "viewport", str(width), str(height))
                    result = json.loads(run("eval", """(() => {
                        document.querySelectorAll('#paper-list details').forEach(el => el.open = true);
                        const pane = document.querySelector('.paper-tree-scroll');
                        const row = document.querySelector('.paper-row');
                        const filter = document.querySelector('.library-filter-row');
                        const scroll = pane.getBoundingClientRect();
                        const paper = row.getBoundingClientRect();
                        return {scrollHeight: scroll.height, rowHeight: paper.height,
                          rowVisible: paper.top < scroll.bottom && paper.bottom > scroll.top && paper.top < innerHeight,
                          filterHeight: filter.getBoundingClientRect().height};
                    })()""", "--json"))["data"]["result"]
                    with self.subTest(viewport=(width, height)):
                        self.assertGreater(result["scrollHeight"], 100)
                        self.assertGreater(result["rowHeight"], 20)
                        self.assertTrue(result["rowVisible"])
                        self.assertGreater(result["filterHeight"], 20)
            finally:
                subprocess.run(browser + ["close"], capture_output=True, timeout=30)
                server.shutdown()
                thread.join(timeout=5)
                app.action_queue.stop()
                app.job_queue.stop_all()
