from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.import_links import main as import_links_main


class SeedScriptTests(unittest.TestCase):
    def test_dry_run_classifies_jsonl_without_creating_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "links.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "url": "https://github.com/dexgem/project",
                        "chat_name": "DexGEM",
                        "msg_time_iso": "2026-04-20T00:00:00Z",
                        "context_text": "project link",
                        "sender_open_id": "ou_test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = import_links_main([str(source), "--dry-run", "--library-root", str(root / "library")])
            self.assertEqual(result, 0)
            payloads = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(payloads[0]["resource_kind"], "project")
            self.assertEqual(payloads[0]["status"], "would-import")
            self.assertFalse((root / "library").exists())


if __name__ == "__main__":
    unittest.main()
