from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.paper_reader.agent_chat import (
    PiAgentChat,
    agent_session_key,
    extract_pi_chat_events,
    pi_assistant_is_final_turn_end,
)
from src.paper_reader.config import load_config


FIXTURE = Path(__file__).with_name("fixtures") / "pi_session.jsonl"


def fixture_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_agent_chat_config_defaults_and_overrides(tmp_path: Path) -> None:
    defaults = load_config(project_dir=tmp_path, cwd=tmp_path, environ={})
    assert defaults.agent_chat_enabled is False
    assert defaults.pi_bin == "pi"
    assert defaults.pi_home == Path("/opt/paper-reader/pi-home")
    assert defaults.pi_timeout == 300

    configured = load_config(
        project_dir=tmp_path,
        cwd=tmp_path,
        environ={
            "PAPER_READER_AGENT_CHAT_ENABLED": "true",
            "PAPER_READER_PI_BIN": "/opt/pi/bin/pi",
            "PAPER_READER_PI_HOME": "runtime/pi-home",
            "PAPER_READER_PI_TIMEOUT": "45",
        },
    )
    assert configured.agent_chat_enabled is True
    assert configured.pi_bin == "/opt/pi/bin/pi"
    assert configured.pi_home == (tmp_path / "runtime/pi-home").resolve()
    assert configured.pi_timeout == 45


def test_session_keying_private_and_shared() -> None:
    assert agent_session_key(visibility="private", user_id=7, paper_id=42) == "private:7:42"
    assert agent_session_key(visibility="shared", user_id=7, paper_id=42) == "shared:42"
    assert agent_session_key(visibility="shared", user_id=99, paper_id=42) == "shared:42"


def test_pi_log_events_distinguish_narration_tools_and_final() -> None:
    rows = fixture_rows()
    events = [event for row in rows for event in extract_pi_chat_events(row)]

    assert [(event.kind, event.text) for event in events] == [
        ("status", "正在使用工具: read"),
        ("narration", "我先核对论文中的实验表格。"),
        (
            "final",
            "核心结论是方法显著提升了成功率。\n\n来源定位：`materials/source/01-paper.pdf`，第 4 节。",
        ),
    ]
    assert events[-1].final is True
    assert pi_assistant_is_final_turn_end(rows[-1]) is True
    assert "RAW TOOL DUMP" not in "\n".join(event.text for event in events)


class FakeTeamStore:
    def __init__(self, root: Path, attachment: Path | None = None) -> None:
        self.root = root
        self.attachment = attachment

    def agent_paper_metadata(self, rel_path: str):
        return {
            "id": 42,
            "rel_path": rel_path,
            "title": "Agent Paper",
            "display_title": "Agent Paper: Evidence First",
            "canonical_url": "https://arxiv.org/abs/2601.00001",
            "sources": [{"source_url": "https://arxiv.org/abs/2601.00001"}],
        }

    def paper_context(self, rel_path: str):
        attachments = []
        if self.attachment is not None:
            attachments.append(
                SimpleNamespace(
                    title="Supplement",
                    local_path=self.attachment.relative_to(self.root).as_posix(),
                    source_url="https://example.org/supp.pdf",
                )
            )
        return {"attachments": attachments}


class FakeLibrary:
    def __init__(self, source: Path, result: Path | None = None) -> None:
        self.source = source
        self.result = result

    def resolve_relative_path(self, rel_path: str) -> Path:
        if rel_path == "paper.pdf":
            return self.source
        if self.result is not None and rel_path == "results/core.md":
            return self.result
        raise FileNotFoundError(rel_path)

    def ensure_arxiv_markdown_for_rel_path(self, rel_path: str):
        return None

    def prompt_result_info(self, rel_path: str, slug: str):
        return {"result_rel_path": "results/core.md"} if self.result is not None else {"result_rel_path": None}


class FakePromptStore:
    def list_prompts(self):
        return [SimpleNamespace(slug="core-zh", name="核心解读")]


class CompletedProcess:
    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


def test_material_package_and_agents_prompt_refresh(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    library_root.mkdir()
    source = library_root / "paper.pdf"
    source.write_bytes(b"pdf")
    attachment = library_root / "attachments" / "supp.pdf"
    attachment.parent.mkdir()
    attachment.write_bytes(b"supp")
    result = library_root / "results" / "core.md"
    result.parent.mkdir()
    result.write_text("# Core", encoding="utf-8")
    backend = PiAgentChat(
        data_dir=tmp_path / "data",
        pi_bin="pi",
        pi_home=tmp_path / "pi-home",
        timeout_seconds=5,
        library=FakeLibrary(source, result),
        prompt_store=FakePromptStore(),
        team_store=FakeTeamStore(library_root, attachment),
    )

    cwd = backend.prepare_material_package(
        "paper.pdf",
        session_key="shared:42",
        visibility="shared",
    )

    manifest = (cwd / "README.md").read_text(encoding="utf-8")
    agents = (cwd / "AGENTS.md").read_text(encoding="utf-8")
    assert "Agent Paper: Evidence First" in manifest
    assert "https://arxiv.org/abs/2601.00001" in manifest
    assert "论文原文件" in manifest
    assert "附件：Supplement" in manifest
    assert "已有解读：核心解读" in manifest
    assert len(list((cwd / "materials" / "source").iterdir())) == 1
    assert len(list((cwd / "materials" / "attachments").iterdir())) == 1
    assert len(list((cwd / "materials" / "interpretations").iterdir())) == 1
    assert "团队共享会话" in agents
    assert "用户使用中文时用中文回答" in agents
    assert "完整工具权限" in agents
    assert "来源定位" in agents


def test_mock_pi_subprocess_streams_fixture_and_uses_configured_interface(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    library_root.mkdir()
    source = library_root / "paper.pdf"
    source.write_bytes(b"pdf")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        sessions_dir = Path(command[command.index("--session-dir") + 1])
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "session-1.jsonl").write_bytes(FIXTURE.read_bytes())
        return CompletedProcess()

    backend = PiAgentChat(
        data_dir=tmp_path / "data",
        pi_bin="/mock/bin/pi",
        pi_home=tmp_path / "pi-home",
        timeout_seconds=5,
        library=FakeLibrary(source),
        prompt_store=FakePromptStore(),
        team_store=FakeTeamStore(library_root),
        popen_factory=fake_popen,
        poll_interval=0,
    )
    key = backend.session_key_for("paper.pdf", visibility="shared", user_id=7)
    assert backend.try_reserve(key) is True
    events = []
    answer = backend.run_turn(
        rel_path="paper.pdf",
        visibility="shared",
        user_id=7,
        display_name="Alice",
        question="请总结",
        session_key=key,
        on_event=events.append,
        reserved=True,
    )

    command, kwargs = calls[0]
    assert command[0] == "/mock/bin/pi"
    assert "--session-dir" in command
    assert command[-2:] == ["-p", "[团队成员：Alice]\n请总结"]
    assert kwargs["cwd"].endswith("/cwd")
    assert kwargs["env"]["PI_HOME"] == str((tmp_path / "pi-home").resolve())
    assert answer.startswith("核心结论")
    assert [event.kind for event in events] == ["status", "narration", "final"]
    assert backend.try_reserve(key) is True
    backend.release(key)
