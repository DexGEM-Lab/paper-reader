"""Opt-in real Docker/Pi tests. See ops/agent-sandbox/README.md.

Requires the built image, root-owned network.sh already applied, and Docker
access. Does not mount production files, credentials or the repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from src.paper_reader.agent_chat import PiAgentChat
from src.paper_reader.agent_sandbox import SharedAgentSandbox

IMAGE = os.environ.get("PAPER_READER_TEST_SANDBOX_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="opt-in real shared sandbox image required")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path):
    sessions = tmp_path / "data/agent-sessions"
    sessions.mkdir(parents=True)
    pi_home = tmp_path / "pi-home"
    pi_home.mkdir()
    (pi_home / "settings.json").write_text(json.dumps({"defaultProvider": "litellm", "defaultModel": "sandbox-probe"}))
    (pi_home / "models.json").write_text(json.dumps({"providers": {"litellm": {
        "baseUrl": "http://127.0.0.1:19867/v1", "api": "openai-completions", "apiKey": "probe-not-a-secret",
        "models": [{"id": "sandbox-probe", "name": "Sandbox probe", "reasoning": False,
                    "input": ["text"], "contextWindow": 200000, "maxTokens": 4000}],
    }}}))
    name = f"paper-reader-test-{uuid.uuid4().hex[:12]}"
    command = [
        "docker", "run", "-d", "--name", name, "--init", "--label", "ai.dexgem.paper-reader.sandbox=1",
        "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "2g", "--cpus", "2", "--pids-limit", "256", "--network", "paper-reader-agents",
        "--dns", "1.1.1.1", "--dns", "8.8.8.8", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--mount", f"type=bind,source={sessions},target=/workspaces",
        "--mount", f"type=bind,source={pi_home},target=/pi-home,readonly", IMAGE,
    ]
    if os.environ.get("PAPER_READER_TEST_LLM_ENV"):
        command[-1:-1] = ["--env-file", os.environ["PAPER_READER_TEST_LLM_ENV"]]
    subprocess.run(command, check=True, capture_output=True)
    adapter = SharedAgentSandbox(sessions_root=sessions, pi_dir=pi_home, container=name)
    adapter.validate()
    try:
        yield adapter, tmp_path
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


def execute(adapter, command, check=True):
    return subprocess.run(["docker", "exec", adapter.container, "bash", "-c", command], check=check, capture_output=True, text=True, timeout=15)


def test_actual_container_host_boundaries_and_two_workspaces(sandbox):
    adapter, root = sandbox
    sentinel = root / "host-only-sentinel"
    sentinel.write_text("host remains intact")
    for paper in ("paper-a", "paper-b"):
        (adapter.sessions_root / paper / "cwd").mkdir(parents=True)
    before = subprocess.check_output(["docker", "inspect", "-f", "{{.Id}}", adapter.container], text=True).strip()
    for paper in ("paper-a", "paper-b"):
        command = adapter.command(cwd=adapter.sessions_root / paper / "cwd", pi_bin="bash", pi_args=["-c", "pwd; echo result > result.txt"], timeout=5, turn_id=uuid.uuid4().hex)
        out = subprocess.run(command, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == f"/workspaces/{paper}/cwd"
        assert (adapter.sessions_root / paper / "cwd/result.txt").read_text() == "result\n"
    after = subprocess.check_output(["docker", "inspect", "-f", "{{.Id}}", adapter.container], text=True).strip()
    assert before == after  # same runtime for both papers, no per-paper container
    assert execute(adapter, f"test ! -e {sentinel} && test ! -e /run/docker.sock && test ! -e /var/run/docker.sock && test ! -e /etc/paper-reader").returncode == 0
    assert execute(adapter, "echo forbidden > /etc/sandbox-test", check=False).returncode != 0
    assert execute(adapter, "echo forbidden > /pi-home/settings.json", check=False).returncode != 0
    assert execute(adapter, "mount -t tmpfs tmpfs /mnt", check=False).returncode != 0
    assert sentinel.read_text() == "host remains intact"
    status = execute(adapter, "grep -E 'CapEff|NoNewPrivs' /proc/self/status").stdout
    assert "0000000000000000" in status and "NoNewPrivs:\t1" in status
    # An existing host process is absent from the container PID namespace.
    assert execute(adapter, f"test ! -e /proc/{os.getpid()}").returncode == 0
    # Even a host listener on all interfaces is unreachable through the bridge;
    # this exercises the INPUT rule, not merely a loopback-bound web service.
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        port = listener.getsockname()[1]
        gateway = json.loads(subprocess.check_output(["docker", "inspect", adapter.container]))[0]["NetworkSettings"]["Networks"]["paper-reader-agents"]["Gateway"]
        result = execute(adapter, f"python3 -c \"import socket; socket.create_connection(('{gateway}', {port}), timeout=2)\"", check=False)
        assert result.returncode != 0, "sandbox connected to host listener"
    # Private-network egress is rejected; public HTTPS remains usable.
    assert execute(adapter, "curl --max-time 2 -s http://169.254.169.254", check=False).returncode != 0
    assert "Example Domain" in execute(adapter, "curl --max-time 10 -fsS https://example.com").stdout


def test_actual_pi_bash_search_fetch_and_session_resume(sandbox):
    adapter, root = sandbox
    source = root / "paper.pdf"
    source.write_bytes(b"paper evidence")

    class Library:
        def resolve_relative_path(self, rel):
            return source
        def ensure_arxiv_markdown_for_rel_path(self, rel):
            return None
    team = SimpleNamespace(
        agent_paper_metadata=lambda rel: {"id": 42, "title": "Sandbox Paper", "sources": []},
        paper_context=lambda rel: {"attachments": []},
    )
    backend = PiAgentChat(
        data_dir=root / "data", pi_bin="pi", pi_dir=adapter.pi_dir, timeout_seconds=90,
        library=Library(), prompt_store=SimpleNamespace(list_prompts=lambda: []), team_store=team, sandbox=adapter,
    )
    cwd = backend.prepare_material_package("paper.pdf", session_key="shared:42", visibility="shared")
    shutil.copyfile(ROOT / "tests/fixtures/scripted_pi_provider.py", cwd / "provider.py")
    subprocess.run(["docker", "exec", "-d", adapter.container, "python3", str(adapter.container_path(cwd / "provider.py"))], check=True)
    # Scripted model calls real tools; inspect toolResult records, not just the
    # provider's synthetic final text, for successful external evidence.
    events = []
    result = backend.run_turn(rel_path="paper.pdf", visibility="shared", user_id=1, display_name="Alice",
                              question="probe tools", session_key="shared:42", on_event=events.append)
    assert "Sandbox tool results" in result
    assert {e.tool_name for e in events if e.tool_name} == {"bash", "Bash", "web_search", "web_fetch"}
    assert (cwd / "notes.txt").read_text() == "sandbox scratch\n"
    assert (cwd / "lowercase.txt").read_text() == "lowercase works\n"
    assert source.read_bytes() == b"paper evidence"
    files = list((backend.session_root_for("shared:42") / "sessions").rglob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    tools = {row["message"]["toolName"]: row["message"] for row in rows if row.get("message", {}).get("role") == "toolResult"}
    assert not tools["Bash"].get("isError")
    assert not tools["bash"].get("isError")
    assert "Example Domain" in json.dumps(tools["web_fetch"])
    search = tools["web_search"]
    search_text = json.dumps(search)
    # DDG is a fallback, not the deployment's native-search profile. A network
    # failure must be explicit in the tool content; live native success is a
    # separate credentialed test below.
    assert ("arxiv.org" in search_text and search.get("details", {}).get("method") == "ddg") or "Error:" in search_text, search
    # One log is resumed; older tool calls/finals are not projected again.
    events.clear()
    backend.run_turn(rel_path="paper.pdf", visibility="shared", user_id=2, display_name="Bob",
                     question="continue", session_key="shared:42", on_event=events.append)
    assert len(list(files[0].parent.rglob("*.jsonl"))) == 1
    assert [e.kind for e in events] == ["final"]


@pytest.mark.skipif(not os.environ.get("PAPER_READER_TEST_LLM_ENV"), reason="limited live-search credential required")
def test_actual_plugin_native_search_from_shared_sandbox(sandbox):
    adapter, root = sandbox
    cwd = adapter.sessions_root / "live-search/cwd"
    cwd.mkdir(parents=True)
    command = adapter.command(cwd=cwd, pi_bin="node", pi_args=["/opt/paper-reader/probe-search.mjs"], timeout=90, turn_id=uuid.uuid4().hex)
    result = subprocess.run(command, capture_output=True, text=True, timeout=100)
    artifact = cwd / "search-probe.json"
    assert result.returncode == 0, (result.stdout, result.stderr, artifact.read_text() if artifact.exists() else "no output")
    data = json.loads(artifact.read_text())
    assert data["details"]["method"] == "native"
    assert "arxiv.org" in json.dumps(data)


@pytest.mark.skipif(not os.environ.get("PAPER_READER_TEST_LLM_ENV"), reason="limited live-model credential required")
def test_live_paper_bot_completes_bash_and_search_task(sandbox):
    adapter, root = sandbox
    for name in ("settings.json", "models.json"):
        shutil.copyfile(ROOT / "ops/agent-sandbox/pi-home" / name, adapter.pi_dir / name)
    source = root / "paper.md"
    source.write_text("# Sandbox Test Paper\nThis is a local fixture, not the Transformer paper.\n")
    library = SimpleNamespace(
        resolve_relative_path=lambda rel: source,
        ensure_arxiv_markdown_for_rel_path=lambda rel: None,
    )
    team = SimpleNamespace(
        agent_paper_metadata=lambda rel: {"id": 9, "title": "Sandbox Test Paper", "sources": []},
        paper_context=lambda rel: {"attachments": []},
    )
    backend = PiAgentChat(
        data_dir=root / "data", pi_bin="pi", pi_dir=adapter.pi_dir, timeout_seconds=150,
        library=library, prompt_store=SimpleNamespace(list_prompts=lambda: []), team_store=team, sandbox=adapter,
    )
    events = []
    answer = backend.run_turn(
        rel_path="paper.md", visibility="private", user_id=1, display_name="Tester", session_key="private:1:9",
        question="请实际调用 bash 计算 6*7，把单独的数字保存到当前 cwd 的 answer.txt。然后调用 web_search 查找 Attention Is All You Need 的原始 arXiv 地址。简短回答计算结果和该 URL，注明该 URL 是外部来源。只在当前 cwd 内写文件。",
        on_event=events.append,
    )
    cwd = backend.session_root_for("private:1:9") / "cwd"
    assert (cwd / "answer.txt").read_text().strip() == "42"
    assert "1706.03762" in answer and "arxiv.org" in answer
    assert {"bash", "web_search"} <= {event.tool_name for event in events if event.tool_name}
    assert source.read_text().startswith("# Sandbox Test Paper")
