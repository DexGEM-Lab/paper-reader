from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from src.paper_reader.agent_sandbox import SharedAgentSandbox

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ops/agent-sandbox/run-turn.py"


def good_info(tmp_path):
    return {
        "State": {"Running": True},
        "Config": {"User": "1000:1000", "Labels": {"ai.dexgem.paper-reader.sandbox": "1"}},
        "HostConfig": {
            "Privileged": False, "ReadonlyRootfs": True, "CapDrop": ["ALL"], "CapAdd": [],
            "SecurityOpt": ["no-new-privileges:true"], "NetworkMode": "paper-reader-agents",
            "PidMode": "", "IpcMode": "private", "Devices": [], "DeviceRequests": [],
            "PortBindings": {}, "Memory": 2**30, "PidsLimit": 256, "NanoCpus": 2_000_000_000,
        },
        "NetworkSettings": {"Networks": {"paper-reader-agents": {}}},
        "Mounts": [
            {"Type": "bind", "Source": str(tmp_path / "sessions"), "Destination": "/workspaces", "RW": True},
            {"Type": "bind", "Source": str(tmp_path / "pi"), "Destination": "/pi-home", "RW": False},
        ],
    }


def test_sandbox_maps_paths_and_validates_exact_export(tmp_path):
    info = good_info(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([info]))

    sandbox = SharedAgentSandbox(sessions_root=tmp_path / "sessions", pi_dir=tmp_path / "pi", run=run)
    sandbox.validate()
    assert calls == [["docker", "inspect", "paper-reader-agents"]]
    assert str(sandbox.container_path(tmp_path / "sessions" / "paper-1" / "cwd")) == "/workspaces/paper-1/cwd"
    with pytest.raises(ValueError):
        sandbox.container_path(tmp_path / "outside")
    command = sandbox.command(cwd=tmp_path / "sessions/paper-1/cwd", pi_args=["-p", "hello"], pi_bin="pi", timeout=2, turn_id="turn")
    assert command[:5] == ["docker", "exec", "--workdir", "/workspaces/paper-1/cwd", "paper-reader-agents"]
    assert command[-3:] == ["pi", "-p", "hello"]


@pytest.mark.parametrize("change", [
    lambda i: i["State"].update(Running=False),
    lambda i: i["Config"].update(User="0:0"),
    lambda i: i["Config"]["Labels"].clear(),
    lambda i: i["HostConfig"].update(Privileged=True),
    lambda i: i["HostConfig"].update(ReadonlyRootfs=False),
    lambda i: i["HostConfig"].update(CapDrop=[]),
    lambda i: i["HostConfig"].update(CapAdd=["SYS_ADMIN"]),
    lambda i: i["HostConfig"].update(SecurityOpt=[]),
    lambda i: i["HostConfig"].update(SecurityOpt=["no-new-privileges", "seccomp=unconfined"]),
    lambda i: i["HostConfig"].update(NetworkMode="host"),
    lambda i: i["NetworkSettings"]["Networks"].update(other={}),
    lambda i: i["HostConfig"].update(PidMode="host"),
    lambda i: i["HostConfig"].update(IpcMode="host"),
    lambda i: i["HostConfig"].update(Devices=[{"PathOnHost": "/dev/sda"}]),
    lambda i: i["HostConfig"].update(PortBindings={"80/tcp": [{}]}),
    lambda i: i["HostConfig"].update(Memory=0),
    lambda i: i["HostConfig"].update(PidsLimit=-1),
    lambda i: i["HostConfig"].update(NanoCpus=0),
    lambda i: i["Mounts"][0].update(Source="/"),
    lambda i: i["Mounts"][1].update(RW=True),
    lambda i: i["Mounts"].append({"Type": "bind", "Source": "/run/docker.sock", "Destination": "/run/docker.sock", "RW": True}),
])
def test_sandbox_rejects_unsafe_runtime_without_fallback(tmp_path, change):
    info = deepcopy(good_info(tmp_path))
    change(info)
    sandbox = SharedAgentSandbox(
        sessions_root=tmp_path / "sessions", pi_dir=tmp_path / "pi",
        run=lambda *a, **kw: SimpleNamespace(stdout=json.dumps([info])),
    )
    with pytest.raises(RuntimeError, match="不会回退"):
        sandbox.validate()


def test_sandbox_missing_daemon_fails_explicitly(tmp_path):
    def unavailable(*a, **kw):
        raise FileNotFoundError("docker")
    sandbox = SharedAgentSandbox(sessions_root=tmp_path / "sessions", pi_dir=tmp_path / "pi", run=unavailable)
    with pytest.raises(RuntimeError, match="不会回退"):
        sandbox.validate()


@pytest.mark.parametrize("mode", ["normal", "timeout", "cancel"])
def test_turn_supervisor_reaps_detached_children(tmp_path, mode):
    if sys.platform != "linux":
        pytest.skip("Linux subreaper")
    child_pid_file = tmp_path / "child.pid"
    child = (
        "import subprocess,time,pathlib,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); "
        + ("time.sleep(60)" if mode != "normal" else "time.sleep(0.1)")
    )
    turn_id = uuid.uuid4().hex
    process = subprocess.Popen(
        [sys.executable, str(RUNNER), "run", turn_id, "0.5" if mode == "timeout" else "10", sys.executable, "-c", child],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.01)
    assert child_pid_file.exists()
    pid = int(child_pid_file.read_text())
    if mode == "cancel":
        subprocess.run([sys.executable, str(RUNNER), "cancel", turn_id], check=True)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == {"normal": 0, "timeout": 124, "cancel": 130}[mode], (stdout, stderr)
    assert not Path(f"/proc/{pid}").exists(), "detached bash descendant survived"
    assert not Path(f"/tmp/paper-reader-turn-{turn_id}.pid").exists()
