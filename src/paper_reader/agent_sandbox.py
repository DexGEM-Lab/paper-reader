"""Host-side adapter for the single, operator-provisioned paper-bot container.

No agent command runs on the host. Container creation is an operator action;
missing/misconfigured sandboxes fail closed instead of falling back to host Pi.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

SANDBOX_LABEL = "ai.dexgem.paper-reader.sandbox"
SANDBOX_VERSION = "1"
WORKSPACES = Path("/workspaces")
PI_HOME = Path("/pi-home")
RUNNER = "/opt/paper-reader/run-turn.py"
SEARCH_EXTENSION = "/opt/pi-native-search/extensions/index.ts"
TOOLS = "read,bash,edit,write,grep,find,ls,web_search,web_fetch"
CWD_RULE = (
    "You have web_search, web_fetch and general bash tools. Use them to complete the user's task. "
    "Only create, modify, or delete files inside the initial current working directory for this paper session. "
    "Do not change files outside that directory, including other papers' workspaces, even after cd or via symlinks. "
    "Keep downloads, code, notes and temporary files inside the initial cwd. "
    "Cite external sources with their URLs and distinguish them from local paper evidence."
)


class SharedAgentSandbox:
    def __init__(
        self,
        *,
        sessions_root: Path,
        pi_dir: Path,
        container: str = "paper-reader-agents",
        docker_bin: str = "docker",
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.sessions_root = sessions_root.resolve()
        self.pi_dir = pi_dir.resolve()
        self.container = container
        self.docker_bin = docker_bin
        self.run = run

    def container_path(self, host_path: Path) -> Path:
        # resolve() rejects a symlink pointing outside the exported session tree.
        return WORKSPACES / host_path.resolve().relative_to(self.sessions_root)

    def validate(self) -> None:
        try:
            result = self.run(
                [self.docker_bin, "inspect", self.container],
                capture_output=True, text=True, timeout=10, check=True,
            )
            info = json.loads(result.stdout)[0]
            self._validate_info(info)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "共享 agent sandbox 不可用或隔离配置不匹配；请按 ops/agent-sandbox/README.md 配置，"
                "不会回退到宿主机执行。"
            ) from exc

    def _validate_info(self, info: dict[str, Any]) -> None:
        config = info["Config"]
        host = info["HostConfig"]
        if not info["State"]["Running"] or config.get("Labels", {}).get(SANDBOX_LABEL) != SANDBOX_VERSION:
            raise ValueError("not a running Paper Reader sandbox")
        # Require numeric non-root identities, matching host bind ownership.
        uid, gid = config["User"].split(":")
        if not uid.isdecimal() or not gid.isdecimal() or int(uid) == 0 or int(gid) == 0:
            raise ValueError("sandbox must run as non-root uid:gid")
        if host.get("Privileged") or not host.get("ReadonlyRootfs"):
            raise ValueError("sandbox must be unprivileged with a read-only root")
        if "ALL" not in {str(cap).upper() for cap in host.get("CapDrop") or []} or host.get("CapAdd"):
            raise ValueError("all capabilities must be dropped")
        security = host.get("SecurityOpt") or []
        if not any(value in security for value in ("no-new-privileges", "no-new-privileges:true")):
            raise ValueError("no-new-privileges required")
        if any("unconfined" in value for value in security):
            raise ValueError("default seccomp/LSM confinement required")
        if (host.get("NetworkMode") != "paper-reader-agents"
                or set(info["NetworkSettings"]["Networks"]) != {"paper-reader-agents"}
                or host.get("PidMode") or host.get("IpcMode") == "host"):
            raise ValueError("private network/PID/IPC namespaces with egress required")
        if host.get("Devices") or host.get("DeviceRequests") or host.get("PortBindings"):
            raise ValueError("no devices or published ports allowed")
        expected = {
            str(WORKSPACES): (self.sessions_root, True),
            str(PI_HOME): (self.pi_dir, False),
        }
        mounts = info["Mounts"]
        binds = [mount for mount in mounts if mount["Type"] != "tmpfs"]
        if len(binds) != len(expected):
            raise ValueError("unexpected host mount")
        for mount in binds:
            source, writable = expected[mount["Destination"]]
            if mount["Type"] != "bind" or Path(mount["Source"]).resolve() != source or mount["RW"] != writable:
                raise ValueError("sandbox mount mismatch")
        if not host.get("Memory") or not host.get("PidsLimit") or host["PidsLimit"] < 0 or not host.get("NanoCpus"):
            raise ValueError("shared memory, PID and CPU limits required")

    def command(self, *, cwd: Path, pi_args: list[str], pi_bin: str, timeout: float, turn_id: str) -> list[str]:
        return [
            self.docker_bin, "exec", "--workdir", str(self.container_path(cwd)),
            self.container, "python3", RUNNER,
            "run", turn_id, str(timeout), pi_bin, *pi_args,
        ]

    def stop(self, process: Any, turn_id: str) -> None:
        if process.poll() is None:
            try:
                self.cancel(turn_id)
            finally:
                process.kill()
                process.wait(timeout=5)

    def cancel(self, turn_id: str) -> None:
        self.run(
            [self.docker_bin, "exec", self.container, "python3", RUNNER, "cancel", turn_id],
            capture_output=True, text=True, timeout=10, check=True,
        )
