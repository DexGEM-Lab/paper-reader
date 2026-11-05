from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class AgentChatEvent:
    kind: str
    text: str
    final: bool = False
    tool_name: str | None = None


def agent_session_key(*, visibility: str, user_id: int, paper_id: int) -> str:
    if visibility == "shared":
        return f"shared:{paper_id}"
    if visibility == "private":
        return f"private:{user_id}:{paper_id}"
    raise ValueError("不支持的聊天模式。")


def agent_session_slug(key: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-._") or "session"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}"


def _assistant_parts(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if row.get("type") != "message":
        return []
    message = row.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, Mapping)]


def _assistant_text(row: Mapping[str, Any]) -> str | None:
    pieces = [
        str(part.get("text"))
        for part in _assistant_parts(row)
        if part.get("type") == "text" and isinstance(part.get("text"), str) and str(part.get("text")).strip()
    ]
    return "".join(pieces) if pieces else None


def pi_assistant_is_final_turn_end(row: Mapping[str, Any]) -> bool:
    """Port Pi's final-answer discriminator without depending on codoxear."""
    if row.get("type") != "message":
        return False
    message = row.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return False
    if message.get("stopReason") == "length" or not _assistant_text(row):
        return False
    parts = _assistant_parts(row)
    if any(part.get("type") == "toolCall" for part in parts):
        return False

    has_thinking = any(part.get("type") == "thinking" for part in parts)
    stop_reason = message.get("stopReason")
    if not has_thinking and stop_reason != "toolUse":
        return True
    if isinstance(stop_reason, str) and stop_reason and stop_reason != "toolUse":
        return True

    for part in parts:
        if part.get("type") != "text":
            continue
        signature = part.get("textSignature")
        if not isinstance(signature, str) or not signature.strip():
            continue
        try:
            decoded = json.loads(signature)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, Mapping) and decoded.get("phase") == "final_answer":
            return True
    return False


def extract_pi_chat_events(row: Mapping[str, Any]) -> list[AgentChatEvent]:
    """Project one Pi JSONL row into quiet, user-facing chat events."""
    if row.get("type") != "message":
        return []
    message = row.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return []

    stop_reason = message.get("stopReason")
    if stop_reason == "error":
        detail = message.get("errorMessage")
        text = str(detail).strip() if isinstance(detail, str) and detail.strip() else "Pi 会话执行失败。"
        return [AgentChatEvent(kind="error", text=text)]
    if stop_reason == "aborted":
        return [AgentChatEvent(kind="error", text="Pi 会话已中止。")]

    events: list[AgentChatEvent] = []
    for part in _assistant_parts(row):
        if part.get("type") != "toolCall":
            continue
        raw_name = part.get("name") or part.get("toolName")
        tool_name = str(raw_name).strip() if raw_name else "工具"
        events.append(
            AgentChatEvent(
                kind="status",
                text=f"正在使用工具: {tool_name}",
                tool_name=tool_name,
            )
        )

    text = _assistant_text(row)
    if text:
        is_final = pi_assistant_is_final_turn_end(row)
        events.append(AgentChatEvent(kind="final" if is_final else "narration", text=text, final=is_final))
    return events


def _safe_file_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value).name).strip("-.")
    return (cleaned or fallback)[:160]


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


class PiAgentChat:
    def __init__(
        self,
        *,
        data_dir: Path,
        pi_bin: str,
        pi_dir: Path,
        timeout_seconds: float,
        library: Any,
        prompt_store: Any,
        team_store: Any,
        popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        poll_interval: float = 0.08,
    ) -> None:
        self.sessions_root = Path(data_dir).resolve() / "agent-sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.pi_bin = pi_bin
        self.pi_dir = Path(pi_dir).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.library = library
        self.prompt_store = prompt_store
        self.team_store = team_store
        self.popen_factory = popen_factory
        self.poll_interval = poll_interval
        self._busy: set[str] = set()
        self._busy_lock = threading.Lock()

    def session_key_for(self, rel_path: str, *, visibility: str, user_id: int) -> str:
        metadata = self.team_store.agent_paper_metadata(rel_path)
        if metadata is None:
            raise FileNotFoundError(rel_path)
        return agent_session_key(visibility=visibility, user_id=user_id, paper_id=int(metadata["id"]))

    def try_reserve(self, key: str) -> bool:
        with self._busy_lock:
            if key in self._busy:
                return False
            self._busy.add(key)
            return True

    def release(self, key: str) -> None:
        with self._busy_lock:
            self._busy.discard(key)

    def session_root_for(self, key: str) -> Path:
        return self.sessions_root / agent_session_slug(key)

    def prepare_material_package(
        self,
        rel_path: str,
        *,
        session_key: str,
        visibility: str,
    ) -> Path:
        metadata = self.team_store.agent_paper_metadata(rel_path)
        if metadata is None:
            raise FileNotFoundError(rel_path)
        session_root = self.session_root_for(session_key)
        cwd = session_root / "cwd"
        cwd.mkdir(parents=True, exist_ok=True)

        next_materials = cwd / f".materials-next-{uuid.uuid4().hex}"
        next_materials.mkdir(parents=True)
        manifest: list[tuple[str, str]] = []

        def add_file(source: Path, section: str, label: str, index: int) -> None:
            if not source.is_file():
                return
            filename = _safe_file_name(source.name, f"file-{index}")
            target = next_materials / section / f"{index:02d}-{filename}"
            _link_or_copy(source, target)
            manifest.append((label, f"materials/{target.relative_to(next_materials).as_posix()}"))

        source_path = Path(self.library.resolve_relative_path(rel_path))
        source_label = "网页快照" if source_path.suffix.lower() in {".md", ".markdown"} else "论文原文件"
        add_file(source_path, "source", source_label, 1)

        try:
            arxiv_info = self.library.ensure_arxiv_markdown_for_rel_path(rel_path)
        except Exception:
            arxiv_info = None
        if arxiv_info is not None:
            markdown_path = Path(arxiv_info.markdown_path)
            if markdown_path.resolve() != source_path.resolve():
                add_file(markdown_path, "snapshots", "arXiv 网页快照", 1)

        context = self.team_store.paper_context(rel_path)
        for index, attachment in enumerate(context.get("attachments", []), start=1):
            local_path = getattr(attachment, "local_path", None)
            if local_path:
                candidate = (Path(self.team_store.root) / str(local_path)).resolve()
                title = str(getattr(attachment, "title", "附件"))
                add_file(candidate, "attachments", f"附件：{title}", index)

        seen_results: set[str] = set()
        for prompt in self.prompt_store.list_prompts():
            info = self.library.prompt_result_info(rel_path, prompt.slug)
            result_rel_path = info.get("result_rel_path") if isinstance(info, Mapping) else None
            if not result_rel_path or str(result_rel_path) in seen_results:
                continue
            try:
                result_path = Path(self.library.resolve_relative_path(str(result_rel_path)))
            except (FileNotFoundError, ValueError):
                continue
            seen_results.add(str(result_rel_path))
            add_file(result_path, "interpretations", f"已有解读：{prompt.name}", len(seen_results))

        materials = cwd / "materials"
        if materials.exists() or materials.is_symlink():
            if materials.is_dir() and not materials.is_symlink():
                shutil.rmtree(materials)
            else:
                materials.unlink()
        next_materials.rename(materials)

        sources = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
        provenance_url = str(metadata.get("canonical_url") or "").strip()
        if not provenance_url:
            provenance_url = next(
                (str(item.get("source_url")) for item in sources if isinstance(item, Mapping) and item.get("source_url")),
                "未记录",
            )
        title = str(metadata.get("display_title") or metadata.get("title") or source_path.name)
        readme_lines = [
            f"# {title}",
            "",
            f"- 原始条目：`{rel_path}`",
            f"- 来源 URL：{provenance_url}",
            f"- 会话：`{session_key}`",
            "",
            "## 本地材料",
            "",
        ]
        readme_lines.extend(f"- **{label}**：`{path}`" for label, path in manifest)
        remote_attachments = [
            attachment for attachment in context.get("attachments", []) if not getattr(attachment, "local_path", None)
        ]
        for attachment in remote_attachments:
            readme_lines.append(
                f"- **远程附件：{getattr(attachment, 'title', '附件')}**：{getattr(attachment, 'source_url', '')}"
            )
        if not manifest:
            readme_lines.append("- 暂无可用的本地材料。")
        (cwd / "README.md").write_text("\n".join(readme_lines).rstrip() + "\n", encoding="utf-8")

        team_context = (
            "这是团队共享会话。不同成员的提问会带成员署名；结合此前对话继续回答。"
            if visibility == "shared"
            else "这是当前用户针对本论文的私人学习会话；不要假定团队其他成员能看到本会话。"
        )
        agents = f"""# Paper Reader 研究阅读伴侣

你是论文《{title}》的研究阅读伴侣。

- 论文标题：{title}
- 来源：{provenance_url}
- 团队上下文：{team_context}
- 本地材料清单：先读 `README.md`，证据文件位于 `materials/`。

## 工作规则

1. 用户使用中文时用中文回答；用户使用英文时用英文回答。
2. 结论必须尽量引用本地文件证据，注明页码、章节或文件路径；不确定时明确指出证据边界。
3. 用户要求提取数据、写代码、起草文本或执行其他任务时，直接完成任务。你在当前目录拥有完整工具权限。
4. 保持回答紧凑，优先给出结论与可执行结果，避免复述问题。
5. 实质性回答结尾必须列出“来源定位”，给出页码、章节或本地文件路径。
"""
        (cwd / "AGENTS.md").write_text(agents, encoding="utf-8")
        return cwd

    def run_turn(
        self,
        *,
        rel_path: str,
        visibility: str,
        user_id: int,
        display_name: str,
        question: str,
        session_key: str,
        on_event: Callable[[AgentChatEvent], None] | None = None,
        reserved: bool = False,
    ) -> str:
        if not reserved and not self.try_reserve(session_key):
            raise RuntimeError("agent busy")
        emit = on_event or (lambda event: None)
        try:
            cwd = self.prepare_material_package(rel_path, session_key=session_key, visibility=visibility)
            prompt = question.strip()
            if visibility == "shared":
                prompt = f"[团队成员：{display_name}]\n{prompt}"
            return self._invoke_pi(cwd=cwd, session_key=session_key, prompt=prompt, emit=emit)
        finally:
            self.release(session_key)

    def _session_files(self, sessions_dir: Path) -> list[Path]:
        return sorted(
            (path for path in sessions_dir.rglob("*.jsonl") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )

    def _invoke_pi(
        self,
        *,
        cwd: Path,
        session_key: str,
        prompt: str,
        emit: Callable[[AgentChatEvent], None],
    ) -> str:
        session_root = self.session_root_for(session_key)
        sessions_dir = session_root / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        existing_files = self._session_files(sessions_dir)
        log_path = existing_files[-1] if existing_files else None
        offset = log_path.stat().st_size if log_path is not None else 0

        command = [self.pi_bin]
        if log_path is not None:
            command.extend(["--session", str(log_path)])
        else:
            command.extend(["--session-dir", str(sessions_dir)])
        command.extend(["-p", prompt])

        env = os.environ.copy()
        env["PI_CODING_AGENT_DIR"] = str(self.pi_dir)
        stderr_path = session_root / "last-stderr.log"
        started = time.monotonic()
        with stderr_path.open("wb") as stderr_handle:
            try:
                process = self.popen_factory(
                    command,
                    cwd=str(cwd),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_handle,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("Pi agent 未安装或 PAPER_READER_PI_BIN 配置错误。") from exc

            final_text = ""
            error_text = ""
            while True:
                if log_path is None:
                    files = self._session_files(sessions_dir)
                    if files:
                        log_path = files[-1]
                        offset = 0
                if log_path is not None:
                    rows, offset = self._read_complete_rows(log_path, offset)
                    for row in rows:
                        for event in extract_pi_chat_events(row):
                            if event.kind == "error":
                                error_text = event.text
                                continue
                            emit(event)
                            if event.final:
                                final_text = event.text
                                error_text = ""
                return_code = process.poll()
                if return_code is not None:
                    if log_path is not None:
                        rows, offset = self._read_complete_rows(log_path, offset)
                        for row in rows:
                            for event in extract_pi_chat_events(row):
                                if event.kind == "error":
                                    error_text = event.text
                                    continue
                                emit(event)
                                if event.final:
                                    final_text = event.text
                                    error_text = ""
                    break
                if time.monotonic() - started >= self.timeout_seconds:
                    process.kill()
                    process.wait(timeout=5)
                    raise RuntimeError(f"Pi agent 超过 {int(self.timeout_seconds)} 秒未完成，已停止本轮任务。")
                time.sleep(self.poll_interval)

        if error_text:
            emit(AgentChatEvent(kind="error", text=error_text))
            raise RuntimeError(error_text)
        if return_code != 0:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            detail = stderr_text[-800:] if stderr_text else f"退出码 {return_code}"
            raise RuntimeError(f"Pi agent 执行失败：{detail}")
        if not final_text.strip():
            raise RuntimeError("Pi agent 已结束，但没有生成最终回答。")
        return final_text.strip()

    @staticmethod
    def _read_complete_rows(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read()
        except FileNotFoundError:
            return [], offset
        if not payload:
            return [], offset
        complete_end = payload.rfind(b"\n")
        if complete_end < 0:
            return [], offset
        consumed = payload[: complete_end + 1]
        rows: list[dict[str, Any]] = []
        for raw_line in consumed.splitlines():
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows, offset + len(consumed)
