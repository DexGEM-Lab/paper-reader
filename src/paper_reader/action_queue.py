from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ACTION_STORE_NAME = ".paper-reader-actions.json"
ACTIVE_ACTION_STATUSES = {"queued", "running"}
TERMINAL_ACTION_STATUSES = {"completed", "failed", "stopped"}
DEFAULT_ACTION_MAX_CONCURRENCY = 4
MAX_ACTION_WORKERS = 8
MAX_PERSISTED_TERMINAL_ACTIONS = 200


@dataclass
class ActionRecord:
    id: str
    kind: str
    title: str
    source: str
    requested_by_user_id: int | None
    requested_by_display_name: str | None
    status: str
    progress: int
    message: str
    error: str | None
    result: dict[str, Any]
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


ActionHandler = Callable[[ActionRecord, Callable[[int, str, dict[str, Any] | None], None], Callable[[], bool]], dict[str, Any] | None]


class ActionTaskQueue:
    def __init__(self, root: Path, *, state_file_name: str = ACTION_STORE_NAME, max_concurrency: int = DEFAULT_ACTION_MAX_CONCURRENCY):
        self.root = root
        self.state_path = self.root / state_file_name
        self.max_concurrency = max(1, int(max_concurrency))
        self._lock = threading.Lock()
        self._slot_condition = threading.Condition()
        self._active_executions = 0
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._actions: dict[str, ActionRecord] = {}
        self._handlers: dict[str, ActionHandler] = {}
        self._cancelled_actions: set[str] = set()
        self._load_state()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        with self._lock:
            if self._workers:
                return
            for index in range(MAX_ACTION_WORKERS):
                worker = threading.Thread(target=self._run_loop, name=f"paper-reader-action-{index + 1}", daemon=True)
                worker.start()
                self._workers.append(worker)

    def register_handler(self, kind: str, handler: ActionHandler) -> None:
        with self._lock:
            self._handlers[kind] = handler

    def stop(self, *, join_timeout: float = 1.0) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers = []
        if not workers:
            return
        for _ in workers:
            self._queue.put(None)
        for worker in workers:
            worker.join(timeout=join_timeout)

    def _timestamp(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _load_state(self) -> None:
        payload: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}

        now = self._timestamp()
        for item in payload.get("actions", []):
            if not isinstance(item, dict):
                continue
            try:
                action = ActionRecord(**item)
            except TypeError:
                continue
            if action.status in ACTIVE_ACTION_STATUSES:
                action.status = "queued"
                action.progress = min(action.progress, 5)
                action.message = "服务重启后已重新排队。"
                action.updated_at = now
            self._actions[action.id] = action

        self._persist_locked()
        for action in self._actions.values():
            if action.status == "queued":
                self._queue.put(action.id)

    def _persist_locked(self) -> None:
        terminal_actions = sorted(
            (action for action in self._actions.values() if action.status in TERMINAL_ACTION_STATUSES),
            key=lambda item: (item.updated_at, item.created_at),
            reverse=True,
        )
        keep_terminal_ids = {action.id for action in terminal_actions[:MAX_PERSISTED_TERMINAL_ACTIONS]}
        self._actions = {
            action_id: action
            for action_id, action in self._actions.items()
            if action.status in ACTIVE_ACTION_STATUSES or action_id in keep_terminal_ids
        }
        payload = {"actions": [asdict(action) for action in sorted(self._actions.values(), key=lambda item: item.created_at, reverse=True)]}
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def submit(
        self,
        *,
        kind: str,
        title: str,
        payload: dict[str, Any],
        source: str,
        requested_by_user_id: int | None = None,
        requested_by_display_name: str | None = None,
    ) -> ActionRecord:
        now = self._timestamp()
        action = ActionRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            title=title,
            source=source,
            requested_by_user_id=requested_by_user_id,
            requested_by_display_name=requested_by_display_name,
            status="queued",
            progress=0,
            message="任务已提交，等待处理。",
            error=None,
            result={},
            payload=payload,
            created_at=now,
            updated_at=now,
            started_at=None,
            finished_at=None,
        )
        with self._lock:
            self._actions[action.id] = action
            self._queue.put(action.id)
            self._persist_locked()
        return action

    def _update_action(self, action_id: str, **changes: Any) -> ActionRecord | None:
        with self._lock:
            action = self._actions.get(action_id)
            if action is None:
                return None
            for key, value in changes.items():
                setattr(action, key, value)
            action.updated_at = self._timestamp()
            if action.status == "running" and action.started_at is None:
                action.started_at = action.updated_at
            if action.status in TERMINAL_ACTION_STATUSES and action.finished_at is None:
                action.finished_at = action.updated_at
            self._persist_locked()
            return action

    def _run_loop(self) -> None:
        while True:
            action_id = self._queue.get()
            try:
                if action_id is None:
                    return
                self._process_action(action_id)
            finally:
                self._queue.task_done()

    def _process_action(self, action_id: str) -> None:
        with self._lock:
            action = self._actions.get(action_id)
            handler = self._handlers.get(action.kind) if action is not None else None
            if action is None or handler is None or action.status in TERMINAL_ACTION_STATUSES or action.status == "stopped":
                return

        if not self._acquire_execution_slot(action_id):
            return

        try:
            action = self._update_action(action_id, status="running", progress=1, message="任务开始执行。")
            if action is None or self._is_cancelled(action_id):
                self._mark_stopped(action_id)
                return

            def report(progress: int, message: str, result: dict[str, Any] | None = None) -> None:
                if self._is_cancelled(action_id):
                    raise InterruptedError("Action interrupted.")
                changes: dict[str, Any] = {
                    "status": "running",
                    "progress": max(1, min(progress, 99)),
                    "message": message,
                }
                if result is not None:
                    changes["result"] = result
                self._update_action(action_id, **changes)

            handler_result = handler(action, report, lambda: self._is_cancelled(action_id))
        except InterruptedError:
            self._mark_stopped(action_id)
            return
        except Exception as exc:
            if self._is_cancelled(action_id):
                self._mark_stopped(action_id)
                return
            self._update_action(
                action_id,
                status="failed",
                progress=100,
                message="任务执行失败。",
                error=str(exc),
            )
            return
        finally:
            self._release_execution_slot()

        completed_message = "任务完成。"
        if isinstance(handler_result, dict):
            completed_message = str(handler_result.get("_final_message") or completed_message)
        self._update_action(
            action_id,
            status="completed",
            progress=100,
            message=completed_message,
            error=None,
            result=handler_result or {},
        )

    def _acquire_execution_slot(self, action_id: str) -> bool:
        while True:
            with self._slot_condition:
                if self._is_cancelled(action_id):
                    return False
                if self._active_executions < self.max_concurrency:
                    self._active_executions += 1
                    return True
                self._slot_condition.wait(timeout=0.5)

    def _release_execution_slot(self) -> None:
        with self._slot_condition:
            self._active_executions = max(0, self._active_executions - 1)
            self._slot_condition.notify_all()

    def _is_cancelled(self, action_id: str) -> bool:
        with self._lock:
            return action_id in self._cancelled_actions

    def _mark_stopped(self, action_id: str) -> None:
        self._update_action(
            action_id,
            status="stopped",
            progress=100,
            message="任务已停止。",
            error="Interrupted",
        )

    def get(self, action_id: str) -> ActionRecord | None:
        with self._lock:
            action = self._actions.get(action_id)
            return ActionRecord(**asdict(action)) if action is not None else None

    def snapshot(self, *, kind: str | None = None, limit: int = 24) -> dict[str, Any]:
        with self._lock:
            all_actions = list(self._actions.values())
        if kind:
            all_actions = [action for action in all_actions if action.kind == kind]
        all_actions.sort(key=lambda item: (item.updated_at, item.created_at), reverse=True)
        actions = all_actions[:limit]
        with self._slot_condition:
            active_executions = self._active_executions
        return {
            "actions": [asdict(action) for action in actions],
            "active_count": sum(1 for action in all_actions if action.status in ACTIVE_ACTION_STATUSES),
            "queued_count": sum(1 for action in all_actions if action.status == "queued"),
            "running_count": sum(1 for action in all_actions if action.status == "running"),
            "max_concurrency": self.max_concurrency,
            "active_executions": active_executions,
        }
