from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .chat_ai import answer_question_about_document
from .config import load_config

MAX_CHAT_WORKERS = 1


@dataclass(slots=True)
class ChatJob:
    rel_path: str
    visibility: str
    user_id: int
    display_name: str
    question: str
    history: list[dict[str, str]]
    assistant_message_id: int
    model: str


class PaperChatQueue:
    def __init__(
        self,
        *,
        library: Any,
        prompt_store: Any,
        team_store: Any,
        prompt_context_loader: Callable[[str, list[Any]], list[tuple[str, str]]],
        worker_count: int = MAX_CHAT_WORKERS,
    ) -> None:
        self.library = library
        self.prompt_store = prompt_store
        self.team_store = team_store
        self.prompt_context_loader = prompt_context_loader
        self._queue: queue.Queue[ChatJob] = queue.Queue()
        self.team_store.mark_pending_chat_messages_failed()
        self._workers: list[threading.Thread] = []
        for index in range(max(1, int(worker_count))):
            worker = threading.Thread(target=self._run_loop, name=f"paper-reader-chat-{index + 1}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit(
        self,
        *,
        rel_path: str,
        visibility: str,
        user_id: int,
        display_name: str,
        question: str,
        history: list[dict[str, str]],
        assistant_message_id: int,
        model: str | None = None,
    ) -> None:
        self._queue.put(
            ChatJob(
                rel_path=rel_path,
                visibility=visibility,
                user_id=user_id,
                display_name=display_name,
                question=question,
                history=history,
                assistant_message_id=assistant_message_id,
                model=model or load_config().llm_model,
            )
        )

    def _run_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._process_job(job)
            finally:
                self._queue.task_done()

    def _process_job(self, job: ChatJob) -> None:
        try:
            document_path = Path(self.library.resolve_relative_path(job.rel_path))
            arxiv_markdown = self.library.ensure_arxiv_markdown_for_rel_path(job.rel_path)
            prompt_contexts = self.prompt_context_loader(job.rel_path, self.prompt_store.list_prompts())
            answer = answer_question_about_document(
                document_path,
                question=job.question,
                visibility=job.visibility,
                history=job.history,
                prompt_contexts=prompt_contexts,
                arxiv_markdown_path=(arxiv_markdown.markdown_path if arxiv_markdown is not None else None),
                model=job.model,
            )
            self.team_store.update_chat_message(
                job.assistant_message_id,
                body=answer,
                status="completed",
                model=job.model,
            )
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "Too Many Requests" in error_text:
                body = "Paper Bot 现在有点忙，刚刚碰到了速率限制。请等十几秒，再把这条问题发一次。"
            else:
                body = f"这次回复没成功：{error_text}"
            self.team_store.update_chat_message(
                job.assistant_message_id,
                body=body,
                status="failed",
                model=job.model,
            )
