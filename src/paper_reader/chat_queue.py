from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agent_chat import AgentChatEvent, PiAgentChat
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
    agent_session_key: str | None = None


class PaperChatQueue:
    def __init__(
        self,
        *,
        library: Any,
        prompt_store: Any,
        team_store: Any,
        prompt_context_loader: Callable[[str, list[Any]], list[tuple[str, str]]],
        agent_chat: PiAgentChat | None = None,
        worker_count: int = MAX_CHAT_WORKERS,
    ) -> None:
        self.library = library
        self.prompt_store = prompt_store
        self.team_store = team_store
        self.prompt_context_loader = prompt_context_loader
        self.agent_chat = agent_chat
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
        agent_session_key: str | None = None,
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
                agent_session_key=agent_session_key,
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
            if self.agent_chat is not None and job.agent_session_key is not None:
                def update_progress(event: AgentChatEvent) -> None:
                    self.team_store.update_chat_message(
                        job.assistant_message_id,
                        body=event.text,
                        status="completed" if event.final else "pending",
                        model="pi",
                    )

                answer = self.agent_chat.run_turn(
                    rel_path=job.rel_path,
                    visibility=job.visibility,
                    user_id=job.user_id,
                    display_name=job.display_name,
                    question=job.question,
                    session_key=job.agent_session_key,
                    on_event=update_progress,
                    reserved=True,
                )
                model = "pi"
            else:
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
                    attachment_paths=self.team_store.attachment_paths_for_rel_path(job.rel_path),
                    model=job.model,
                )
                model = job.model
            self.team_store.update_chat_message(
                job.assistant_message_id,
                body=answer,
                status="completed",
                model=model,
            )
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "Too Many Requests" in error_text:
                body = "Paper Bot 现在有点忙，刚刚碰到了速率限制。请等十几秒，再把这条问题发一次。"
            else:
                body = f"这次回复没成功：{error_text}"
            if self.agent_chat is not None and job.agent_session_key is not None:
                self.agent_chat.release(job.agent_session_key)
            self.team_store.update_chat_message(
                job.assistant_message_id,
                body=body,
                status="failed",
                model=job.model,
            )
