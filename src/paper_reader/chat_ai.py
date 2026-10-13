from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from .ai_summary import SUPPORTED_SUMMARY_EXTENSIONS, extract_document_context
from .config import load_config
from .document_utils import UnsupportedDocumentError
from .llm import get_llm_client, resolve_model

MAX_CHAT_HISTORY_MESSAGES = 12
MAX_PROMPT_CONTEXT_CHARS = 12000


def _format_history(history: Sequence[dict[str, str]]) -> str:
    if not history:
        return "(暂无历史消息)"

    lines: list[str] = []
    for item in history[-MAX_CHAT_HISTORY_MESSAGES:]:
        role = str(item.get("role") or "user")
        display_name = str(item.get("display_name") or ("Paper Bot" if role == "assistant" else "成员"))
        body = str(item.get("body") or "").strip()
        if body:
            lines.append(f"[{role}] {display_name}:\n{body}")
    return "\n\n".join(lines) if lines else "(暂无历史消息)"


def _format_prompt_context(prompt_contexts: Sequence[tuple[str, str]]) -> str:
    if not prompt_contexts:
        return "(暂无可复用的 Prompt 结果)"

    remaining = MAX_PROMPT_CONTEXT_CHARS
    blocks: list[str] = []
    for name, content in prompt_contexts:
        snippet = content.strip()
        if len(snippet) > remaining:
            snippet = snippet[: max(0, remaining)].rstrip()
        if not snippet:
            break
        blocks.append(f"### {name}\n{snippet}")
        remaining -= len(snippet)
        if remaining <= 0:
            break
    return "\n\n".join(blocks) if blocks else "(暂无可复用的 Prompt 结果)"


def build_chat_prompt(
    document_path: Path,
    *,
    question: str,
    visibility: str,
    history: Sequence[dict[str, str]],
    prompt_contexts: Sequence[tuple[str, str]],
    arxiv_markdown_path: Path | None = None,
) -> str:
    mode = "Shared Chat" if visibility == "shared" else "Private Chat"
    collaboration_hint = (
        "请用适合团队协作的口吻回答，优先给出可讨论、可执行的结论。"
        if visibility == "shared"
        else "请用适合个人学习和复现的口吻回答，可以更细致地拆解思路。"
    )
    title, document_text, extraction_note = extract_document_context(
        document_path,
        source_markdown_path=arxiv_markdown_path,
        max_chars=load_config().llm_context_tokens,
    )
    return (
        "你是 paper-reader 里的论文问答助手。\n\n"
        f"论文标题：{title}\n"
        f"当前模式：{mode}\n"
        f"{collaboration_hint}\n"
        f"{extraction_note}\n\n"
        "请严格基于下面提供的论文文本、共享 Prompt 结果和最近对话历史回答。"
        "如果资料不足，请明确指出不确定之处，不要编造论文细节。\n\n"
        f"## 论文正文\n{document_text}\n\n"
        "## 可复用的 Prompt 结果\n"
        f"{_format_prompt_context(prompt_contexts)}\n\n"
        "## 最近对话历史\n"
        f"{_format_history(history)}\n\n"
        "## 当前问题\n"
        f"{question.strip()}\n\n"
        "## 回答要求\n"
        "- 使用 Markdown 输出，优先用小标题、列表和短段落组织内容\n"
        "- 使用中文\n"
        "- 先直接回答问题，再补充依据或步骤\n"
        "- 如果问题涉及公式、推导、损失函数、概率表达式或符号定义，必须写出关键公式，不要只做口头描述\n"
        "- 行内公式使用 `$...$`，独立公式使用 `$$...$$`\n"
        "- 不要用 ``` 代码块包裹数学公式，只有在展示代码时才使用代码块\n"
        "- 写公式时尽量补一句符号含义，避免只堆公式\n"
        "- 如果需要展示代码、伪代码、配置、命令或接口示例，必须使用三反引号代码块，并标注语言，例如 ```python```、```bash```、```json```\n"
        "- 不要用普通段落假装代码，也不要把代码和数学公式混在同一个代码块里\n"
        "- 如果只是引用很短的标识符、函数名、变量名或命令，使用行内代码 ``...``\n"
        "- 如果适合，给出 2-4 条下一步建议\n"
    )


def _chat_request(
    document_path: Path,
    *,
    question: str,
    visibility: str,
    history: Sequence[dict[str, str]],
    prompt_contexts: Sequence[tuple[str, str]],
    arxiv_markdown_path: Path | None,
    model: str | None,
) -> tuple[str, str]:
    document_path = document_path.resolve()
    if not document_path.exists() or not document_path.is_file():
        raise FileNotFoundError(document_path)
    if document_path.suffix.lower() not in SUPPORTED_SUMMARY_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported file type for chat processing: {document_path.suffix}")
    prompt = build_chat_prompt(
        document_path,
        question=question,
        visibility=visibility,
        history=history,
        prompt_contexts=prompt_contexts,
        arxiv_markdown_path=arxiv_markdown_path,
    )
    config = load_config()
    return prompt, resolve_model(model, config)


def answer_question_about_document(
    document_path: Path,
    *,
    question: str,
    visibility: str,
    history: Sequence[dict[str, str]],
    prompt_contexts: Sequence[tuple[str, str]],
    arxiv_markdown_path: Path | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> str:
    prompt, resolved_model = _chat_request(
        document_path,
        question=question,
        visibility=visibility,
        history=history,
        prompt_contexts=prompt_contexts,
        arxiv_markdown_path=arxiv_markdown_path,
        model=model,
    )
    return get_llm_client().complete(
        [{"role": "user", "content": prompt}], model=resolved_model, session_id=session_id
    )


def stream_answer_question_about_document(
    document_path: Path,
    *,
    question: str,
    visibility: str,
    history: Sequence[dict[str, str]],
    prompt_contexts: Sequence[tuple[str, str]],
    arxiv_markdown_path: Path | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> Iterator[str]:
    prompt, resolved_model = _chat_request(
        document_path,
        question=question,
        visibility=visibility,
        history=history,
        prompt_contexts=prompt_contexts,
        arxiv_markdown_path=arxiv_markdown_path,
        model=model,
    )
    yield from get_llm_client().stream(
        [{"role": "user", "content": prompt}], model=resolved_model, session_id=session_id
    )
