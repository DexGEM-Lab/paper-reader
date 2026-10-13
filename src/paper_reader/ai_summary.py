from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import DEFAULT_LLM_MODEL, load_config
from .document_utils import UnsupportedDocumentError, extract_document_metadata, extract_document_text
from .llm import get_llm_client, resolve_model

DEFAULT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_USER_PROMPT = (
    "请用通俗、准确、结构化的中文解释下面提供的论文正文。\n\n"
    "重点回答：\n"
    "1. 这篇论文到底想解决什么问题\n"
    "2. 核心思想是什么，为什么这样做有效\n"
    "3. 具体实现方法是什么，按步骤讲清楚输入、关键模块、训练/推理流程\n"
    "4. 实验结果说明了什么\n"
    "5. 这篇论文的优点、局限和适用场景\n\n"
    "输出要求：\n"
    "- 标题\n"
    "- 一句话概括\n"
    "- 核心思想（通俗解释）\n"
    "- 具体实现方法（分步骤）\n"
    "- 实验结果怎么看\n"
    "- 优点与局限\n"
    "- 如果我要自己复现，最该先做什么\n"
    "- 全文使用标准 Markdown 组织内容，优先使用 `##` / `###` 小标题、列表和短段落\n"
    "- 如果涉及公式、损失函数、概率表达式、矩阵或符号定义：行内公式使用 `$...$`，独立公式使用 `$$...$$` 或 `\\[...\\]`\n"
    "- 不要把数学公式放进代码块；公式前后尽量补一句自然语言解释符号含义\n"
    "- 如果涉及算法、伪代码、配置、命令、接口或代码片段：必须使用三反引号代码块，并显式标注语言，例如 ```python```、```bash```、```json```\n"
    "- 不要用缩进冒充代码块；代码块前先用一句话说明这段代码或命令在做什么\n"
    "- 除非论文原文确实给出可直接复现的代码，否则不要编造实现细节；不确定时明确写“论文未说明”\n"
)
ProgressCallback = Callable[[int, str], None]
AbortCallback = Callable[[], bool]
SUPPORTED_SUMMARY_EXTENSIONS = {".pdf", ".doc", ".docx"}


class SafePromptValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_user_prompt(user_prompt: str, document_path: Path, *, source_markdown_path: Path | None = None) -> str:
    """Render legacy prompt placeholders without asking the remote model to read files."""
    values = SafePromptValues(
        document_name=document_path.name,
        document_path=document_path.name,
        document_dir="",
        document_stem=document_path.stem,
        document_suffix=document_path.suffix.lower(),
        arxiv_markdown_path=source_markdown_path.name if source_markdown_path else "",
    )
    return user_prompt.format_map(values).strip()


def truncate_context(text: str, *, max_chars: int) -> str:
    """Keep both abstract/front matter and conclusions/references within the LLM budget."""
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        normalized[:head_chars].rstrip()
        + "\n\n[... 为控制上下文长度，已省略中间正文 ...]\n\n"
        + normalized[-tail_chars:].lstrip()
    )


def extract_document_context(
    document_path: Path,
    *,
    source_markdown_path: Path | None = None,
    max_chars: int | None = None,
) -> tuple[str, str, str]:
    """Extract the actual text that an HTTP-only model needs to ground a response."""
    document_path = document_path.resolve()
    title = document_path.stem
    try:
        metadata = extract_document_metadata(document_path)
        title = str(metadata.get("title") or title).strip() or title
    except Exception:
        metadata = {"preview_text": ""}

    if source_markdown_path is not None:
        source_markdown_path = source_markdown_path.resolve()
        if not source_markdown_path.exists() or not source_markdown_path.is_file():
            raise FileNotFoundError(source_markdown_path)
        text = source_markdown_path.read_text(encoding="utf-8", errors="replace").strip()
        note = "以下是系统缓存的 arXiv Markdown 原文；回答必须以其内容为依据。"
    else:
        try:
            text = extract_document_text(document_path).strip()
            note = "以下正文由系统从论文文件中提取；回答必须以其内容为依据。"
        except UnsupportedDocumentError:
            text = str(metadata.get("preview_text") or "").strip()
            note = "该文件只能提取预览文本；回答时需说明信息可能不完整。"
        except Exception:
            text = str(metadata.get("preview_text") or "").strip()
            note = "系统提取全文时遇到问题；回答仅能依据以下可用文本，并需说明信息可能不完整。"

    budget = max_chars if max_chars is not None else load_config().llm_context_tokens
    text = truncate_context(text, max_chars=budget) if text else ""
    if not text:
        text = "(未能提取到可用正文。)"
    return title, text, note


def _document_prompt(
    document_path: Path,
    *,
    user_prompt: str,
    source_markdown_path: Path | None,
    max_chars: int,
) -> str:
    title, text, note = extract_document_context(
        document_path, source_markdown_path=source_markdown_path, max_chars=max_chars
    )
    rendered = render_user_prompt(user_prompt, document_path, source_markdown_path=source_markdown_path)
    return f"{rendered}\n\n## 文档标题\n{title}\n\n## 文档上下文\n{note}\n\n{text}"


def run_text_prompt(
    prompt_text: str,
    *,
    workdir: Path | None = None,
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
    should_abort: AbortCallback | None = None,
    session_id: str | None = None,
) -> str:
    """Run a text-only prompt through the configured LiteLLM gateway."""
    del workdir  # retained as a compatibility argument for the insight builders.
    if should_abort and should_abort():
        raise InterruptedError("Job interrupted before the LLM request.")
    config = load_config()
    resolved_model = resolve_model(model, config)
    if progress_callback:
        progress_callback(5, "正在通过 LiteLLM 网关请求模型。")
    result = get_llm_client(config).complete(
        [{"role": "user", "content": prompt_text}], model=resolved_model, session_id=session_id
    )
    if should_abort and should_abort():
        raise InterruptedError("Job interrupted during the LLM request.")
    if progress_callback:
        progress_callback(98, "模型已完成输出，正在写入结果。")
    return result


def run_prompt_on_document(
    document_path: Path,
    *,
    user_prompt: str,
    model: str | None = None,
    source_markdown_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    should_abort: AbortCallback | None = None,
    session_id: str | None = None,
) -> str:
    document_path = document_path.resolve()
    if not document_path.exists() or not document_path.is_file():
        raise FileNotFoundError(document_path)
    if document_path.suffix.lower() not in SUPPORTED_SUMMARY_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported file type for LLM processing: {document_path.suffix}")
    config = load_config()
    prompt = _document_prompt(
        document_path,
        user_prompt=user_prompt,
        source_markdown_path=source_markdown_path,
        max_chars=config.llm_context_tokens,
    )
    return run_text_prompt(
        prompt,
        model=model,
        progress_callback=progress_callback,
        should_abort=should_abort,
        session_id=session_id,
    )


def explain_document(document_path: Path, *, model: str | None = None) -> str:
    return run_prompt_on_document(document_path, user_prompt=DEFAULT_USER_PROMPT, model=model)
