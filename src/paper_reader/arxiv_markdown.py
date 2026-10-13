from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from .document_utils import extract_document_metadata
from .netutils import fetch_public_url

ARXIV_MARKDOWN_DIR_NAME = ".paper-reader-arxiv-markdown"
ARXIV_MARKDOWN_TIMEOUT_SECONDS = 30
MARKDOWN_NEW_BASE_URL = "https://markdown.new/https://arxiv.org/html/"

_MODERN_ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)", re.IGNORECASE)
_LEGACY_ARXIV_ID_RE = re.compile(r"([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.IGNORECASE)
_ARXIV_URL_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
_ARXIV_PREFIX_RE = re.compile(
    r"arxiv[:\s]+([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ArxivMarkdownInfo:
    arxiv_id: str
    markdown_path: Path
    metadata_path: Path
    source_url: str
    markdown_url: str
    fetched_at: str | None
    cached: bool


def normalize_arxiv_id(raw: str) -> str | None:
    text = raw.strip().rstrip(".,);]")
    if not text:
        return None

    for pattern in (_ARXIV_URL_ID_RE, _ARXIV_PREFIX_RE, _MODERN_ARXIV_ID_RE, _LEGACY_ARXIV_ID_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def base_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)


def choose_preferred_arxiv_id(*candidates: str | None) -> str | None:
    normalized = [candidate for candidate in (normalize_arxiv_id(item or "") for item in candidates) if candidate]
    if not normalized:
        return None

    versioned = [item for item in normalized if re.search(r"v\d+$", item, re.IGNORECASE)]
    if versioned:
        return versioned[0]
    return normalized[0]


def infer_arxiv_id_from_path(document_path: Path) -> str | None:
    candidates = [
        document_path.name,
        document_path.stem,
        document_path.as_posix(),
    ]
    try:
        meta = extract_document_metadata(document_path)
    except Exception:
        meta = {}
    candidates.extend(
        [
            str(meta.get("title") or ""),
            str(meta.get("preview_text") or "")[:8000],
        ]
    )
    for candidate in candidates:
        arxiv_id = normalize_arxiv_id(candidate)
        if arxiv_id:
            return arxiv_id
    return None


def markdown_new_url_for(arxiv_id: str) -> str:
    return f"{MARKDOWN_NEW_BASE_URL}{arxiv_id}"


def fetch_arxiv_markdown(arxiv_id: str, *, timeout: int = ARXIV_MARKDOWN_TIMEOUT_SECONDS) -> str:
    url = markdown_new_url_for(arxiv_id)
    response = fetch_public_url(
        url,
        accept="text/markdown,text/plain;q=0.9,*/*;q=0.1",
        timeout=timeout,
    )
    text = response.body.decode("utf-8", "replace").strip()
    if not text:
        raise RuntimeError(f"无法获取 arXiv Markdown：{arxiv_id}")
    return text


def metadata_payload(*, arxiv_id: str, markdown_text: str) -> dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "markdown_url": markdown_new_url_for(arxiv_id),
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
        "char_count": len(markdown_text),
    }


def write_markdown_cache(markdown_path: Path, metadata_path: Path, *, arxiv_id: str, markdown_text: str) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    metadata = metadata_payload(arxiv_id=arxiv_id, markdown_text=markdown_text)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

