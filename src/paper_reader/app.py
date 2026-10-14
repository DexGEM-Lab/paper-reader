from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import queue
import re
import secrets
import shutil
import string
import tempfile
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, abort, flash, redirect, render_template, request, send_file, send_from_directory, session, stream_with_context, url_for
from markupsafe import Markup
from werkzeug.utils import secure_filename

from .action_queue import ActionRecord, ActionTaskQueue
from .ai_summary import DEFAULT_MODEL, DEFAULT_USER_PROMPT, run_prompt_on_document
from .arxiv_markdown import (
    ARXIV_MARKDOWN_DIR_NAME,
    ArxivMarkdownInfo,
    base_arxiv_id,
    choose_preferred_arxiv_id,
    fetch_arxiv_markdown,
    infer_arxiv_id_from_path,
    markdown_new_url_for,
    normalize_arxiv_id,
    write_markdown_cache,
)
from .chat_ai import stream_answer_question_about_document
from .chat_queue import PaperChatQueue
from .config import PaperReaderConfig, load_config, load_env_file_values, normalize_public_base_url
from .document_utils import ALLOWED_EXTENSIONS, LIBRARY_EXTENSIONS, extract_document_metadata
from .insights_history import HistoricalInsightsStore, extract_digest
from .llm import get_llm_client, resolve_model
from .insights_momentum import MomentumInsightsStore
from .insights_opportunity import OpportunityInsightsStore
from .markdown_render import render_markdown
from .netutils import fetch_public_url
from .offline_package import build_offline_manifest, manifest_json as build_manifest_json, offline_prompt_arcname, offline_source_arcname
from .prompt_manager import DEFAULT_PROMPT_SLUG, PromptDefinition, PromptStore, parse_checkbox
from .screening import ScreeningPacket, screen_packets
from .settings import SettingsStore
from .source_archive import day_paper_map, load_source_day, load_source_days, local_pdf_path_for
from .task_queue import PaperJobQueue
from .team_store import TeamStore, TeamUser, flatten_comments, slugify_text
from .web_resources import ImportTarget, WebResourceStore, classify_import_target

CACHE_FILE_NAME = ".paper_reader_index.json"
DONE_INDEX_FILE_NAME = ".paper_reader_done_index.json"
MANUAL_DATE_FILE_NAME = ".paper-reader-manual-dates.json"
SUMMARY_DIR_NAME = ".paper-reader-ai"
DONE_DIR_NAME = "DONE"
AVATAR_DIR_NAME = ".paper-reader-avatars"
UPLOAD_STAGING_DIR_NAME = ".paper-reader-upload-staging"
ALLOWED_AVATAR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DEFAULT_BATCH_PANEL_PAGE_SIZE = 50
MAX_LOGIN_FAILURES = 3
LOGIN_LOCK_SECONDS = 5 * 60
REMOTE_PDF_DOWNLOAD_TIMEOUT = 30
REMOTE_PDF_DOWNLOAD_ATTEMPTS = 3
FEISHU_LOGIN_REDIRECT_PATH = "/auth/feishu/callback"
FEISHU_AUTH_PAGE_URL = "https://open.feishu.cn/open-apis/authen/v1/index"
FEISHU_APP_ACCESS_TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
FEISHU_USER_ACCESS_TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v1/access_token"
FEISHU_USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"
FEISHU_CHAT_MEMBERSHIP_PATH = "/open-apis/im/v1/chats/{chat_id}/members/is_in_chat"
FEISHU_USERNAME_PREFIX = "feishu-"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeishuOAuthConfig:
    app_id: str
    app_secret: str
    base_url: str
    api_base_url: str
    redirect_path: str = FEISHU_LOGIN_REDIRECT_PATH

    @property
    def callback_url(self) -> str:
        return build_external_url(self.base_url, self.redirect_path)

    @property
    def auth_page_url(self) -> str:
        return f"{self.api_base_url}/open-apis/authen/v1/index"

    @property
    def app_access_token_url(self) -> str:
        return f"{self.api_base_url}/open-apis/auth/v3/app_access_token/internal"

    @property
    def user_access_token_url(self) -> str:
        return f"{self.api_base_url}/open-apis/authen/v1/access_token"

    @property
    def user_info_url(self) -> str:
        return f"{self.api_base_url}/open-apis/authen/v1/user_info"

    def chat_membership_url(self, chat_id: str) -> str:
        return f"{self.api_base_url}{FEISHU_CHAT_MEMBERSHIP_PATH.format(chat_id=quote(chat_id, safe=''))}"


@dataclass
class PaperRecord:
    rel_path: str
    file_name: str
    folder: str
    extension: str
    title: str
    display_title: str
    preview_text: str
    extracted_date: str | None
    date_precision: str | None
    date_source: str | None
    sort_date: str | None
    file_size: int
    modified_at: str
    preview_kind: str
    prompt_result_count: int
    prompt_result_slugs: list[str]
    is_done: bool
    resource_kind: str = "paper"
    source_url: str | None = None
    source_domain: str | None = None
    fetch_status: str = "ok"
    fetched_at: str | None = None
    author: str | None = None
    site_name: str | None = None


@dataclass
class ScanResult:
    papers: list[PaperRecord]
    folders: list[str]


@dataclass
class LoginAttemptState:
    failed_count: int = 0
    locked_until: float = 0.0


class LoginGuard:
    def __init__(self) -> None:
        self._attempts: dict[str, LoginAttemptState] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def key_for_request(self, req: Any) -> str:
        forwarded = (req.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return forwarded or req.remote_addr or "local"

    def status_for(self, key: str) -> dict[str, Any]:
        with self._lock:
            state = self._attempts.get(key)
            if state is None:
                return {"failed_count": 0, "locked": False, "remaining_seconds": 0}

            now = self._now()
            if state.locked_until and state.locked_until > now:
                remaining = int(state.locked_until - now)
                return {"failed_count": state.failed_count, "locked": True, "remaining_seconds": max(1, remaining)}

            if state.locked_until and state.locked_until <= now:
                state.locked_until = 0.0
                state.failed_count = 0
                self._attempts.pop(key, None)
            return {"failed_count": 0, "locked": False, "remaining_seconds": 0}

    def register_failure(self, key: str) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            state = self._attempts.get(key)
            if state is None:
                state = LoginAttemptState()
                self._attempts[key] = state

            if state.locked_until and state.locked_until > now:
                remaining = int(state.locked_until - now)
                return {"locked": True, "remaining_seconds": max(1, remaining), "failed_count": state.failed_count}

            if state.locked_until and state.locked_until <= now:
                state.failed_count = 0
                state.locked_until = 0.0

            state.failed_count += 1
            if state.failed_count >= MAX_LOGIN_FAILURES:
                state.locked_until = now + LOGIN_LOCK_SECONDS
                return {"locked": True, "remaining_seconds": LOGIN_LOCK_SECONDS, "failed_count": state.failed_count}

            return {"locked": False, "remaining_seconds": 0, "failed_count": state.failed_count}

    def register_success(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


_TAG_OUTPUT_CODE_BLOCK_RE = re.compile(r"```(?:json|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SHARE_SUMMARY_HEADINGS = (
    "一句话概括",
    "一句话总结",
    "一段话概括",
    "摘要速览",
)


def _normalize_generated_tag(candidate: str) -> str:
    cleaned = re.sub(r"^[-*#\d.\s]+", "", candidate).strip().strip("`\"'")
    if " - " in cleaned:
        cleaned = cleaned.split(" - ", 1)[0].strip()
    if ": " in cleaned:
        cleaned = cleaned.split(": ", 1)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def parse_tag_generation_output(output: str) -> list[str]:
    text = output.strip()
    if not text:
        return []

    candidates: list[str] = []
    payload_candidates = [text]
    fenced = _TAG_OUTPUT_CODE_BLOCK_RE.findall(text)
    payload_candidates.extend(block.strip() for block in fenced if block.strip())

    for payload in payload_candidates:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("tags", [])
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    candidates.append(item["name"])
            break

    if not candidates:
        normalized_text = text.replace("\r", "\n")
        for raw_line in normalized_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                line = line.strip("[]")
            if "," in line:
                parts = [part for part in line.split(",") if part.strip()]
                if len(parts) > 1:
                    candidates.extend(parts)
                    continue
            candidates.append(line)

    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_generated_tag(candidate)
        slug = slugify_text(normalized)
        if not slug or slug in seen:
            continue
        tags.append(normalized)
        seen.add(slug)
        if len(tags) >= 8:
            break
    return tags


def _normalize_share_summary_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).strip()
    return cleaned.strip("：:- ").strip()


def extract_share_summary(content: str) -> str:
    body = content.strip()
    marker = "\n---\n"
    if marker in body:
        body = body.split(marker, 1)[1].strip()
    if not body:
        return ""

    lines = body.splitlines()
    capture = False
    fragments: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            if capture and fragments:
                break
            continue

        normalized = re.sub(r"^[#>*\-\s]+", "", stripped).strip().strip("*_` ")
        normalized = _normalize_share_summary_line(normalized)
        heading = next((item for item in _SHARE_SUMMARY_HEADINGS if normalized.startswith(item)), None)
        if heading is not None:
            remainder = _normalize_share_summary_line(normalized[len(heading) :])
            if remainder:
                return remainder[:360]
            capture = True
            fragments = []
            continue

        if capture:
            if stripped.startswith("#") or (stripped.startswith("**") and stripped.endswith("**")):
                break
            fragments.append(_normalize_share_summary_line(stripped.lstrip("-* ")))

    if fragments:
        return _normalize_share_summary_line(" ".join(fragment for fragment in fragments if fragment))[:360]
    return _normalize_share_summary_line(extract_digest(body))[:360]


def normalize_import_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return ""

    lowered = target.lower()
    arxiv_prefix_match = re.fullmatch(
        r"arxiv[:\s]+([a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)",
        lowered,
        flags=re.IGNORECASE,
    )
    if arxiv_prefix_match:
        return arxiv_prefix_match.group(1)

    if lowered.startswith("arxiv.org/") or lowered.startswith("www.arxiv.org/"):
        return f"https://{target.lstrip('/')}"
    if lowered.startswith("abs/") or lowered.startswith("pdf/") or lowered.startswith("html/"):
        return f"https://arxiv.org/{target.lstrip('/')}"

    return target


def parse_import_targets(raw_target: str) -> list[str]:
    target = raw_target.strip()
    if not target:
        return []

    try:
        parsed = json.loads(target)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        normalized = normalize_import_target(parsed)
        return [normalized] if normalized else []
    if isinstance(parsed, list):
        items: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                raise ValueError("JSON 导入列表里的每一项都必须是字符串。")
            normalized = normalize_import_target(item)
            if normalized:
                items.append(normalized)
        return items

    single_target = normalize_import_target(target)
    if single_target and re.fullmatch(r"[^\s,]+", target):
        return [single_target]

    token_pattern = re.compile(
        r"""
        https?://[^\s,]+
        |(?:www\.)?arxiv\.org/[^\s,]+
        |(?:abs|pdf|html)/[^\s,]+
        |arxiv[:\s]+(?:[a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)
        |(?:[a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    raw_items = [match.group(0) for match in token_pattern.finditer(target)]
    if not raw_items:
        return [single_target] if single_target else []

    remainder = token_pattern.sub("", target)
    if remainder.strip().strip(","):
        split_candidates = [part.strip() for part in re.split(r"[\s,]+", target) if part.strip()]
        if len(split_candidates) <= 1:
            return [single_target] if single_target else split_candidates
        raw_items = split_candidates

    return [normalized for item in raw_items if (normalized := normalize_import_target(item))]


def arxiv_version_number(arxiv_id: str | None) -> int:
    normalized = normalize_arxiv_id(arxiv_id or "")
    if normalized is None:
        return 0
    match = re.search(r"v(\d+)$", normalized, re.IGNORECASE)
    return int(match.group(1)) if match is not None else 0


class PaperLibrary:
    def __init__(
        self,
        root: Path,
        prompt_store: PromptStore,
        team_store: Any | None = None,
        *,
        page_fetch_max_bytes: int = 25 * 1024 * 1024,
        document_fetch_max_bytes: int = 100 * 1024 * 1024,
        fetch_max_redirects: int = 10,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.root / CACHE_FILE_NAME
        self.done_index_path = self.root / DONE_INDEX_FILE_NAME
        self.manual_date_path = self.root / MANUAL_DATE_FILE_NAME
        self.summary_root = self.root / SUMMARY_DIR_NAME
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.arxiv_markdown_root = self.root / ARXIV_MARKDOWN_DIR_NAME
        self.arxiv_markdown_root.mkdir(parents=True, exist_ok=True)
        self.web_resources = WebResourceStore(
            self.root,
            fetch_max_bytes=page_fetch_max_bytes,
            document_max_bytes=document_fetch_max_bytes,
            max_redirects=fetch_max_redirects,
        )
        self.prompt_store = prompt_store
        self.team_store = team_store
        self._hash_cache: dict[str, tuple[float, int, str]] = {}
        self._scan_lock = threading.RLock()
        self._manual_date_lock = threading.RLock()
        self._scan_cache: dict[tuple[bool, bool], ScanResult] = {}
        self._migrate_manual_date_json_to_sqlite()

    def invalidate_scan_cache(self) -> None:
        with self._scan_lock:
            self._scan_cache.clear()

    def scan(self, *, force: bool = False, lightweight: bool = False, include_done: bool = False) -> ScanResult:
        with self._scan_lock:
            cache_key = (lightweight, include_done)
            if cache_key in self._scan_cache and not force:
                return self._scan_cache[cache_key]

            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            if force or not self.cache_path.exists():
                papers = self.rebuild_active_index(lightweight=lightweight)
            else:
                papers = self.load_active_index(active_prompt_slugs)

            folders = {""}
            for paper in papers:
                folders.add(paper.folder)

            if include_done:
                papers.extend(self.load_done_index(active_prompt_slugs))
                for paper in papers:
                    folders.add(paper.folder)

            result = ScanResult(papers=papers, folders=sorted(folders))
            self._scan_cache[cache_key] = result
            return result

    def iter_documents(self, *, include_done: bool = False) -> list[Path]:
        documents: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name == CACHE_FILE_NAME:
                continue
            if path.name == self.prompt_store.store_path.name:
                continue
            if SUMMARY_DIR_NAME in path.parts:
                continue
            if ARXIV_MARKDOWN_DIR_NAME in path.parts:
                continue
            if UPLOAD_STAGING_DIR_NAME in path.parts:
                continue
            if not include_done and DONE_DIR_NAME in path.parts:
                continue
            relative_parts = path.relative_to(self.root).parts
            if any(part.startswith(".") for part in relative_parts[:-1]):
                continue
            if path.suffix.lower() not in LIBRARY_EXTENSIONS:
                continue
            documents.append(path)
        return documents

    def iter_done_documents(self) -> list[Path]:
        done_root = self.root / DONE_DIR_NAME
        if not done_root.exists():
            return []

        documents: list[Path] = []
        for path in sorted(done_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LIBRARY_EXTENSIONS:
                continue
            documents.append(path)
        return documents

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_manual_date_json_payload(self) -> dict[str, Any]:
        if not self.manual_date_path.exists():
            return {}
        try:
            payload = json.loads(self.manual_date_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_manual_date_payload(self) -> dict[str, Any]:
        payload = self._load_manual_date_json_payload()
        if self.team_store is not None:
            if payload:
                try:
                    self.team_store.migrate_manual_date_overrides(payload)
                except Exception:
                    pass
            try:
                db_payload = self.team_store.all_manual_date_overrides()
            except Exception:
                db_payload = {}
            payload.update(db_payload)
        return payload

    def _migrate_manual_date_json_to_sqlite(self) -> None:
        if self.team_store is None or not self.manual_date_path.exists():
            return
        payload = self._load_manual_date_json_payload()
        if not payload:
            return
        try:
            migrated = self.team_store.migrate_manual_date_overrides(payload)
        except Exception:
            return
        if migrated:
            self.invalidate_scan_cache()

    def _write_manual_date_payload(self, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        temp_path = self.manual_date_path.with_name(f"{self.manual_date_path.name}.tmp")
        temp_path.write_text(serialized, encoding="utf-8", errors="backslashreplace")
        temp_path.replace(self.manual_date_path)

    def _manual_date_override_from_payload(
        self,
        payload: dict[str, Any] | None,
        rel_path: str,
    ) -> dict[str, str | None] | None:
        if not payload:
            return None
        raw = payload.get(rel_path)
        if not isinstance(raw, dict):
            return None
        display_date = str(raw.get("display_date") or "").strip()
        sort_date = str(raw.get("sort_date") or "").strip()
        precision = str(raw.get("precision") or "month").strip() or "month"
        if not display_date or not sort_date:
            return None
        return {
            "display_date": display_date,
            "precision": precision,
            "source": "manual",
            "sort_date": sort_date,
        }

    def _record_from_index_item(
        self,
        item: dict[str, Any],
        active_prompt_slugs: list[str],
        *,
        manual_date_overrides: dict[str, Any] | None = None,
    ) -> PaperRecord | None:
        try:
            record = PaperRecord(**item)
        except TypeError:
            return None

        visible_slugs = set(active_prompt_slugs)
        stored_slugs = [slug for slug in record.prompt_result_slugs if isinstance(slug, str)]
        active_result_slugs = [slug for slug in stored_slugs if slug in visible_slugs]
        manual_date = self._manual_date_override_from_payload(manual_date_overrides, record.rel_path)
        return PaperRecord(
            rel_path=record.rel_path,
            file_name=record.file_name,
            folder=record.folder,
            extension=record.extension,
            title=record.title,
            display_title=record.display_title,
            preview_text=record.preview_text,
            extracted_date=(manual_date["display_date"] if manual_date is not None else record.extracted_date),
            date_precision=(manual_date["precision"] if manual_date is not None else record.date_precision),
            date_source=(manual_date["source"] if manual_date is not None else record.date_source),
            sort_date=(manual_date["sort_date"] if manual_date is not None else record.sort_date),
            file_size=record.file_size,
            modified_at=record.modified_at,
            preview_kind=record.preview_kind,
            prompt_result_count=len(active_result_slugs),
            prompt_result_slugs=stored_slugs,
            is_done=record.is_done,
            resource_kind=record.resource_kind,
            source_url=record.source_url,
            source_domain=record.source_domain,
            fetch_status=record.fetch_status,
            fetched_at=record.fetched_at,
            author=record.author,
            site_name=record.site_name,
        )

    def _load_active_index_payload(self) -> dict[str, Any]:
        payload = self._load_cache()
        if "records" in payload and isinstance(payload.get("records"), list):
            return payload

        # Backward compatibility for the old rel_path -> {signature, record} cache format.
        records: list[dict[str, Any]] = []
        for item in payload.values():
            if isinstance(item, dict) and isinstance(item.get("record"), dict):
                records.append(item["record"])
        if not records:
            return {}
        return {"records": records}

    def _write_active_index_payload(self, payload: dict[str, Any]) -> None:
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            errors="backslashreplace",
        )

    def load_active_index(self, active_prompt_slugs: list[str]) -> list[PaperRecord]:
        payload = self._load_active_index_payload()
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        records: list[PaperRecord] = []
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            record = self._record_from_index_item(item, active_prompt_slugs, manual_date_overrides=manual_date_overrides)
            if record is None or record.is_done:
                continue
            records.append(record)
        return records

    def rebuild_active_index(self, *, lightweight: bool = False) -> list[PaperRecord]:
        active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        records = [
            self._build_record(path, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides)
            for path in self.iter_documents(include_done=False)
        ]
        self._write_active_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(records),
                "records": [asdict(record) for record in records],
            }
        )
        self.invalidate_scan_cache()
        return records

    def _update_active_index_entry(self, rel_path: str, *, lightweight: bool = False) -> None:
        payload = self._load_active_index_payload()
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        if not payload.get("records") and not self.cache_path.exists():
            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            payload = {
                "records": [
                    asdict(self._build_record(path, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides))
                    for path in self.iter_documents(include_done=False)
                ]
            }
        items = [item for item in payload.get("records", []) if isinstance(item, dict) and item.get("rel_path") != rel_path]
        if rel_path and not self.is_done_rel_path(rel_path):
            try:
                absolute = self.resolve_relative_path(rel_path)
            except ValueError:
                absolute = None
            if absolute is not None and absolute.exists() and absolute.is_file():
                active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
                items.append(asdict(self._build_record(absolute, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides)))
        self._write_active_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(items),
                "records": items,
            }
        )
        self.invalidate_scan_cache()

    def _load_done_index_payload(self) -> dict[str, Any]:
        if not self.done_index_path.exists():
            return {}
        try:
            payload = json.loads(self.done_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_done_index_payload(self, payload: dict[str, Any]) -> None:
        self.done_index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            errors="backslashreplace",
        )

    def load_done_index(self, active_prompt_slugs: list[str]) -> list[PaperRecord]:
        payload = self._load_done_index_payload()
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        records: list[PaperRecord] = []
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            record = self._record_from_index_item(item, active_prompt_slugs, manual_date_overrides=manual_date_overrides)
            if record is None:
                continue
            if not self.is_done_rel_path(record.rel_path):
                continue
            records.append(record)
        return records

    def rebuild_done_index(self, *, lightweight: bool = True) -> list[PaperRecord]:
        active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        records = [
            self._build_record(path, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides)
            for path in self.iter_done_documents()
        ]
        self._write_done_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(records),
                "records": [asdict(record) for record in records],
            }
        )
        self.invalidate_scan_cache()
        return records

    def _update_done_index_entry(self, rel_path: str, *, lightweight: bool = False) -> None:
        payload = self._load_done_index_payload()
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        if not payload.get("records") and not self.done_index_path.exists():
            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            payload = {
                "records": [
                    asdict(self._build_record(path, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides))
                    for path in self.iter_done_documents()
                ]
            }
        items = [item for item in payload.get("records", []) if isinstance(item, dict) and item.get("rel_path") != rel_path]
        if self.is_done_rel_path(rel_path):
            try:
                absolute = self.resolve_relative_path(rel_path)
            except ValueError:
                absolute = None
            if absolute is not None and absolute.exists() and absolute.is_file():
                active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
                items.append(asdict(self._build_record(absolute, active_prompt_slugs, lightweight=lightweight, manual_date_overrides=manual_date_overrides)))
        self._write_done_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(items),
                "records": items,
            }
        )
        self.invalidate_scan_cache()

    def _build_record(
        self,
        path: Path,
        active_prompt_slugs: list[str],
        *,
        lightweight: bool = False,
        manual_date_overrides: dict[str, Any] | None = None,
    ) -> PaperRecord:
        rel_path = path.relative_to(self.root).as_posix()
        folder = path.relative_to(self.root).parent.as_posix()
        if folder == ".":
            folder = ""

        if lightweight:
            meta = {"title": path.stem, "preview_text": "", "full_text": ""}
        else:
            try:
                meta = extract_document_metadata(path)
            except Exception:
                meta = {"title": path.stem, "preview_text": "", "full_text": ""}

        title = self._safe_text(meta.get("title") or path.stem)
        preview_text = self._safe_text(meta.get("preview_text") or "")
        resource_metadata = self.web_resources.metadata_for_rel_path(rel_path) or {}
        if resource_metadata.get("title"):
            title = self._safe_text(str(resource_metadata["title"]))
        date_info = self._extract_date(title, preview_text, path.name)
        resource_sort_date = str(resource_metadata.get("sort_date") or "").strip()
        resource_published_date = str(resource_metadata.get("published_date") or "").strip()
        if resource_sort_date:
            date_info = {
                "display_date": resource_published_date or resource_sort_date,
                "precision": "day",
                "source": "web_metadata",
                "sort_date": resource_sort_date,
            }
        manual_date = self._manual_date_override_from_payload(manual_date_overrides, rel_path)
        if manual_date is not None:
            date_info = manual_date
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        all_prompt_result_slugs = self.list_existing_prompt_slugs(rel_path)
        visible_prompt_result_slugs = [slug for slug in all_prompt_result_slugs if slug in set(active_prompt_slugs)]

        return PaperRecord(
            rel_path=self._safe_text(rel_path),
            file_name=self._safe_text(path.name),
            folder=self._safe_text(folder),
            extension=path.suffix.lower(),
            title=title,
            display_title=title if title else path.stem,
            preview_text=preview_text,
            extracted_date=date_info["display_date"],
            date_precision=date_info["precision"],
            date_source=date_info["source"],
            sort_date=date_info["sort_date"],
            file_size=path.stat().st_size,
            modified_at=modified.isoformat(timespec="seconds"),
            preview_kind=self.preview_kind(path),
            prompt_result_count=len(visible_prompt_result_slugs),
            prompt_result_slugs=all_prompt_result_slugs,
            is_done=self.is_done_rel_path(rel_path),
            resource_kind=str(resource_metadata.get("resource_kind") or "paper"),
            source_url=(str(resource_metadata.get("canonical_url")) if resource_metadata.get("canonical_url") else None),
            source_domain=(str(resource_metadata.get("source_domain")) if resource_metadata.get("source_domain") else None),
            fetch_status=str(resource_metadata.get("fetch_status") or "ok"),
            fetched_at=(str(resource_metadata.get("fetched_at")) if resource_metadata.get("fetched_at") else None),
            author=(str(resource_metadata.get("author")) if resource_metadata.get("author") else None),
            site_name=(str(resource_metadata.get("site_name")) if resource_metadata.get("site_name") else None),
        )

    def _safe_text(self, value: str) -> str:
        if not value:
            return ""
        return value.encode("utf-8", "backslashreplace").decode("utf-8")

    def build_record_for_rel_path(self, rel_path: str, active_prompt_slugs: list[str]) -> PaperRecord:
        with self._manual_date_lock:
            manual_date_overrides = self._load_manual_date_payload()
        return self._build_record(
            self.resolve_relative_path(rel_path),
            active_prompt_slugs,
            manual_date_overrides=manual_date_overrides,
        )

    def set_manual_date(self, rel_path: str, precision: str, date_value: str, *, updated_by_user_id: int | None = None) -> None:
        if self.team_store is None:
            raise RuntimeError("Manual date storage requires team store.")
        target = self.resolve_relative_path(rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(rel_path)
        normalized_rel_path = target.relative_to(self.root).as_posix()
        precision = precision.strip().lower()
        date_value = date_value.strip()
        try:
            if precision == "year":
                parsed = datetime.strptime(date_value, "%Y")
                display_date = parsed.strftime("%Y")
                sort_date = parsed.replace(month=1, day=1).date().isoformat()
            elif precision == "month":
                parsed = datetime.strptime(date_value, "%Y-%m")
                display_date = parsed.strftime("%Y-%m")
                sort_date = parsed.replace(day=1).date().isoformat()
            elif precision == "day":
                parsed = datetime.strptime(date_value, "%Y-%m-%d")
                display_date = parsed.strftime("%Y-%m-%d")
                sort_date = parsed.date().isoformat()
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError("请选择合法的论文日期。") from exc

        active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
        record = self._build_record(target, active_prompt_slugs, manual_date_overrides={})
        self.team_store.sync_papers([record])
        self.team_store.set_manual_date_override(
            normalized_rel_path,
            display_date=display_date,
            precision=precision,
            sort_date=sort_date,
            updated_by_user_id=updated_by_user_id,
        )
        self.invalidate_scan_cache()

    def set_manual_month_date(self, rel_path: str, year_month: str) -> None:
        self.set_manual_date(rel_path, "month", year_month)

    def _move_manual_date(self, old_rel_path: str, new_rel_path: str) -> None:
        if self.team_store is not None:
            return
        with self._manual_date_lock:
            payload = self._load_manual_date_json_payload()
            entry = payload.pop(old_rel_path, None)
            if entry is None:
                return
            payload[new_rel_path] = entry
            self._write_manual_date_payload(payload)
        self.invalidate_scan_cache()

    def _delete_manual_date(self, rel_path: str) -> None:
        if self.team_store is not None:
            return
        with self._manual_date_lock:
            payload = self._load_manual_date_json_payload()
            if rel_path not in payload:
                return
            payload.pop(rel_path, None)
            self._write_manual_date_payload(payload)
        self.invalidate_scan_cache()

    def _prompt_state(self, rel_path: str, active_prompt_slugs: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            for path in sorted(result_dir.glob("*.md")):
                if path.stem not in active_prompt_slugs:
                    continue
                stat = path.stat()
                entries.append({"slug": path.stem, "mtime": stat.st_mtime, "size": stat.st_size})

        legacy_path = self.legacy_summary_path_for(rel_path)
        if DEFAULT_PROMPT_SLUG in active_prompt_slugs and legacy_path.exists():
            stat = legacy_path.stat()
            entries.append({"slug": DEFAULT_PROMPT_SLUG, "mtime": stat.st_mtime, "size": stat.st_size, "legacy": True})
        return entries

    def preview_kind(self, path: Path) -> str:
        return {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
            ".md": "web" if self.web_resources.metadata_for_rel_path(path.relative_to(self.root).as_posix()) else "markdown",
        }.get(path.suffix.lower(), "download")

    def _extract_date(self, title: str, preview_text: str, file_name: str) -> dict[str, str | None]:
        candidate_text = "\n".join([title, preview_text[:4000], file_name])

        patterns = [
            (r"Submitted on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %b %Y", "submitted_on"),
            (r"Submitted on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %B %Y", "submitted_on"),
            (r"\[v\d+\]\s+\w{3},\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %b %Y", "arxiv_version"),
            (r"\[v\d+\]\s+\w{3},\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %B %Y", "arxiv_version"),
            (r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", "%B %d, %Y", "text_date"),
        ]
        for pattern, fmt, source in patterns:
            match = re.search(pattern, candidate_text, re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1)
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            return {
                "display_date": parsed.strftime("%Y-%m-%d"),
                "precision": "day",
                "source": source,
                "sort_date": parsed.date().isoformat(),
            }

        modern_id = re.search(r"(?<!\d)(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?(?!\d)", candidate_text)
        if modern_id:
            year = int(modern_id.group(1))
            year += 2000 if year < 90 else 1900
            month = int(modern_id.group(2))
            try:
                parsed = datetime(year, month, 1)
            except ValueError:
                return {"display_date": None, "precision": None, "source": None, "sort_date": None}
            return {
                "display_date": parsed.strftime("%Y-%m"),
                "precision": "month",
                "source": "arxiv_id",
                "sort_date": parsed.date().isoformat(),
            }

        legacy_id = re.search(r"[a-z\-]+\/(\d{2})(\d{2})\d{3,4}", candidate_text, re.IGNORECASE)
        if legacy_id:
            year = int(legacy_id.group(1))
            year += 2000 if year < 90 else 1900
            month = int(legacy_id.group(2))
            try:
                parsed = datetime(year, month, 1)
            except ValueError:
                return {"display_date": None, "precision": None, "source": None, "sort_date": None}
            return {
                "display_date": parsed.strftime("%Y-%m"),
                "precision": "month",
                "source": "legacy_arxiv_id",
                "sort_date": parsed.date().isoformat(),
            }

        return {"display_date": None, "precision": None, "source": None, "sort_date": None}

    def resolve_relative_path(self, rel_path: str) -> Path:
        rel_path = unquote(rel_path).split("#", 1)[0].strip("/")
        candidate = (self.root / rel_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Path escapes library root")
        return candidate

    def make_unique_destination(self, folder: str, original_name: str) -> Path:
        folder_path = self.resolve_relative_path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(original_name)
        if not filename:
            raise ValueError("Invalid file name")
        candidate = folder_path / filename
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = folder_path / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    @staticmethod
    def _ensure_readable_file(path: Path) -> None:
        path.chmod(path.stat().st_mode | 0o644)

    def import_external_file(self, source_path: Path, target_folder: str, *, preferred_name: str | None = None) -> dict[str, Any]:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(str(source_path))

        source_name = preferred_name or source_path.name
        suffix = Path(source_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {source_name}")

        file_size = source_path.stat().st_size
        file_hash = sha256_for_path(source_path)
        duplicate_rel_path = self.find_duplicate_by_hash(file_size, file_hash)
        if duplicate_rel_path:
            return {
                "status": "duplicate",
                "message": f"检测到重复文件，已跳过：{Path(duplicate_rel_path).name}",
                "saved_rel_path": None,
                "duplicate_rel_path": duplicate_rel_path,
            }

        destination = self.make_unique_destination(target_folder, source_name)
        shutil.copy2(source_path, destination)
        self._ensure_readable_file(destination)
        rel_path = destination.relative_to(self.root).as_posix()
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)
        return {
            "status": "saved",
            "message": f"导入成功：{destination.name}",
            "saved_rel_path": rel_path,
            "duplicate_rel_path": None,
        }

    def clear_derived_artifacts(self, rel_path: str) -> None:
        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            shutil.rmtree(result_dir)

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists():
            legacy_path.unlink()

        arxiv_markdown_path = self.arxiv_markdown_path_for(rel_path)
        if arxiv_markdown_path.exists():
            arxiv_markdown_path.unlink()

        arxiv_metadata_path = self.arxiv_markdown_metadata_path_for(rel_path)
        if arxiv_metadata_path.exists():
            arxiv_metadata_path.unlink()

    def replace_existing_file(self, rel_path: str, source_path: Path, *, clear_derived: bool = True) -> dict[str, Any]:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(str(source_path))

        target = self.resolve_relative_path(rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(rel_path)
        if source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {source_path.name}")

        source_size = source_path.stat().st_size
        source_hash = sha256_for_path(source_path)
        target_size = target.stat().st_size
        target_hash = self.hash_for_path(target)
        if source_size == target_size and source_hash == target_hash:
            return {
                "status": "existing",
                "message": f"文件内容未变化：{target.name}",
                "saved_rel_path": rel_path,
                "duplicate_rel_path": None,
            }

        shutil.copy2(source_path, target)
        self._ensure_readable_file(target)
        self._hash_cache.pop(rel_path, None)
        if clear_derived:
            self.clear_derived_artifacts(rel_path)
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)
        return {
            "status": "saved",
            "message": f"已覆盖更新：{target.name}",
            "saved_rel_path": rel_path,
            "duplicate_rel_path": None,
        }

    def is_done_rel_path(self, rel_path: str) -> bool:
        parts = Path(rel_path).parts
        return bool(parts) and parts[0] == DONE_DIR_NAME

    def done_destination_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        base_folder = self.root / DONE_DIR_NAME / rel.parent
        base_folder.mkdir(parents=True, exist_ok=True)
        candidate = base_folder / rel.name
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = base_folder / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def restore_destination_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        if not self.is_done_rel_path(rel_path):
            raise ValueError("Paper is not in DONE folder")
        original_rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
        folder_path = self.root / original_rel.parent
        folder_path.mkdir(parents=True, exist_ok=True)
        candidate = folder_path / original_rel.name
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = folder_path / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def find_duplicate_by_hash(self, file_size: int, file_hash: str) -> str | None:
        for path in self.iter_documents():
            try:
                if path.stat().st_size != file_size:
                    continue
            except FileNotFoundError:
                continue
            if self.hash_for_path(path) == file_hash:
                return path.relative_to(self.root).as_posix()
        return None

    def hash_for_path(self, path: Path) -> str:
        rel_path = path.relative_to(self.root).as_posix()
        stat = path.stat()
        cached = self._hash_cache.get(rel_path)
        signature = (stat.st_mtime, stat.st_size)
        if cached and cached[:2] == signature:
            return cached[2]
        digest = sha256_for_path(path)
        self._hash_cache[rel_path] = (stat.st_mtime, stat.st_size, digest)
        return digest

    def _move_prompt_results(self, old_rel_path: str, new_rel_path: str) -> None:
        old_result_dir = self.prompt_result_dir_for(old_rel_path)
        new_result_dir = self.prompt_result_dir_for(new_rel_path)
        if old_result_dir.exists():
            new_result_dir.parent.mkdir(parents=True, exist_ok=True)
            if new_result_dir.exists():
                shutil.rmtree(new_result_dir)
            shutil.move(str(old_result_dir), str(new_result_dir))

        old_legacy = self.legacy_summary_path_for(old_rel_path)
        new_legacy = self.legacy_summary_path_for(new_rel_path)
        if old_legacy.exists():
            new_legacy.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_legacy), str(new_legacy))

        old_arxiv_markdown = self.arxiv_markdown_path_for(old_rel_path)
        new_arxiv_markdown = self.arxiv_markdown_path_for(new_rel_path)
        if old_arxiv_markdown.exists():
            new_arxiv_markdown.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_arxiv_markdown), str(new_arxiv_markdown))

        old_arxiv_meta = self.arxiv_markdown_metadata_path_for(old_rel_path)
        new_arxiv_meta = self.arxiv_markdown_metadata_path_for(new_rel_path)
        if old_arxiv_meta.exists():
            new_arxiv_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_arxiv_meta), str(new_arxiv_meta))

        old_path = self.resolve_relative_path(old_rel_path)
        new_path = self.resolve_relative_path(new_rel_path)
        old_web_meta = self.web_resources.metadata_path_for(old_path)
        new_web_meta = self.web_resources.metadata_path_for(new_path)
        if old_web_meta.exists():
            new_web_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_web_meta), str(new_web_meta))

    def create_folder(self, rel_folder: str) -> Path:
        rel_folder = rel_folder.strip().strip("/")
        if not rel_folder:
            raise ValueError("Folder name is required")
        destination = self.resolve_relative_path(rel_folder)
        destination.mkdir(parents=True, exist_ok=True)
        self.invalidate_scan_cache()
        return destination

    def rename_file(self, rel_path: str, new_name: str) -> str:
        source = self.resolve_relative_path(rel_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(rel_path)
        filename = secure_filename(new_name)
        if not filename:
            raise ValueError("Invalid target file name")
        if Path(filename).suffix.lower() not in LIBRARY_EXTENSIONS:
            raise ValueError("Only PDF / DOC / DOCX / Markdown files are allowed")
        destination = source.with_name(filename)
        if destination.exists() and destination != source:
            raise ValueError("Target file already exists")

        old_rel_path = source.relative_to(self.root).as_posix()
        source.rename(destination)
        new_rel_path = destination.relative_to(self.root).as_posix()
        self._move_prompt_results(old_rel_path, new_rel_path)
        self._move_manual_date(old_rel_path, new_rel_path)
        self._hash_cache.pop(old_rel_path, None)
        if self.is_done_rel_path(old_rel_path) or self.is_done_rel_path(new_rel_path):
            self._update_done_index_entry(old_rel_path)
            self._update_done_index_entry(new_rel_path)
        else:
            self._update_active_index_entry(old_rel_path)
            self._update_active_index_entry(new_rel_path)

        return new_rel_path

    def delete_file(self, rel_path: str) -> None:
        target = self.resolve_relative_path(rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(rel_path)
        web_metadata_path = self.web_resources.metadata_path_for(target)
        if web_metadata_path.exists():
            self.web_resources.delete_snapshot(rel_path)
        else:
            target.unlink()
        self._hash_cache.pop(rel_path, None)
        self._delete_manual_date(rel_path)
        self.clear_derived_artifacts(rel_path)
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)

    def toggle_done(self, rel_path: str) -> str:
        source = self.resolve_relative_path(rel_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(rel_path)

        if self.is_done_rel_path(rel_path):
            destination = self.restore_destination_for(rel_path)
        else:
            destination = self.done_destination_for(rel_path)

        old_rel_path = source.relative_to(self.root).as_posix()
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        new_rel_path = destination.relative_to(self.root).as_posix()
        self._move_prompt_results(old_rel_path, new_rel_path)
        self._move_manual_date(old_rel_path, new_rel_path)
        cached = self._hash_cache.pop(old_rel_path, None)
        if cached:
            self._hash_cache[new_rel_path] = cached
        self._update_active_index_entry(old_rel_path)
        self._update_active_index_entry(new_rel_path)
        self._update_done_index_entry(old_rel_path)
        self._update_done_index_entry(new_rel_path)
        return new_rel_path

    def legacy_summary_path_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        return self.summary_root / rel.parent / f"{rel.stem}.explained.zh.md"

    def prompt_result_dir_for(self, rel_path: str) -> Path:
        return self.summary_root / Path(rel_path)

    def prompt_result_path_for(self, rel_path: str, prompt_slug: str) -> Path:
        return self.prompt_result_dir_for(rel_path) / f"{prompt_slug}.md"

    def arxiv_markdown_path_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        return self.arxiv_markdown_root / rel.parent / f"{rel.name}.md"

    def arxiv_markdown_metadata_path_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        return self.arxiv_markdown_root / rel.parent / f"{rel.name}.json"

    def cached_arxiv_markdown_path(self, rel_path: str) -> Path | None:
        path = self.arxiv_markdown_path_for(rel_path)
        return path if path.exists() else None

    def arxiv_markdown_status_for_rel_path(self, rel_path: str) -> dict[str, Any]:
        markdown_path = self.arxiv_markdown_path_for(rel_path)
        metadata_path = self.arxiv_markdown_metadata_path_for(rel_path)
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except json.JSONDecodeError:
                metadata = {}

        cached_id = normalize_arxiv_id(str(metadata.get("arxiv_id") or ""))
        arxiv_id = cached_id or self.resolve_arxiv_id_for(rel_path)
        cached = markdown_path.exists()
        fetched_at = str(metadata.get("fetched_at") or "") or None
        return {
            "available": arxiv_id is not None,
            "arxiv_id": arxiv_id,
            "cached": cached,
            "source_url": (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
            "markdown_url": (markdown_new_url_for(arxiv_id) if arxiv_id else None),
            "markdown_rel_path": (markdown_path.relative_to(self.root).as_posix() if cached else None),
            "metadata_rel_path": (metadata_path.relative_to(self.root).as_posix() if metadata_path.exists() else None),
            "fetched_at": fetched_at,
            "char_count": int(metadata.get("char_count") or 0) if metadata.get("char_count") else None,
        }

    def _arxiv_source_ids_for(self, rel_path: str) -> list[str]:
        if self.team_store is None:
            return []
        try:
            sources = self.team_store.sources_for_rel_path(rel_path)
        except Exception:
            return []
        return [
            str(source.get("source_value") or "").strip()
            for source in sources
            if str(source.get("source_type") or "") == "arxiv" and str(source.get("source_value") or "").strip()
        ]

    def resolve_arxiv_id_for(self, rel_path: str) -> str | None:
        document_path = self.resolve_relative_path(rel_path)
        explicit_ids = self._arxiv_source_ids_for(rel_path)
        explicit_id = explicit_ids[0] if explicit_ids else None
        if self.web_resources.metadata_for_rel_path(rel_path) is not None:
            return explicit_id
        inferred_id = infer_arxiv_id_from_path(document_path)
        if explicit_id and inferred_id and base_arxiv_id(explicit_id) == base_arxiv_id(inferred_id):
            return choose_preferred_arxiv_id(inferred_id, explicit_id)
        return choose_preferred_arxiv_id(explicit_id, inferred_id)

    def ensure_arxiv_markdown_for_rel_path(self, rel_path: str) -> ArxivMarkdownInfo | None:
        markdown_path = self.arxiv_markdown_path_for(rel_path)
        metadata_path = self.arxiv_markdown_metadata_path_for(rel_path)
        if markdown_path.exists():
            fetched_at = None
            cached_arxiv_id = None
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    cached_arxiv_id = normalize_arxiv_id(str(metadata.get("arxiv_id") or ""))
                    fetched_at = str(metadata.get("fetched_at") or "") or None
                except json.JSONDecodeError:
                    fetched_at = None
                    cached_arxiv_id = None
            if cached_arxiv_id is None:
                cached_arxiv_id = self.resolve_arxiv_id_for(rel_path)
            if cached_arxiv_id is None:
                return None
            return ArxivMarkdownInfo(
                arxiv_id=cached_arxiv_id,
                markdown_path=markdown_path,
                metadata_path=metadata_path,
                source_url=f"https://arxiv.org/abs/{cached_arxiv_id}",
                markdown_url=markdown_new_url_for(cached_arxiv_id),
                fetched_at=fetched_at,
                cached=True,
            )

        cached_arxiv_id = self.resolve_arxiv_id_for(rel_path)
        if cached_arxiv_id is None:
            return None

        try:
            markdown_text = fetch_arxiv_markdown(cached_arxiv_id)
        except Exception:
            return None

        write_markdown_cache(markdown_path, metadata_path, arxiv_id=cached_arxiv_id, markdown_text=markdown_text)
        if self.team_store is not None and not self._arxiv_source_ids_for(rel_path):
            try:
                self.team_store.add_source(
                    rel_path,
                    source_type="arxiv",
                    source_value=cached_arxiv_id,
                    source_url=f"https://arxiv.org/abs/{cached_arxiv_id}",
                    imported_by_user_id=None,
                )
            except Exception:
                pass
        return ArxivMarkdownInfo(
            arxiv_id=cached_arxiv_id,
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            source_url=f"https://arxiv.org/abs/{cached_arxiv_id}",
            markdown_url=markdown_new_url_for(cached_arxiv_id),
            fetched_at=datetime.utcnow().isoformat(timespec="seconds"),
            cached=False,
        )

    def list_existing_prompt_slugs(self, rel_path: str, visible_slugs: set[str] | None = None) -> list[str]:
        found: set[str] = set()
        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            for path in result_dir.glob("*.md"):
                if visible_slugs is None or path.stem in visible_slugs:
                    found.add(path.stem)

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists() and (visible_slugs is None or DEFAULT_PROMPT_SLUG in visible_slugs):
            found.add(DEFAULT_PROMPT_SLUG)
        return sorted(found)

    def _existing_prompt_result_path(self, rel_path: str, prompt_slug: str) -> Path | None:
        prompt_path = self.prompt_result_path_for(rel_path, prompt_slug)
        if prompt_path.exists():
            return prompt_path
        legacy_path = self.legacy_summary_path_for(rel_path)
        if prompt_slug == DEFAULT_PROMPT_SLUG and legacy_path.exists():
            return legacy_path
        return None

    def prompt_result_info(self, rel_path: str, prompt_slug: str) -> dict[str, Any]:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return {"exists": False, "updated_at": None, "result_rel_path": None}
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        return {
            "exists": True,
            "updated_at": updated_at,
            "result_rel_path": path.relative_to(self.root).as_posix(),
        }

    def read_prompt_result(self, rel_path: str, prompt_slug: str) -> str | None:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def existing_prompt_result_path(self, rel_path: str, prompt_slug: str) -> Path | None:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return None
        return path

    def _format_prompt_result(self, rel_path: str, prompt: PromptDefinition, content: str) -> str:
        generated_at = datetime.now().isoformat(timespec="seconds")
        return (
            f"# {prompt.name}\n\n"
            f"- Source file: `{rel_path}`\n"
            f"- Prompt slug: `{prompt.slug}`\n"
            f"- Model: `{prompt.model or DEFAULT_MODEL}`\n"
            f"- Generated at: `{generated_at}`\n\n"
            "---\n\n"
            f"{content.rstrip()}\n"
        )

    def write_prompt_result(self, rel_path: str, prompt: PromptDefinition, content: str) -> Path:
        result_path = self.prompt_result_path_for(rel_path, prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(self._format_prompt_result(rel_path, prompt, content), encoding="utf-8")

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists() and legacy_path != result_path:
            legacy_path.unlink()
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)
        return result_path

    def _record_prompt_run(self, rel_path: str, prompt: PromptDefinition, result_path: Path, *, triggered_by_user_id: int | None) -> None:
        if self.team_store is None:
            return
        try:
            generated_at = datetime.fromtimestamp(result_path.stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            generated_at = datetime.utcnow().isoformat(timespec="seconds")
        try:
            result_rel_path = result_path.relative_to(self.root).as_posix()
        except ValueError:
            result_rel_path = None
        self.team_store.record_prompt_run(
            rel_path,
            prompt_slug=prompt.slug,
            prompt_name=prompt.name,
            prompt_version_id=prompt.version_id,
            prompt_version=prompt.version,
            model=prompt.model or DEFAULT_MODEL,
            result_rel_path=result_rel_path,
            status="completed",
            triggered_by_user_id=triggered_by_user_id,
            generated_at=generated_at,
            shared=True,
        )

    def generate_ai_tags(
        self,
        rel_path: str,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        triggered_by_user_id: int | None = None,
    ) -> list[str]:
        if self.team_store is None:
            return []
        tag_prompt = self.prompt_store.get_tag_prompt()

        document_path = self.resolve_relative_path(rel_path)
        if not document_path.exists() or not document_path.is_file():
            raise FileNotFoundError(rel_path)
        arxiv_markdown = self.ensure_arxiv_markdown_for_rel_path(rel_path)

        content = run_prompt_on_document(
            document_path,
            user_prompt=tag_prompt.user_prompt,
            model=load_config().llm_cheap_model,
            source_markdown_path=(arxiv_markdown.markdown_path if arxiv_markdown is not None else None),
            attachment_paths=(self.team_store.attachment_paths_for_rel_path(rel_path) if self.team_store is not None else []),
            progress_callback=progress_callback,
            should_abort=should_abort,
        )
        tags = parse_tag_generation_output(content)
        if not tags:
            raise ValueError("标签 Prompt 没有返回可用标签。")
        self.team_store.replace_generated_tags(
            rel_path,
            tags,
            source_type="ai",
            user_id=triggered_by_user_id,
            is_locked=True,
        )
        return tags

    def _maybe_refresh_ai_tags(
        self,
        rel_path: str,
        *,
        prompt_slug: str,
        progress_callback: Callable[[int, str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        triggered_by_user_id: int | None = None,
    ) -> None:
        if prompt_slug != DEFAULT_PROMPT_SLUG:
            return
        tag_prompt = self.prompt_store.get_tag_prompt()
        if not tag_prompt.enabled:
            return

        def wrapped_progress(progress: int, message: str) -> None:
            if progress_callback is None:
                return
            scaled = min(99, 90 + max(0, min(progress, 100)) // 10)
            progress_callback(scaled, f"标签生成：{message}")

        self.generate_ai_tags(
            rel_path,
            progress_callback=wrapped_progress,
            should_abort=should_abort,
            triggered_by_user_id=triggered_by_user_id,
        )

    def generate_prompt_result(
        self,
        rel_path: str,
        prompt: PromptDefinition,
        *,
        force: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        triggered_by_user_id: int | None = None,
    ) -> tuple[Path, bool]:
        document_path = self.resolve_relative_path(rel_path)
        if not document_path.exists() or not document_path.is_file():
            raise FileNotFoundError(rel_path)

        existing = self._existing_prompt_result_path(rel_path, prompt.slug)
        if existing is not None and existing.exists() and not force:
            self._record_prompt_run(rel_path, prompt, existing, triggered_by_user_id=triggered_by_user_id)
            return existing, False
        arxiv_markdown = self.ensure_arxiv_markdown_for_rel_path(rel_path)
        effective_prompt = replace(prompt, model=resolve_model(prompt.model or DEFAULT_MODEL))

        content = run_prompt_on_document(
            document_path,
            user_prompt=effective_prompt.user_prompt,
            model=effective_prompt.model,
            source_markdown_path=(arxiv_markdown.markdown_path if arxiv_markdown is not None else None),
            attachment_paths=(self.team_store.attachment_paths_for_rel_path(rel_path) if self.team_store is not None else []),
            progress_callback=progress_callback,
            should_abort=should_abort,
        )
        result_path = self.write_prompt_result(rel_path, effective_prompt, content)
        self._record_prompt_run(rel_path, effective_prompt, result_path, triggered_by_user_id=triggered_by_user_id)
        try:
            self._maybe_refresh_ai_tags(
                rel_path,
                prompt_slug=prompt.slug,
                progress_callback=progress_callback,
                should_abort=should_abort,
                triggered_by_user_id=triggered_by_user_id,
            )
        except Exception:
            # Tag generation is optional; keep the main interpretation result even if tag refresh fails.
            pass
        return result_path, True

    def run_prompt_batch(
        self,
        rel_paths: list[str],
        prompt_slugs: list[str],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        prompt_map = {prompt.slug: prompt for prompt in self.prompt_store.list_prompts()}
        prompts = [prompt_map[slug] for slug in prompt_slugs if slug in prompt_map]
        unique_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
        result = {"generated": 0, "skipped": 0, "failed": 0, "errors": []}

        for rel_path in unique_rel_paths:
            for prompt in prompts:
                try:
                    _, generated = self.generate_prompt_result(rel_path, prompt, force=force)
                    if generated:
                        result["generated"] += 1
                    else:
                        result["skipped"] += 1
                except Exception as exc:
                    result["failed"] += 1
                    result["errors"].append(f"{Path(rel_path).name} / {prompt.name}: {exc}")
        return result


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def format_source_type(source_type: str) -> str:
    mapping = {
        "arxiv": "arXiv",
        "openreview": "OpenReview",
        "pdf_url": "PDF 链接",
        "canonical_url": "网页",
        "feishu": "飞书",
        "huggingface_daily_papers": "Hugging Face Daily",
    }
    return mapping.get(source_type, source_type.replace("_", " ").strip().title())


def format_resource_kind(resource_kind: str) -> str:
    return {
        "paper": "PAPER",
        "article": "ARTICLE",
        "project": "PROJECT SITE",
        "news": "NEWS",
    }.get(resource_kind, resource_kind.replace("_", " ").upper())


def sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_for_filestorage(file_storage: Any) -> tuple[int, str]:
    stream = file_storage.stream
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
    stream.seek(0)
    return total, digest.hexdigest()


def safe_download_name(value: str, *, fallback: str) -> str:
    allowed = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = "".join(char if char in allowed else "_" for char in value).strip().rstrip(".")
    return cleaned or fallback


def build_source_groups(days: list[Any]) -> list[dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for day in days:
        year_group = tree.setdefault(day.year, {"key": day.year, "label": day.year, "count": 0, "months": {}})
        month_group = year_group["months"].setdefault(
            day.month,
            {"key": day.month, "label": f"{day.year}-{day.month}", "count": 0, "days": []},
        )
        year_group["count"] += 1
        month_group["count"] += 1
        month_group["days"].append(day)

    groups: list[dict[str, Any]] = []
    for year_key in sorted(tree.keys(), reverse=True):
        year_group = tree[year_key]
        months = [year_group["months"][month_key] for month_key in sorted(year_group["months"].keys(), reverse=True)]
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
            }
        )
    return groups



def build_groups(papers: list[PaperRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[PaperRecord]] = {}
    for paper in papers:
        label = paper.extracted_date[:7] if paper.extracted_date else "未提取日期"
        groups.setdefault(label, []).append(paper)
    ordered: list[dict[str, Any]] = []
    for label in sorted((item for item in groups if item != "未提取日期"), reverse=True):
        ordered.append({"label": label, "papers": groups[label], "count": len(groups[label])})
    if "未提取日期" in groups:
        ordered.append({"label": "未提取日期", "papers": groups["未提取日期"], "count": len(groups["未提取日期"])})
    return ordered


def build_sidebar_groups(papers: list[PaperRecord], selected_rel_path: str | None = None) -> list[dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for paper in papers:
        year_key = paper.sort_date[:4] if paper.sort_date else "unknown"
        month_key = paper.sort_date[:7] if paper.sort_date else "unknown"
        year_label = year_key if year_key != "unknown" else "未提取日期"
        month_label = month_key if month_key != "unknown" else "未分类"

        year_group = tree.setdefault(
            year_key,
            {"key": year_key, "label": year_label, "count": 0, "months": {}},
        )
        month_group = year_group["months"].setdefault(
            month_key,
            {"key": month_key, "label": month_label, "count": 0, "papers": [], "is_open": False},
        )
        year_group["count"] += 1
        month_group["count"] += 1
        month_group["papers"].append(paper)
        if selected_rel_path and paper.rel_path == selected_rel_path:
            month_group["is_open"] = True

    groups: list[dict[str, Any]] = []
    for year_key in sorted((key for key in tree if key != "unknown"), reverse=True):
        year_group = tree[year_key]
        months = [
            year_group["months"][month_key]
            for month_key in sorted((key for key in year_group["months"] if key != "unknown"), reverse=True)
        ]
        if "unknown" in year_group["months"]:
            months.append(year_group["months"]["unknown"])
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
                "is_open": any(month["is_open"] for month in months) or not groups,
            }
        )

    if "unknown" in tree:
        year_group = tree["unknown"]
        months = list(year_group["months"].values())
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
                "is_open": any(month["is_open"] for month in months),
            }
        )
    return groups



def filter_and_sort_papers(
    papers: list[PaperRecord],
    folder: str,
    query: str,
    sort_by: str,
    *,
    show_done: bool = False,
    metadata_matches: set[str] | None = None,
) -> list[PaperRecord]:
    query_text = query.strip().lower()
    folder = folder.strip().strip("/")
    metadata_matches = metadata_matches or set()
    filtered: list[PaperRecord] = []
    for paper in papers:
        if paper.is_done and not show_done:
            continue
        if folder and not (paper.folder == folder or paper.folder.startswith(folder + "/")):
            continue
        haystack = f"{paper.file_name} {paper.display_title}".lower()
        if query_text and query_text not in haystack and paper.rel_path not in metadata_matches:
            continue
        filtered.append(paper)

    if sort_by == "title":
        filtered.sort(key=lambda paper: (paper.display_title.lower(), paper.file_name.lower()))
    elif sort_by == "date_asc":
        filtered.sort(key=lambda paper: (paper.sort_date or "9999-99-99", paper.display_title.lower()))
    else:
        filtered.sort(key=lambda paper: (paper.sort_date or "0000-00-00", paper.display_title.lower()), reverse=True)
    return filtered


def parse_page(value: str | None, default: int = 1) -> int:
    try:
        page = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(1, page)


def paginate_items(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    total = len(items)
    if total == 0:
        return {
            "items": [],
            "page": 1,
            "page_size": page_size,
            "total": 0,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False,
            "prev_page": 1,
            "next_page": 1,
            "start_index": 0,
            "end_index": 0,
        }

    total_pages = max(1, (total + page_size - 1) // page_size)
    safe_page = min(max(1, page), total_pages)
    start = (safe_page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": safe_page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
        "prev_page": safe_page - 1 if safe_page > 1 else 1,
        "next_page": safe_page + 1 if safe_page < total_pages else total_pages,
        "start_index": start + 1,
        "end_index": min(end, total),
    }


def build_external_url(base_url: str, path: str) -> str:
    cleaned_path = "/" + path.lstrip("/")
    cleaned_base = normalize_public_base_url(base_url)
    return f"{cleaned_base}{cleaned_path}" if cleaned_base else cleaned_path


def resolve_feishu_oauth_config(config: PaperReaderConfig) -> FeishuOAuthConfig | None:
    if not config.feishu_enabled:
        return None

    missing: list[str] = []
    if not config.public_base_url:
        missing.append("PAPER_READER_BASE_URL")
    if not config.feishu_app_id:
        missing.append("PAPER_READER_FEISHU_APP_ID")
    if not config.feishu_app_secret:
        missing.append("PAPER_READER_FEISHU_APP_SECRET")
    if missing:
        raise ValueError(f"启用飞书登录缺少配置: {', '.join(missing)}")

    return FeishuOAuthConfig(
        app_id=config.feishu_app_id,
        app_secret=config.feishu_app_secret,
        base_url=config.public_base_url,
        api_base_url=config.feishu_api_base_url,
    )


def build_feishu_username(identity_key: str) -> str:
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:20]
    return f"{FEISHU_USERNAME_PREFIX}{digest}"


def is_safe_next_url(candidate: str | None) -> bool:
    value = (candidate or "").strip()
    if not value or value.startswith("//") or any(char in value for char in "\r\n"):
        return False
    parsed = urlparse(value)
    return not parsed.scheme and not parsed.netloc and value.startswith("/")


def normalize_next_url(candidate: str | None) -> str:
    value = (candidate or "").strip()
    return value if is_safe_next_url(value) else url_for("index")


def open_json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = None
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "paper-reader/1.0",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"
    if headers:
        request_headers.update(headers)

    request_obj = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request_obj, timeout=timeout) as response:
            raw_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"请求飞书接口失败: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise ValueError(f"请求飞书接口失败: {exc.reason}") from exc

    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("飞书接口返回了无法解析的 JSON。") from exc

    if not isinstance(parsed, dict):
        raise ValueError("飞书接口返回格式不是 JSON 对象。")
    return parsed


def feishu_response_message(payload: dict[str, Any]) -> str:
    for key in ("msg", "message", "error", "error_description"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "未知错误"


def fetch_feishu_app_access_token(config: FeishuOAuthConfig) -> str:
    payload = open_json_request(
        config.app_access_token_url,
        method="POST",
        payload={"app_id": config.app_id, "app_secret": config.app_secret},
    )
    if payload.get("code") not in {0, "0"}:
        raise ValueError(f"获取飞书 app_access_token 失败: {feishu_response_message(payload)}")
    token = payload.get("app_access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("飞书没有返回 app_access_token。")
    return token


def fetch_feishu_user_access_token(config: FeishuOAuthConfig, app_access_token: str, code: str) -> str:
    payload = open_json_request(
        config.user_access_token_url,
        method="POST",
        headers={"Authorization": f"Bearer {app_access_token}"},
        payload={"grant_type": "authorization_code", "code": code, "app_id": config.app_id},
    )
    if payload.get("code") not in {0, "0"}:
        raise ValueError(f"获取飞书 user_access_token 失败: {feishu_response_message(payload)}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("飞书 user_access_token 返回缺少 data。")
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("飞书没有返回 user_access_token。")
    return token


def fetch_feishu_user_profile(config: FeishuOAuthConfig, user_access_token: str) -> dict[str, Any]:
    payload = open_json_request(
        config.user_info_url,
        headers={"Authorization": f"Bearer {user_access_token}"},
    )
    if payload.get("code") not in {0, "0"}:
        raise ValueError(f"获取飞书用户信息失败: {feishu_response_message(payload)}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("飞书用户信息返回缺少 data。")
    return data


def is_feishu_chat_member(config: FeishuOAuthConfig, chat_id: str, user_access_token: str) -> bool:
    """Check membership with the OAuth user's token; API errors must deny access."""
    payload = open_json_request(
        config.chat_membership_url(chat_id),
        headers={"Authorization": f"Bearer {user_access_token}"},
    )
    if payload.get("code") not in {0, "0"}:
        raise ValueError(f"飞书群成员校验失败: {feishu_response_message(payload)}")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("is_in_chat"), bool):
        raise ValueError("飞书群成员校验返回缺少 is_in_chat。")
    return data["is_in_chat"]


def redirect_to_index(
    current_folder: str,
    query: str,
    sort_by: str,
    selected_paper: str | None = None,
    tab: str | None = None,
    *,
    show_done: bool = False,
    batch_show_done: bool = False,
) -> Any:
    params = {"folder": current_folder, "q": query, "sort": sort_by}
    if selected_paper:
        params["paper"] = selected_paper
    if tab:
        params["tab"] = tab
    if show_done:
        params["show_done"] = "1"
    if batch_show_done:
        params["batch_show_done"] = "1"
    return redirect(url_for("index", **params))



def create_app(
    library_root: Path | None = None,
    source_archive_root: Path | None = None,
    *,
    config: PaperReaderConfig | None = None,
) -> Flask:
    runtime_config = config or load_config()
    root = (library_root or runtime_config.library_dir).resolve()
    source_root = (source_archive_root or runtime_config.source_archive_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    # A partially supplied OAuth tuple is a deployment mistake, never a reason to
    # silently fall back to local credentials.
    if runtime_config.feishu_enabled and not runtime_config.feishu_oauth_configured:
        resolve_feishu_oauth_config(runtime_config)
    feishu_oauth = resolve_feishu_oauth_config(runtime_config) if runtime_config.auth_mode in {"feishu", "mixed"} else None
    if runtime_config.auth_mode in {"feishu", "mixed"} and feishu_oauth is None:
        raise ValueError(f"PAPER_READER_AUTH_MODE={runtime_config.auth_mode} requires configured Feishu OAuth.")
    if runtime_config.auth_mode_is_default and runtime_config.auth_mode == "local":
        LOGGER.warning("Feishu OAuth is not configured; defaulting PAPER_READER_AUTH_MODE to local for development.")
    if runtime_config.auth_mode in {"feishu", "mixed"} and not runtime_config.feishu_allowed_chat_id:
        LOGGER.warning("PAPER_READER_FEISHU_ALLOWED_CHAT_ID is unset; any Feishu-authenticated user may enter.")
    if runtime_config.secret_key_is_ephemeral:
        LOGGER.warning("PAPER_READER_SECRET_KEY is unset; using an ephemeral secret key. Sessions will reset on restart.")
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.config["SECRET_KEY"] = runtime_config.secret_key
    app.config["LIBRARY_ROOT"] = root.resolve()
    app.config["SOURCE_ARCHIVE_ROOT"] = source_root.resolve()
    app.config["AVATAR_ROOT"] = (root / AVATAR_DIR_NAME).resolve()
    Path(app.config["AVATAR_ROOT"]).mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_STAGING_ROOT"] = (root / UPLOAD_STAGING_DIR_NAME).resolve()
    Path(app.config["UPLOAD_STAGING_ROOT"]).mkdir(parents=True, exist_ok=True)
    app.config["LOGIN_USERNAME"] = runtime_config.login_username
    app.config["LOGIN_PASSWORD"] = runtime_config.login_password
    app.config["PUBLIC_BASE_URL"] = runtime_config.public_base_url
    app.config["FEISHU_OAUTH"] = feishu_oauth
    app.config["AUTH_MODE"] = runtime_config.auth_mode
    app.config["FEISHU_ALLOWED_CHAT_ID"] = runtime_config.feishu_allowed_chat_id
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=runtime_config.session_ttl_hours)
    app.config["SESSION_COOKIE_SECURE"] = runtime_config.session_cookie_secure
    app.config["CSRF_PROTECTION_ENABLED"] = True
    app.config["LLM_BASE_URL"] = runtime_config.llm_base_url
    app.config["LLM_API_KEY"] = runtime_config.llm_api_key
    app.config["LLM_MODEL"] = runtime_config.llm_model
    app.config["LLM_CHEAP_MODEL"] = runtime_config.llm_cheap_model
    app.config["LLM_ALLOWED_MODELS"] = runtime_config.llm_allowed_models
    app.config["LLM_QPS"] = runtime_config.llm_qps
    app.config["LLM_TIMEOUT"] = runtime_config.llm_timeout
    app.config["SCREEN_IMPORTS"] = runtime_config.screen_imports
    app.config["MAX_CONTENT_LENGTH"] = runtime_config.max_upload_bytes
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.settings_store = SettingsStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.team_store = TeamStore(  # type: ignore[attr-defined]
        app.config["LIBRARY_ROOT"], admin_usernames=runtime_config.admin_usernames
    )
    if runtime_config.auth_mode != "feishu" and runtime_config.login_username and runtime_config.login_password:
        app.team_store.bootstrap_default_user(runtime_config.login_username, runtime_config.login_password)  # type: ignore[attr-defined]
    app.prompt_store = PromptStore(app.config["LIBRARY_ROOT"], app.team_store.db_path)  # type: ignore[attr-defined]
    app.library = PaperLibrary(  # type: ignore[attr-defined]
        app.config["LIBRARY_ROOT"],
        app.prompt_store,
        app.team_store,
        page_fetch_max_bytes=runtime_config.fetch_page_max_bytes,
        document_fetch_max_bytes=runtime_config.fetch_max_bytes,
        fetch_max_redirects=runtime_config.fetch_max_redirects,
    )
    app.login_guard = LoginGuard()  # type: ignore[attr-defined]
    app.history_store = HistoricalInsightsStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.momentum_store = MomentumInsightsStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.opportunity_store = OpportunityInsightsStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.job_queue = PaperJobQueue(  # type: ignore[attr-defined]
        app.library,
        app.prompt_store,
        max_concurrency=app.settings_store.max_concurrency(),  # type: ignore[attr-defined]
    )
    app.action_queue = ActionTaskQueue(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]

    def clear_user_session() -> None:
        for key in ("authenticated", "user_id", "username", "display_name", "role"):
            session.pop(key, None)

    def csrf_token() -> str:
        token = session.get("csrf_token")
        if not isinstance(token, str) or not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def csrf_request_is_valid() -> bool:
        expected = session.get("csrf_token")
        supplied = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
        return isinstance(expected, str) and isinstance(supplied, str) and secrets.compare_digest(expected, supplied)

    def feishu_oauth_config() -> FeishuOAuthConfig | None:
        config = app.config.get("FEISHU_OAUTH")
        return config if isinstance(config, FeishuOAuthConfig) else None

    def external_url_for(endpoint: str, **values: Any) -> str:
        path = url_for(endpoint, **values)
        base_url = str(app.config.get("PUBLIC_BASE_URL") or "")
        return build_external_url(base_url, path)

    def sync_feishu_user(profile: dict[str, Any]) -> TeamUser:
        identity_key = str(profile.get("union_id") or profile.get("open_id") or "").strip()
        if not identity_key:
            raise ValueError("飞书用户信息里没有 union_id 或 open_id。")

        username = build_feishu_username(identity_key)
        display_name = str(profile.get("name") or profile.get("en_name") or profile.get("nickname") or username).strip() or username
        user = app.team_store.get_user_by_username(username)  # type: ignore[attr-defined]

        if user is None:
            generated_password = f"{uuid.uuid4().hex}{uuid.uuid4().hex}"
            return app.team_store.create_user(username, display_name, generated_password, "member")  # type: ignore[attr-defined]
        if not user.is_active:
            raise PermissionError("这个飞书账号对应的本地用户已停用，请联系管理员。")
        if user.display_name != display_name:
            user = app.team_store.update_user(user.id, display_name=display_name)  # type: ignore[attr-defined]
        return user

    def current_user() -> TeamUser | None:
        user_id = session.get("user_id")
        if isinstance(user_id, int):
            user = app.team_store.get_user(user_id)  # type: ignore[attr-defined]
            if user is not None and user.is_active:
                return user
            if user is not None and not user.is_active:
                clear_user_session()
                return None
        username = session.get("username")
        if isinstance(username, str) and username:
            user = app.team_store.get_user_by_username(username)  # type: ignore[attr-defined]
            if user is not None and user.is_active:
                session["user_id"] = user.id
                session["display_name"] = user.display_name
                session["role"] = user.role
                session["authenticated"] = True
                return user
            clear_user_session()
        return None

    def current_user_id() -> int | None:
        user = current_user()
        return user.id if user is not None else None

    def is_admin_user() -> bool:
        user = current_user()
        return bool(user and user.role == "admin")

    def require_admin_user() -> TeamUser | None:
        user = current_user()
        if user is None or user.role != "admin":
            flash("这个操作只有管理员能处理。", "error")
            return None
        return user

    def default_member_submission_folder(user: TeamUser | None) -> str:
        if user is None:
            return "TeamInbox"
        safe_username = secure_filename(user.username) or f"user-{user.id}"
        return f"TeamInbox/{safe_username}"

    def resolve_upload_target_folder(actor: TeamUser | None, raw_target: str) -> str:
        if actor is not None and actor.role != "admin":
            return default_member_submission_folder(actor)
        return raw_target.strip().strip("/")

    def start_user_session(user: TeamUser) -> None:
        # Flask's signed-cookie session has no server-side identifier. Clearing it before
        # establishing identity discards any attacker-controlled pre-login state.
        session.clear()
        session.permanent = True
        session["csrf_token"] = secrets.token_urlsafe(32)
        session["authenticated"] = True
        session["user_id"] = user.id
        session["username"] = user.username
        session["display_name"] = user.display_name
        session["role"] = user.role

    def avatar_initials(value: str | None) -> str:
        text = (value or "").strip()
        return (text[:1] or "?").upper()

    def avatar_url_for_rel_path(rel_path: str | None) -> str | None:
        if not rel_path:
            return None
        return url_for("avatar_file_route", filename=rel_path)

    def avatar_url_for(user: Any | None) -> str | None:
        if user is None:
            return None
        rel_path = getattr(user, "avatar_rel_path", None)
        return avatar_url_for_rel_path(str(rel_path) if rel_path else None)

    def avatar_filename_for_user(user_id: int, suffix: str) -> str:
        return f"user-{user_id}{suffix.lower()}"

    def share_url_for_rel_path(rel_path: str) -> str:
        return external_url_for("share_paper_route", rel_path=rel_path)

    def share_summary_for_paper(paper: PaperRecord) -> str:
        core_result = app.library.read_prompt_result(paper.rel_path, DEFAULT_PROMPT_SLUG)  # type: ignore[attr-defined]
        if core_result:
            summary = extract_share_summary(core_result)
            if summary:
                return summary
        preview_fallback = extract_share_summary(paper.preview_text or "")
        if preview_fallback:
            return preview_fallback
        return "这篇论文值得直接打开 PaperReader 看原文和核心解读。"

    def share_payload_for_paper(paper: PaperRecord) -> dict[str, str]:
        summary = share_summary_for_paper(paper)
        share_url = share_url_for_rel_path(paper.rel_path)
        share_text = (
            "这篇文章不错，分享给你\n\n"
            f"《{paper.display_title}》\n"
            f"一句话概括：{summary}\n\n"
            f"PaperReader 阅读链接：\n{share_url}"
        )
        return {
            "title": paper.display_title,
            "summary": summary,
            "url": share_url,
            "text": share_text,
        }

    @app.context_processor
    def inject_helpers() -> dict[str, Any]:
        user = current_user()
        return {
            "format_bytes": format_bytes,
            "format_source_type": format_source_type,
            "format_resource_kind": format_resource_kind,
            "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS)),
            "current_user": user,
            "is_admin": bool(user and user.role == "admin"),
            "member_submission_folder": default_member_submission_folder(user),
            "avatar_initials": avatar_initials,
            "avatar_url_for": avatar_url_for,
            "csrf_token": csrf_token,
        }

    def user_done_rel_paths(user_id: int | None = None) -> set[str]:
        return app.team_store.done_rel_paths_for_user(user_id if user_id is not None else current_user_id())  # type: ignore[attr-defined]

    def apply_user_done_state(papers: list[PaperRecord], *, user_id: int | None = None) -> list[PaperRecord]:
        done_rel_paths = user_done_rel_paths(user_id)
        if not done_rel_paths:
            return [replace(paper, is_done=False) for paper in papers]
        return [replace(paper, is_done=(paper.rel_path in done_rel_paths)) for paper in papers]

    @app.before_request
    def require_login() -> Any:
        endpoint = request.endpoint or ""
        allowed = {"login", "register", "logout", "health", "feishu_login", "feishu_callback"}
        if endpoint == "register" and app.config["AUTH_MODE"] == "feishu":
            abort(404)
        if endpoint in allowed or endpoint.startswith("static"):
            return None
        if session.get("authenticated") and current_user() is not None:
            return None
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=next_url.rstrip("?")))

    @app.before_request
    def protect_mutating_requests() -> Any:
        if request.method not in {"POST", "PUT", "DELETE"} or request.endpoint == "feishu_callback":
            return None
        if request.endpoint == "login" and app.config["AUTH_MODE"] == "feishu":
            abort(404)
        if app.config["CSRF_PROTECTION_ENABLED"] and not csrf_request_is_valid():
            abort(403, "CSRF token is missing or invalid.")

    @app.get("/health")
    def health() -> Any:
        return {"ok": True}

    @app.get("/logout")
    def logout() -> Any:
        session.clear()
        flash("你已退出登录。", "success")
        return redirect(url_for("login"))

    @app.get("/share/<path:rel_path>")
    def share_paper_route(rel_path: str) -> Any:
        rel_path = rel_path.strip("/")
        try:
            absolute = app.library.resolve_relative_path(rel_path)  # type: ignore[attr-defined]
        except ValueError:
            abort(404)
        if not absolute.exists() or not absolute.is_file():
            abort(404)
        return redirect(url_for("index", paper=rel_path, tab="source", show_done="1"))

    @app.get("/avatars/<path:filename>")
    def avatar_file_route(filename: str) -> Any:
        safe_name = secure_filename(filename)
        if not safe_name or safe_name != filename:
            abort(404)
        avatar_root = Path(app.config["AVATAR_ROOT"])
        target = avatar_root / safe_name
        if not target.exists() or not target.is_file():
            abort(404)
        return send_from_directory(avatar_root, safe_name, as_attachment=False)

    @app.post("/profile/avatar")
    def profile_avatar_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        file = request.files.get("avatar")
        if file is None or not getattr(file, "filename", ""):
            flash("先选一张头像图片再上传。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_AVATAR_EXTENSIONS:
            flash("头像暂时只支持 PNG / JPG / JPEG / WEBP / GIF。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        avatar_root = Path(app.config["AVATAR_ROOT"])
        avatar_root.mkdir(parents=True, exist_ok=True)
        for existing in avatar_root.glob(f"user-{actor.id}.*"):
            existing.unlink(missing_ok=True)

        avatar_name = avatar_filename_for_user(actor.id, suffix)
        destination = avatar_root / avatar_name
        file.save(destination)
        app.team_store.update_user_avatar(actor.id, avatar_name)  # type: ignore[attr-defined]
        flash("头像已经更新。", "success")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.route("/login", methods=["GET", "POST"])
    def login() -> Any:
        next_url = normalize_next_url(request.values.get("next"))
        client_key = app.login_guard.key_for_request(request)  # type: ignore[attr-defined]
        status = app.login_guard.status_for(client_key)  # type: ignore[attr-defined]
        feishu_config = feishu_oauth_config()
        local_login_enabled = app.config["AUTH_MODE"] in {"local", "mixed"}

        if request.method == "POST":
            if not local_login_enabled:
                abort(404)
            if status["locked"]:
                flash(f"密码连续输错过多，请等待 {status['remaining_seconds']} 秒后再试。", "error")
            else:
                username = request.form.get("username", "").strip()
                password = request.form.get("password", "")
                user = app.team_store.authenticate_user(username, password)  # type: ignore[attr-defined]
                if user is not None:
                    app.login_guard.register_success(client_key)  # type: ignore[attr-defined]
                    start_user_session(user)
                    return redirect(next_url or url_for("index"))

                failure = app.login_guard.register_failure(client_key)  # type: ignore[attr-defined]
                if failure["locked"]:
                    flash(f"密码连续输错 3 次，已锁定 5 分钟。请等待 {failure['remaining_seconds']} 秒后再试。", "error")
                else:
                    remaining_attempts = MAX_LOGIN_FAILURES - failure["failed_count"]
                    flash(f"用户名或密码错误。还可再试 {remaining_attempts} 次。", "error")
                status = app.login_guard.status_for(client_key)  # type: ignore[attr-defined]

        return render_template(
            "login.html",
            next_url=next_url,
            locked=status["locked"],
            remaining_seconds=status["remaining_seconds"],
            feishu_enabled=feishu_config is not None,
            feishu_callback_url=feishu_config.callback_url if feishu_config is not None else "",
            local_login_enabled=local_login_enabled,
            local_registration_enabled=app.config["AUTH_MODE"] != "feishu",
            show_manual_login=request.method == "POST" or status["locked"] or feishu_config is None,
        )

    @app.get("/login/feishu")
    def feishu_login() -> Any:
        config = feishu_oauth_config()
        if config is None:
            abort(404)

        next_url = normalize_next_url(request.args.get("next"))
        state = uuid.uuid4().hex
        session["feishu_oauth_state"] = state
        session["feishu_oauth_next"] = next_url
        params = {
            "app_id": config.app_id,
            "redirect_uri": config.callback_url,
            "state": state,
            "scope": "im:chat:readonly",
        }
        return redirect(f"{config.auth_page_url}?{urlencode(params)}")

    @app.get(FEISHU_LOGIN_REDIRECT_PATH)
    def feishu_callback() -> Any:
        config = feishu_oauth_config()
        next_url = normalize_next_url(session.pop("feishu_oauth_next", request.args.get("next")))
        if config is None:
            flash("飞书登录还没有配置。", "error")
            return redirect(url_for("login", next=next_url))

        error_message = (request.args.get("error_description") or request.args.get("error") or "").strip()
        if error_message:
            flash(f"飞书登录被取消或失败: {error_message}", "error")
            return redirect(url_for("login", next=next_url))

        returned_state = (request.args.get("state") or "").strip()
        expected_state = str(session.pop("feishu_oauth_state", "") or "").strip()
        if not returned_state or not expected_state or returned_state != expected_state:
            flash("飞书登录状态校验失败，请重新发起登录。", "error")
            return redirect(url_for("login", next=next_url))

        code = (request.args.get("code") or "").strip()
        if not code:
            flash("飞书回调里没有 code。", "error")
            return redirect(url_for("login", next=next_url))

        try:
            app_access_token = fetch_feishu_app_access_token(config)
            user_access_token = fetch_feishu_user_access_token(config, app_access_token, code)
            profile = fetch_feishu_user_profile(config, user_access_token)
            allowed_chat_id = str(app.config["FEISHU_ALLOWED_CHAT_ID"] or "").strip()
            if allowed_chat_id:
                try:
                    if not is_feishu_chat_member(config, allowed_chat_id, user_access_token):
                        return render_template("feishu_access_denied.html"), 403
                except ValueError as exc:
                    LOGGER.error("Feishu group membership verification failed closed: %s", exc)
                    return render_template("feishu_access_denied.html"), 403
            user = sync_feishu_user(profile)
        except (PermissionError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("login", next=next_url))

        start_user_session(user)
        return redirect(next_url)

    @app.route("/register", methods=["GET", "POST"])
    def register() -> Any:
        if app.config["AUTH_MODE"] == "feishu":
            abort(404)
        next_url = normalize_next_url(request.values.get("next"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            display_name = request.form.get("display_name", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            if password != confirm_password:
                flash("两次输入的密码不一致。", "error")
            else:
                try:
                    user = app.team_store.register_user(username, display_name, password)  # type: ignore[attr-defined]
                except ValueError as exc:
                    flash(str(exc), "error")
                else:
                    start_user_session(user)
                    flash(f"欢迎加入，{user.display_name}。账号已经创建并自动登录。", "success")
                    return redirect(next_url or url_for("index"))

        return render_template("register.html", next_url=next_url)

    def serialize_paper_for_view(
        paper: PaperRecord,
        *,
        current_folder: str,
        query: str,
        sort_by: str,
        active_prompt_count: int,
        show_done: bool,
    ) -> dict[str, Any]:
        year_key = paper.sort_date[:4] if paper.sort_date else "unknown"
        month_key = paper.sort_date[:7] if paper.sort_date else "unknown"
        paper_url_params: dict[str, Any] = {
            "folder": current_folder,
            "q": query,
            "sort": sort_by,
            "paper": paper.rel_path,
            "tab": "source",
        }
        if show_done:
            paper_url_params["show_done"] = "1"
        return {
            "rel_path": paper.rel_path,
            "file_name": paper.file_name,
            "folder": paper.folder,
            "extension": paper.extension,
            "display_title": paper.display_title,
            "extracted_date": paper.extracted_date,
            "sort_date": paper.sort_date,
            "prompt_result_count": paper.prompt_result_count,
            "active_prompt_count": active_prompt_count,
            "is_done": paper.is_done,
            "year_key": year_key,
            "month_key": month_key,
            "year_label": year_key if year_key != "unknown" else "未提取日期",
            "month_label": month_key if month_key != "unknown" else "未分类",
            "paper_url": url_for(
                "index",
                **paper_url_params,
            ),
        }

    def ensure_paper_metadata(rel_path: str) -> PaperRecord:
        visible_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
        paper = app.library.build_record_for_rel_path(rel_path, visible_slugs)  # type: ignore[attr-defined]
        app.team_store.sync_papers([paper])  # type: ignore[attr-defined]
        return paper


    def flash_submission_summary(result: dict[str, Any], *, action_label: str) -> None:
        if result["queued"]:
            flash(f"{action_label}，系统已经开始处理 {result['queued']} 项分析。", "success")
        if result["existing"]:
            flash(f"其中有 {result['existing']} 项分析已经在处理中，所以没有重复提交。", "success")
        if result["skipped"]:
            flash(f"其中有 {result['skipped']} 项结果已经存在，所以没有重复生成。", "success")
        if result["invalid"]:
            flash(f"有 {result['invalid']} 项分析没找到对应模板，所以这次没有开始。", "error")

    def summarize_action_items(items: list[dict[str, Any]]) -> dict[str, int]:
        summary = {
            "total": len(items),
            "saved": 0,
            "duplicate": 0,
            "existing": 0,
            "failed": 0,
            "running": 0,
            "queued": 0,
        }
        for item in items:
            status = str(item.get("status") or "queued")
            if status in summary:
                summary[status] += 1
            elif status in {"downloading", "importing", "resolving"}:
                summary["running"] += 1
        return summary

    def build_action_result_payload(
        items: list[dict[str, Any]],
        *,
        summary_overrides: dict[str, Any] | None = None,
        rel_paths: list[str] | None = None,
        submission: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "items": items,
            "summary": summarize_action_items(items),
        }
        if summary_overrides:
            payload["summary"].update(summary_overrides)
        if rel_paths is not None:
            payload["rel_paths"] = rel_paths
        if submission is not None:
            payload["submission"] = submission
        return payload

    def action_progress_percent(items: list[dict[str, Any]]) -> int:
        if not items:
            return 100
        terminal_statuses = {"saved", "duplicate", "existing", "failed"}
        completed = sum(1 for item in items if str(item.get("status")) in terminal_statuses)
        return max(1, min(99, int((completed / len(items)) * 100)))

    def update_action_item(
        items: list[dict[str, Any]],
        *,
        key: str,
        status: str,
        message: str,
        rel_path: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> None:
        for item in items:
            if item.get("key") != key:
                continue
            item["status"] = status
            item["message"] = message
            if rel_path is not None:
                item["rel_path"] = rel_path
            if source is not None:
                item["source"] = source
            return

    def selected_source_papers(day_record: Any, selected_ids: list[str]) -> list[Any]:
        paper_map = day_paper_map(day_record)
        normalized_ids = [paper_id.strip() for paper_id in selected_ids if paper_id.strip()]
        if not normalized_ids:
            return list(day_record.papers)
        return [paper_map[paper_id] for paper_id in normalized_ids if paper_id in paper_map]

    def stage_uploaded_file(file: Any) -> dict[str, Any]:
        filename = getattr(file, "filename", "") or ""
        if not filename:
            raise ValueError("没有选择文件。")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型：{filename}")

        staging_root = Path(app.config["UPLOAD_STAGING_ROOT"])
        staging_root.mkdir(parents=True, exist_ok=True)
        safe_name = secure_filename(Path(filename).name) or Path(filename).name
        unique_dir = staging_root / uuid.uuid4().hex
        unique_dir.mkdir(parents=True, exist_ok=True)
        staged_path = unique_dir / safe_name
        try:
            file.save(staged_path)
        except Exception:
            if staged_path.exists():
                staged_path.unlink()
            shutil.rmtree(unique_dir, ignore_errors=True)
            raise

        return {
            "staged_rel_path": staged_path.relative_to(Path(app.config["LIBRARY_ROOT"])).as_posix(),
            "original_name": filename,
            "size": staged_path.stat().st_size,
        }


    def resolve_import_target(raw_target: str) -> dict[str, Any]:
        target = classify_import_target(normalize_import_target(raw_target))
        return asdict(target)

    def download_remote_pdf(download_url: str) -> Path:
        last_error: Exception | None = None
        for attempt in range(1, REMOTE_PDF_DOWNLOAD_ATTEMPTS + 1):
            temp_path: Path | None = None
            try:
                response = fetch_public_url(
                    download_url,
                    accept="application/pdf,*/*;q=0.1",
                    timeout=REMOTE_PDF_DOWNLOAD_TIMEOUT,
                    max_bytes=runtime_config.fetch_max_bytes,
                    max_redirects=runtime_config.fetch_max_redirects,
                )
                if not response.body.startswith(b"%PDF-"):
                    raise RuntimeError("The remote response is not a PDF document.")
                with tempfile.NamedTemporaryFile(
                    prefix="paper-reader-import-",
                    suffix=".pdf",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    handle.write(response.body)
                return temp_path
            except Exception as exc:
                last_error = exc
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                if attempt >= REMOTE_PDF_DOWNLOAD_ATTEMPTS:
                    raise RuntimeError(f"{exc}（已重试 {REMOTE_PDF_DOWNLOAD_ATTEMPTS} 次）") from exc
                time.sleep(min(2 * attempt, 6))

        raise RuntimeError(str(last_error) if last_error else "下载失败。")

    def submit_remote_import_jobs(rel_paths: list[str], actor: TeamUser, *, source: str) -> dict[str, Any]:
        return submit_remote_import_jobs_with_force(rel_paths, actor, source=source, force=False)

    def submit_remote_import_jobs_with_force(
        rel_paths: list[str],
        actor: TeamUser,
        *,
        source: str,
        force: bool,
    ) -> dict[str, Any]:
        auto_prompts = app.prompt_store.auto_prompts()  # type: ignore[attr-defined]
        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        unique_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
        if auto_prompts and unique_rel_paths:
            submission = app.job_queue.submit(  # type: ignore[attr-defined]
                unique_rel_paths,
                [prompt.slug for prompt in auto_prompts],
                force=force,
                source=source,
                requested_by_user_id=actor.id,
                requested_by_display_name=actor.display_name,
            )
        return submission

    def resolved_arxiv_id(resolved: dict[str, str]) -> str | None:
        if str(resolved.get("source_type") or "") != "arxiv":
            return None
        return normalize_arxiv_id(str(resolved.get("source_url") or resolved.get("source_value") or ""))

    def current_arxiv_id_for_rel_path(rel_path: str) -> str | None:
        candidates: list[str] = []
        try:
            sources = app.team_store.sources_for_rel_path(rel_path)  # type: ignore[attr-defined]
        except Exception:
            sources = []
        for source in sources:
            if str(source.get("source_type") or "") != "arxiv":
                continue
            source_url = str(source.get("source_url") or "")
            source_value = str(source.get("source_value") or "")
            if source_url:
                candidates.append(source_url)
            if source_value:
                candidates.append(source_value)
        try:
            candidates.append(str(app.library.resolve_arxiv_id_for(rel_path) or ""))  # type: ignore[attr-defined]
        except Exception:
            pass
        return choose_preferred_arxiv_id(*candidates)

    def should_upgrade_existing_arxiv(rel_path: str, resolved: dict[str, str]) -> bool:
        incoming_id = resolved_arxiv_id(resolved)
        if incoming_id is None:
            return False
        existing_id = current_arxiv_id_for_rel_path(rel_path)
        return arxiv_version_number(incoming_id) > arxiv_version_number(existing_id)

    def describe_import_source(source_type: str) -> str:
        mapping = {
            "arxiv": "arXiv",
            "openreview": "OpenReview",
            "pdf_url": "PDF 链接",
            "canonical_url": "网页",
        }
        return mapping.get(source_type, source_type)

    def format_batch_import_result_message(item: dict[str, Any]) -> tuple[str, str]:
        source = item.get("source") or {}
        source_type = str(source.get("source_type") or "unknown")
        source_label = describe_import_source(source_type)
        target = str(item.get("target") or "未知目标")
        rel_path = item.get("rel_path")
        status = str(item.get("status") or "failed")

        if status == "saved":
            return ("success", f"[{source_label}] {target} 已导入：{rel_path}")
        if status == "duplicate":
            return ("success", f"[{source_label}] {target} 检测到重复文件，已复用：{rel_path}")
        if status == "existing":
            return ("success", f"[{source_label}] {target} 已在库中：{rel_path}")
        error = str(item.get("error") or "未知错误")
        return ("error", f"[{source_label}] {target} 导入失败：{error}")

    def submit_remote_import_action(raw_targets: str, recommendation_reason: str, actor: TeamUser) -> ActionRecord:
        parsed_targets = parse_import_targets(raw_targets)
        if not parsed_targets:
            raise ValueError("请填写 arXiv ID 或 HTTP/HTTPS 研究链接。")
        title = f"远程导入（{len(parsed_targets)} 项）" if len(parsed_targets) > 1 else f"远程导入：{parsed_targets[0]}"
        return app.action_queue.submit(  # type: ignore[attr-defined]
            kind="remote-import",
            title=title,
            payload={
                "raw_targets": raw_targets,
                "recommendation_reason": recommendation_reason,
            },
            source="remote-import",
            requested_by_user_id=actor.id,
            requested_by_display_name=actor.display_name,
        )

    def submit_source_import_action(day_record: Any, selected: list[Any], actor: TeamUser) -> ActionRecord:
        title = f"来源归档导入：{day_record.run_date}（{len(selected)} 篇）"
        return app.action_queue.submit(  # type: ignore[attr-defined]
            kind="source-import",
            title=title,
            payload={
                "run_date": day_record.run_date,
                "paper_ids": [paper.paper_id for paper in selected if paper.paper_id],
            },
            source="source-import",
            requested_by_user_id=actor.id,
            requested_by_display_name=actor.display_name,
        )

    def submit_ai_tag_action(rel_path: str, actor: TeamUser) -> ActionRecord:
        return app.action_queue.submit(  # type: ignore[attr-defined]
            kind="ai-tags",
            title=f"AI 标签刷新：{Path(rel_path).name}",
            payload={"rel_path": rel_path},
            source="ai-tags",
            requested_by_user_id=actor.id,
            requested_by_display_name=actor.display_name,
        )

    def submit_upload_import_action(staged_files: list[dict[str, Any]], target_folder: str, actor: TeamUser) -> ActionRecord:
        title = f"上传整理（{len(staged_files)} 个文件）" if len(staged_files) > 1 else f"上传整理：{staged_files[0]['original_name']}"
        return app.action_queue.submit(  # type: ignore[attr-defined]
            kind="upload-import",
            title=title,
            payload={
                "staged_files": staged_files,
                "target_folder": target_folder,
            },
            source="upload",
            requested_by_user_id=actor.id,
            requested_by_display_name=actor.display_name,
        )

    def try_prefetch_arxiv_markdown(rel_path: str) -> dict[str, Any] | None:
        try:
            info = app.library.ensure_arxiv_markdown_for_rel_path(rel_path)  # type: ignore[attr-defined]
        except Exception:
            return None
        if info is None:
            return None
        return app.library.arxiv_markdown_status_for_rel_path(rel_path)  # type: ignore[attr-defined]

    def record_action_provenance(action: ActionRecord, rel_path: str, target: str) -> None:
        entries = action.payload.get("provenance") or []
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_url = str(entry.get("url") or "").strip()
            if entry_url and entry_url != target:
                continue
            source_value = str(entry.get("chat_name") or "").strip()
            if not source_value:
                continue
            app.team_store.add_source(  # type: ignore[attr-defined]
                rel_path,
                source_type="feishu",
                source_value=source_value,
                source_url=target,
                source_context=str(entry.get("context_text") or "").strip() or None,
                sender_open_id=str(entry.get("sender_open_id") or "").strip() or None,
                message_time_iso=str(entry.get("msg_time_iso") or "").strip() or None,
                imported_by_user_id=action.requested_by_user_id,
            )

    def perform_remote_import_action(
        action: ActionRecord,
        report: Callable[[int, str, dict[str, Any] | None], None],
        should_abort: Callable[[], bool],
    ) -> dict[str, Any]:
        raw_targets = str(action.payload.get("raw_targets") or "")
        recommendation_reason = str(action.payload.get("recommendation_reason") or "")
        requested_by_user_id = action.requested_by_user_id

        targets = parse_import_targets(raw_targets)
        if not targets:
            raise ValueError("请填写 arXiv ID 或 HTTP/HTTPS 研究链接。")

        unique_targets: list[str] = []
        seen_targets: set[str] = set()
        for target in targets:
            if target in seen_targets:
                continue
            seen_targets.add(target)
            unique_targets.append(target)

        items = [{"key": target, "target": target, "status": "queued", "message": "等待处理。", "rel_path": None, "source": None} for target in unique_targets]
        report(1, f"开始处理 {len(items)} 条远程导入任务。", build_action_result_payload(items))

        actor = app.team_store.get_user(requested_by_user_id) if requested_by_user_id is not None else None  # type: ignore[attr-defined]
        pending_downloads: list[dict[str, Any]] = []
        rel_paths_for_jobs: list[str] = []
        force_regenerate_rel_paths: list[str] = []

        for item in items:
            if should_abort():
                raise InterruptedError("Import interrupted.")
            target = str(item["target"])
            update_action_item(items, key=target, status="resolving", message="正在识别链接类型。")
            report(action_progress_percent(items), f"正在识别：{target}", build_action_result_payload(items))
            try:
                resolved = resolve_import_target(target)
            except Exception as exc:
                item["error"] = str(exc)
                update_action_item(items, key=target, status="failed", message=str(exc))
                report(action_progress_percent(items), f"识别失败：{target}", build_action_result_payload(items))
                continue

            item["source"] = resolved
            existing_rel_path = app.team_store.find_paper_by_source(resolved["source_type"], resolved["source_value"])  # type: ignore[attr-defined]
            if existing_rel_path and resolved.get("route") == "web":
                update_action_item(items, key=target, status="downloading", message="已在库中，正在刷新网页快照。", source=resolved)
                pending_downloads.append(
                    {"target": target, "resolved": resolved, "existing_rel_path": existing_rel_path, "refresh": True}
                )
                report(action_progress_percent(items), f"正在刷新：{target}", build_action_result_payload(items))
                continue
            if existing_rel_path:
                if should_upgrade_existing_arxiv(existing_rel_path, resolved):
                    update_action_item(items, key=target, status="downloading", message="检测到 arXiv 新版本，正在下载并覆盖旧版本。", source=resolved)
                    pending_downloads.append({"target": target, "resolved": resolved, "existing_rel_path": existing_rel_path, "upgrade": True})
                    report(action_progress_percent(items), f"发现新版本：{target}", build_action_result_payload(items))
                    continue
                if recommendation_reason.strip() and actor is not None:
                    app.team_store.add_recommendation(existing_rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]
                markdown_status = try_prefetch_arxiv_markdown(existing_rel_path)
                existing_message = "已在库中。"
                if markdown_status and markdown_status.get("cached"):
                    existing_message = "已在库中，已确认可直接使用 arXiv Markdown。"
                update_action_item(items, key=target, status="existing", message=existing_message, rel_path=existing_rel_path, source=resolved)
                record_action_provenance(action, existing_rel_path, target)
                rel_paths_for_jobs.append(existing_rel_path)
                if actor is not None:
                    submit_remote_import_jobs([existing_rel_path], actor, source="remote-import-batch")
                report(action_progress_percent(items), f"已存在：{target}", build_action_result_payload(items))
                continue

            if resolved.get("route") == "web":
                update_action_item(items, key=target, status="downloading", message="正在抓取并提取网页正文。", source=resolved)
            else:
                update_action_item(items, key=target, status="downloading", message="正在下载 PDF。", source=resolved)
            pending_downloads.append({"target": target, "resolved": resolved})
        if pending_downloads:
            max_workers = min(4, len(pending_downloads))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for pending in pending_downloads:
                    resolved_target = pending["resolved"]
                    if resolved_target.get("route") == "web":
                        future = executor.submit(
                            app.library.web_resources.fetch_and_store,  # type: ignore[attr-defined]
                            ImportTarget(**resolved_target),
                            existing_rel_path=(str(pending.get("existing_rel_path") or "") or None),
                        )
                    else:
                        future = executor.submit(download_remote_pdf, resolved_target["download_url"])
                    future_map[future] = pending
                for future in as_completed(future_map):
                    if should_abort():
                        raise InterruptedError("Import interrupted.")
                    item = future_map[future]
                    target = str(item["target"])
                    resolved = item["resolved"]
                    upgrade_rel_path = str(item.get("existing_rel_path") or "") or None
                    try:
                        fetched_result = future.result()
                    except Exception as exc:
                        update_action_item(items, key=target, status="failed", message=f"下载失败：{exc}", source=resolved)
                        report(action_progress_percent(items), f"下载失败：{target}", build_action_result_payload(items))
                        continue

                    if resolved.get("route") == "web":
                        snapshot = fetched_result
                        rel_path = str(snapshot.rel_path)
                        if (
                            app.config["SCREEN_IMPORTS"]
                            and not item.get("refresh")
                        ):
                            try:
                                snapshot_path = app.library.resolve_relative_path(rel_path)  # type: ignore[attr-defined]
                                sample = snapshot_path.read_text(encoding="utf-8", errors="replace")[:1500]
                                packet = ScreeningPacket(
                                    item_id=rel_path,
                                    rel_path=rel_path,
                                    title=snapshot.title,
                                    url=resolved["canonical_url"],
                                    kind=resolved["resource_kind"],
                                    provenance=[],
                                    content=sample,
                                    fetch_status=snapshot.fetch_status,
                                )
                                decision = screen_packets(
                                    [packet],
                                    get_llm_client(runtime_config),
                                    model=runtime_config.llm_cheap_model,
                                    batch_size=1,
                                )[0]
                            except Exception as exc:
                                decision = None
                                screening_reason = f"内容筛选失败：{exc}"
                            else:
                                screening_reason = decision.error or decision.reason
                            if decision is None or decision.decision in {"drop", "error"} or decision.error:
                                app.library.web_resources.delete_snapshot(rel_path)  # type: ignore[attr-defined]
                                update_action_item(
                                    items,
                                    key=target,
                                    status="failed",
                                    message=f"链接未通过内容筛选：{screening_reason}",
                                    source=resolved,
                                )
                                report(action_progress_percent(items), f"链接未通过内容筛选：{target}", build_action_result_payload(items))
                                continue
                            app.library.web_resources.set_screening_result(  # type: ignore[attr-defined]
                                rel_path,
                                decision=decision.decision,
                                reason=decision.reason,
                            )
                        app.library._update_active_index_entry(rel_path)  # type: ignore[attr-defined]
                        active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
                        imported_record = app.library.build_record_for_rel_path(rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
                        app.team_store.sync_papers([imported_record])  # type: ignore[attr-defined]
                        app.team_store.add_source(  # type: ignore[attr-defined]
                            rel_path,
                            source_type=resolved["source_type"],
                            source_value=resolved["source_value"],
                            source_url=resolved["source_url"],
                            imported_by_user_id=requested_by_user_id,
                        )
                        record_action_provenance(action, rel_path, target)
                        if runtime_config.llm_api_key and snapshot.fetch_status == "ok":
                            try:
                                attachments = app.library.web_resources.discover_attachments(  # type: ignore[attr-defined]
                                    rel_path,
                                    client=get_llm_client(runtime_config),
                                    model=runtime_config.llm_cheap_model,
                                )
                                app.team_store.replace_attachments(rel_path, attachments)  # type: ignore[attr-defined]
                            except Exception as exc:
                                LOGGER.warning("Attachment discovery failed for %s: %s", rel_path, exc)
                        if recommendation_reason.strip() and actor is not None:
                            app.team_store.add_recommendation(rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]
                        if snapshot.fetch_status == "ok":
                            item_status = "existing" if item.get("refresh") else "saved"
                            message = "网页快照已刷新。" if item.get("refresh") else "网页正文已提取并入库。"
                            update_action_item(items, key=target, status=item_status, message=message, rel_path=rel_path, source=resolved)
                            rel_paths_for_jobs.append(rel_path)
                            report(action_progress_percent(items), f"完成：{target}", build_action_result_payload(items))
                        else:
                            message = f"已创建条目，但抓取状态为 {snapshot.fetch_status}：{snapshot.error or '无法提取正文'}"
                            update_action_item(items, key=target, status="failed", message=message, rel_path=rel_path, source=resolved)
                            report(action_progress_percent(items), f"抓取失败但已保留条目：{target}", build_action_result_payload(items))
                        continue

                    temp_path = fetched_result
                    try:
                        update_action_item(items, key=target, status="importing", message="下载完成，正在入库。", source=resolved)
                        report(action_progress_percent(items), f"正在入库：{target}", build_action_result_payload(items))
                        if upgrade_rel_path is not None:
                            import_result = app.library.replace_existing_file(  # type: ignore[attr-defined]
                                upgrade_rel_path,
                                temp_path,
                                clear_derived=True,
                            )
                        else:
                            import_result = app.library.import_external_file(  # type: ignore[attr-defined]
                                temp_path,
                                resolved["target_folder"],
                                preferred_name=resolved["preferred_name"],
                            )
                    finally:
                        temp_path.unlink(missing_ok=True)

                    if import_result["status"] == "duplicate" and import_result.get("duplicate_rel_path"):
                        rel_path = str(import_result["duplicate_rel_path"])
                    elif import_result["status"] == "existing" and import_result.get("saved_rel_path"):
                        rel_path = str(import_result["saved_rel_path"])
                    elif import_result["status"] == "saved" and import_result.get("saved_rel_path"):
                        rel_path = str(import_result["saved_rel_path"])
                        active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
                        imported_record = app.library.build_record_for_rel_path(rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
                        app.team_store.sync_papers([imported_record])  # type: ignore[attr-defined]
                    else:
                        update_action_item(items, key=target, status="failed", message=import_result.get("message") or "远程导入失败。", source=resolved)
                        report(action_progress_percent(items), f"导入失败：{target}", build_action_result_payload(items))
                        continue

                    app.team_store.add_source(  # type: ignore[attr-defined]
                        rel_path,
                        source_type=resolved["source_type"],
                        source_value=resolved["source_value"],
                        source_url=resolved["source_url"],
                        imported_by_user_id=requested_by_user_id,
                    )
                    record_action_provenance(action, rel_path, target)
                    markdown_status = try_prefetch_arxiv_markdown(rel_path)
                    if recommendation_reason.strip() and actor is not None:
                        app.team_store.add_recommendation(rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]

                    final_message = "导入完成。"
                    if upgrade_rel_path is not None:
                        final_message = "检测到 arXiv 新版本，已覆盖旧版本并重新生成。"
                        force_regenerate_rel_paths.append(rel_path)
                    if markdown_status and markdown_status.get("cached"):
                        if upgrade_rel_path is not None:
                            final_message = "检测到 arXiv 新版本，已覆盖旧版本，并刷新为新的 arXiv Markdown。"
                        else:
                            final_message = "导入完成，已缓存 arXiv Markdown。"
                    update_action_item(items, key=target, status=import_result["status"], message=final_message, rel_path=rel_path, source=resolved)
                    rel_paths_for_jobs.append(rel_path)
                    report(action_progress_percent(items), f"完成：{target}", build_action_result_payload(items))

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if rel_paths_for_jobs and actor is not None:
            normal_rel_paths = [path for path in rel_paths_for_jobs if path not in set(force_regenerate_rel_paths)]
            forced_rel_paths = [path for path in rel_paths_for_jobs if path in set(force_regenerate_rel_paths)]
            if normal_rel_paths:
                submission = submit_remote_import_jobs_with_force(normal_rel_paths, actor, source="remote-import-batch", force=False)
            if forced_rel_paths:
                forced_submission = submit_remote_import_jobs_with_force(forced_rel_paths, actor, source="remote-import-batch", force=True)
                submission = {
                    "queued": submission["queued"] + forced_submission["queued"],
                    "existing": submission["existing"] + forced_submission["existing"],
                    "skipped": submission["skipped"] + forced_submission["skipped"],
                    "invalid": submission["invalid"] + forced_submission["invalid"],
                    "job_ids": submission["job_ids"] + forced_submission["job_ids"],
                    "jobs": submission["jobs"] + forced_submission["jobs"],
                }

        result = build_action_result_payload(
            items,
            rel_paths=rel_paths_for_jobs,
            submission=submission,
        )
        summary = result["summary"]
        result["_final_message"] = (
            f"导入完成：新导入 {summary['saved']}，"
            f"复用重复 {summary['duplicate']}，"
            f"已存在 {summary['existing']}，"
            f"失败 {summary['failed']}。"
        )
        report(100, str(result["_final_message"]), result)
        return result

    def perform_source_import_action(
        action: ActionRecord,
        report: Callable[[int, str, dict[str, Any] | None], None],
        should_abort: Callable[[], bool],
    ) -> dict[str, Any]:
        run_date = str(action.payload.get("run_date") or "").strip()
        paper_ids = [str(item).strip() for item in action.payload.get("paper_ids", []) if str(item).strip()]
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            raise ValueError("没有找到这一天的来源归档。")

        selected = selected_source_papers(day_record, paper_ids)
        if not selected:
            raise ValueError("先选至少一篇论文。")

        items = [
            {
                "key": paper.paper_id or paper.title,
                "target": paper.paper_id or paper.title,
                "title": paper.title,
                "status": "queued",
                "message": "等待处理。",
                "rel_path": None,
                "source": {"source_type": day_record.source},
            }
            for paper in selected
        ]
        report(1, f"开始导入来源归档，共 {len(items)} 篇。", build_action_result_payload(items))

        actor = app.team_store.get_user(action.requested_by_user_id) if action.requested_by_user_id is not None else None  # type: ignore[attr-defined]
        rel_paths_for_jobs: list[str] = []
        for paper in selected:
            if should_abort():
                raise InterruptedError("Import interrupted.")
            item_key = paper.paper_id or paper.title
            source_pdf_path = local_pdf_path_for(day_record, paper)
            if source_pdf_path is None or not source_pdf_path.exists():
                update_action_item(items, key=item_key, status="failed", message="缺少可用 PDF。")
                report(action_progress_percent(items), f"缺少 PDF：{item_key}", build_action_result_payload(items))
                continue

            preferred_name = paper.pdf_file_name or f"{paper.paper_id}.pdf"
            update_action_item(items, key=item_key, status="importing", message="正在导入本地 PDF。")
            try:
                import_result = app.library.import_external_file(  # type: ignore[attr-defined]
                    source_pdf_path,
                    f"Sources/HuggingFace/{day_record.year}/{day_record.month}/{day_record.day}",
                    preferred_name=preferred_name,
                )
            except (FileNotFoundError, ValueError) as exc:
                update_action_item(items, key=item_key, status="failed", message=str(exc))
                report(action_progress_percent(items), f"导入失败：{item_key}", build_action_result_payload(items))
                continue

            if import_result["status"] == "saved" and import_result["saved_rel_path"]:
                rel_path = str(import_result["saved_rel_path"])
                rel_paths_for_jobs.append(rel_path)
                active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
                imported_record = app.library.build_record_for_rel_path(rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
                app.team_store.sync_papers([imported_record])  # type: ignore[attr-defined]
                app.team_store.add_source(  # type: ignore[attr-defined]
                    rel_path,
                    source_type=day_record.source,
                    source_value=paper.paper_id or paper.title,
                    source_url=paper.url,
                    imported_by_user_id=action.requested_by_user_id,
                )
                markdown_status = try_prefetch_arxiv_markdown(rel_path)
                final_message = "导入完成。"
                if markdown_status and markdown_status.get("cached"):
                    final_message = "导入完成，已缓存 arXiv Markdown。"
                update_action_item(items, key=item_key, status="saved", message=final_message, rel_path=rel_path)
                if actor is not None:
                    submit_remote_import_jobs([rel_path], actor, source="source-import")
            elif import_result["status"] == "duplicate" and import_result.get("duplicate_rel_path"):
                duplicate_rel_path = str(import_result["duplicate_rel_path"])
                rel_paths_for_jobs.append(duplicate_rel_path)
                markdown_status = try_prefetch_arxiv_markdown(duplicate_rel_path)
                duplicate_message = "检测到重复文件，已复用。"
                if markdown_status and markdown_status.get("cached"):
                    duplicate_message = "检测到重复文件，已复用，并确认已有 arXiv Markdown。"
                update_action_item(items, key=item_key, status="duplicate", message=duplicate_message, rel_path=duplicate_rel_path)
                if actor is not None:
                    submit_remote_import_jobs([duplicate_rel_path], actor, source="source-import")
            else:
                update_action_item(items, key=item_key, status="failed", message=import_result.get("message") or "导入失败。")
            report(action_progress_percent(items), f"已处理：{item_key}", build_action_result_payload(items))

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if rel_paths_for_jobs and actor is not None:
            submission = submit_remote_import_jobs(rel_paths_for_jobs, actor, source="source-import")

        target_folder = f"Sources/HuggingFace/{day_record.year}/{day_record.month}/{day_record.day}"
        result = build_action_result_payload(items, rel_paths=rel_paths_for_jobs, submission=submission)
        result["target_folder"] = target_folder
        summary = result["summary"]
        result["_final_message"] = f"来源导入完成：新导入 {summary['saved']}，复用重复 {summary['duplicate']}，失败 {summary['failed']}。"
        report(100, str(result["_final_message"]), result)
        return result

    def perform_ai_tag_action(
        action: ActionRecord,
        report: Callable[[int, str, dict[str, Any] | None], None],
        should_abort: Callable[[], bool],
    ) -> dict[str, Any]:
        rel_path = str(action.payload.get("rel_path") or "").strip("/")
        if not rel_path:
            raise ValueError("论文不存在。")
        ensure_paper_metadata(rel_path)
        tags = app.library.generate_ai_tags(  # type: ignore[attr-defined]
            rel_path,
            progress_callback=lambda progress, message: report(progress, message, {"tags": []}),
            should_abort=should_abort,
            triggered_by_user_id=action.requested_by_user_id,
        )
        result = {
            "rel_path": rel_path,
            "tags": tags,
            "_final_message": f"AI 标签已刷新：{', '.join(tags)}",
        }
        report(100, str(result["_final_message"]), result)
        return result

    def perform_upload_import_action(
        action: ActionRecord,
        report: Callable[[int, str, dict[str, Any] | None], None],
        should_abort: Callable[[], bool],
    ) -> dict[str, Any]:
        target_folder = str(action.payload.get("target_folder") or "").strip().strip("/")
        staged_files = list(action.payload.get("staged_files") or [])
        if not staged_files:
            raise ValueError("没有可处理的上传文件。")

        items = []
        for entry in staged_files:
            original_name = str(entry.get("original_name") or "")
            staged_rel_path = str(entry.get("staged_rel_path") or "")
            items.append(
                {
                    "key": staged_rel_path or original_name,
                    "target": original_name,
                    "status": "queued",
                    "message": "等待处理。",
                    "rel_path": None,
                    "source": {"source_type": "upload"},
                }
            )
        report(1, f"开始整理上传文件，共 {len(items)} 个。", build_action_result_payload(items))

        actor = app.team_store.get_user(action.requested_by_user_id) if action.requested_by_user_id is not None else None  # type: ignore[attr-defined]
        rel_paths_for_jobs: list[str] = []
        library_root = Path(app.config["LIBRARY_ROOT"])

        for entry in staged_files:
            if should_abort():
                raise InterruptedError("Upload interrupted.")
            staged_rel_path = str(entry.get("staged_rel_path") or "")
            original_name = str(entry.get("original_name") or "")
            key = staged_rel_path or original_name
            staged_path = library_root / staged_rel_path
            update_action_item(items, key=key, status="importing", message="正在查重并入库。")
            report(action_progress_percent(items), f"正在处理：{original_name}", build_action_result_payload(items))
            try:
                if not staged_path.exists() or not staged_path.is_file():
                    raise FileNotFoundError(original_name or staged_rel_path)
                import_result = app.library.import_external_file(  # type: ignore[attr-defined]
                    staged_path,
                    target_folder,
                    preferred_name=original_name or None,
                )
            except Exception as exc:
                update_action_item(items, key=key, status="failed", message=str(exc))
                report(action_progress_percent(items), f"处理失败：{original_name}", build_action_result_payload(items))
                continue
            finally:
                try:
                    staged_path.unlink(missing_ok=True)
                    shutil.rmtree(staged_path.parent, ignore_errors=True)
                except Exception:
                    pass

            if import_result["status"] == "duplicate" and import_result.get("duplicate_rel_path"):
                rel_path = str(import_result["duplicate_rel_path"])
                markdown_status = try_prefetch_arxiv_markdown(rel_path)
                duplicate_message = "检测到重复文件，已复用。"
                if markdown_status and markdown_status.get("cached"):
                    duplicate_message = "检测到重复文件，已复用，并确认已有 arXiv Markdown。"
                update_action_item(items, key=key, status="duplicate", message=duplicate_message, rel_path=rel_path)
                rel_paths_for_jobs.append(rel_path)
                if actor is not None:
                    submit_remote_import_jobs([rel_path], actor, source="upload")
            elif import_result["status"] == "saved" and import_result.get("saved_rel_path"):
                rel_path = str(import_result["saved_rel_path"])
                active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
                imported_record = app.library.build_record_for_rel_path(rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
                app.team_store.sync_papers([imported_record])  # type: ignore[attr-defined]
                markdown_status = try_prefetch_arxiv_markdown(rel_path)
                final_message = "上传整理完成。"
                if markdown_status and markdown_status.get("cached"):
                    final_message = "上传整理完成，已缓存 arXiv Markdown。"
                update_action_item(items, key=key, status="saved", message=final_message, rel_path=rel_path)
                rel_paths_for_jobs.append(rel_path)
                if actor is not None:
                    submit_remote_import_jobs([rel_path], actor, source="upload")
            else:
                update_action_item(items, key=key, status="failed", message=import_result.get("message") or "上传整理失败。")
            report(action_progress_percent(items), f"已处理：{original_name}", build_action_result_payload(items))

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if rel_paths_for_jobs and actor is not None:
            submission = submit_remote_import_jobs(rel_paths_for_jobs, actor, source="upload")
        result = build_action_result_payload(items, rel_paths=rel_paths_for_jobs, submission=submission)
        summary = result["summary"]
        result["_final_message"] = (
            f"上传整理完成：新导入 {summary['saved']}，"
            f"复用重复 {summary['duplicate']}，"
            f"失败 {summary['failed']}。"
        )
        report(100, str(result["_final_message"]), result)
        return result

    app.action_queue.register_handler("remote-import", perform_remote_import_action)  # type: ignore[attr-defined]
    app.action_queue.register_handler("source-import", perform_source_import_action)  # type: ignore[attr-defined]
    app.action_queue.register_handler("ai-tags", perform_ai_tag_action)  # type: ignore[attr-defined]
    app.action_queue.register_handler("upload-import", perform_upload_import_action)  # type: ignore[attr-defined]
    app.action_queue.start()  # type: ignore[attr-defined]


    def import_remote_paper(target: str, recommendation_reason: str) -> dict[str, Any]:
        actor = current_user()
        if actor is None:
            raise PermissionError("需要先登录。")
        resolved = resolve_import_target(target)
        existing_rel_path = app.team_store.find_paper_by_source(resolved["source_type"], resolved["source_value"])  # type: ignore[attr-defined]
        if existing_rel_path:
            if should_upgrade_existing_arxiv(existing_rel_path, resolved):
                temp_path = download_remote_pdf(resolved["download_url"])
                try:
                    result = app.library.replace_existing_file(  # type: ignore[attr-defined]
                        existing_rel_path,
                        temp_path,
                        clear_derived=True,
                    )
                finally:
                    temp_path.unlink(missing_ok=True)
                rel_path = existing_rel_path
                app.team_store.add_source(  # type: ignore[attr-defined]
                    rel_path,
                    source_type=resolved["source_type"],
                    source_value=resolved["source_value"],
                    source_url=resolved["source_url"],
                    imported_by_user_id=actor.id,
                )
                try_prefetch_arxiv_markdown(rel_path)
                if recommendation_reason.strip():
                    app.team_store.add_recommendation(rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]
                submission = submit_remote_import_jobs_with_force([rel_path], actor, source="remote-import", force=True)
                return {"status": result["status"], "rel_path": rel_path, "source": resolved, "submission": submission}
            if recommendation_reason.strip():
                app.team_store.add_recommendation(existing_rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]
            try_prefetch_arxiv_markdown(existing_rel_path)
            return {"status": "existing", "rel_path": existing_rel_path, "source": resolved}

        temp_path = download_remote_pdf(resolved["download_url"])
        try:
            result = app.library.import_external_file(  # type: ignore[attr-defined]
                temp_path,
                resolved["target_folder"],
                preferred_name=resolved["preferred_name"],
            )
        finally:
            temp_path.unlink(missing_ok=True)

        if result["status"] == "duplicate" and result.get("duplicate_rel_path"):
            rel_path = str(result["duplicate_rel_path"])
        elif result["status"] == "saved" and result.get("saved_rel_path"):
            rel_path = str(result["saved_rel_path"])
            active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
            imported_record = app.library.build_record_for_rel_path(rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
            app.team_store.sync_papers([imported_record])  # type: ignore[attr-defined]
        else:
            raise RuntimeError(result.get("message") or "远程导入失败。")

        app.team_store.add_source(  # type: ignore[attr-defined]
            rel_path,
            source_type=resolved["source_type"],
            source_value=resolved["source_value"],
            source_url=resolved["source_url"],
            imported_by_user_id=actor.id,
        )
        try_prefetch_arxiv_markdown(rel_path)
        if recommendation_reason.strip():
            app.team_store.add_recommendation(rel_path, actor.id, recommendation_reason)  # type: ignore[attr-defined]

        submission = submit_remote_import_jobs_with_force([rel_path], actor, source="remote-import", force=False)
        return {"status": result["status"], "rel_path": rel_path, "source": resolved, "submission": submission}

    def ensure_prompt_run_metadata(paper: PaperRecord, prompts: list[PromptDefinition]) -> None:
        for prompt in prompts:
            result_path = app.library.existing_prompt_result_path(paper.rel_path, prompt.slug)  # type: ignore[attr-defined]
            if result_path is None:
                continue
            generated_at = datetime.fromtimestamp(result_path.stat().st_mtime).isoformat(timespec="seconds")
            app.team_store.backfill_prompt_run(  # type: ignore[attr-defined]
                paper.rel_path,
                prompt_slug=prompt.slug,
                prompt_name=prompt.name,
                prompt_version_id=prompt.version_id,
                prompt_version=prompt.version,
                model=prompt.model or DEFAULT_MODEL,
                result_rel_path=result_path.relative_to(app.library.root).as_posix(),  # type: ignore[attr-defined]
                generated_at=generated_at,
            )

    def migrate_legacy_done_documents() -> int:
        legacy_documents = app.library.iter_done_documents()  # type: ignore[attr-defined]
        if not legacy_documents:
            return 0

        active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
        all_prompts = app.prompt_store.list_prompts()  # type: ignore[attr-defined]
        migrated_records: list[PaperRecord] = []
        for path in legacy_documents:
            old_rel_path = path.relative_to(app.library.root).as_posix()  # type: ignore[attr-defined]
            destination = app.library.restore_destination_for(old_rel_path)  # type: ignore[attr-defined]
            path.rename(destination)
            new_rel_path = destination.relative_to(app.library.root).as_posix()  # type: ignore[attr-defined]
            app.library._move_prompt_results(old_rel_path, new_rel_path)  # type: ignore[attr-defined]
            cached = app.library._hash_cache.pop(old_rel_path, None)  # type: ignore[attr-defined]
            if cached:
                app.library._hash_cache[new_rel_path] = cached  # type: ignore[attr-defined]

            moved_paper = app.library.build_record_for_rel_path(new_rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
            app.team_store.rename_paper(old_rel_path, moved_paper)  # type: ignore[attr-defined]
            app.team_store.sync_papers([moved_paper])  # type: ignore[attr-defined]
            app.team_store.mark_done_for_all_users(new_rel_path)  # type: ignore[attr-defined]
            ensure_prompt_run_metadata(moved_paper, all_prompts)
            migrated_records.append(moved_paper)

        app.library.rebuild_active_index(lightweight=True)  # type: ignore[attr-defined]
        app.library.rebuild_done_index(lightweight=True)  # type: ignore[attr-defined]
        return len(migrated_records)

    def load_chat_prompt_contexts(rel_path: str, prompts: list[PromptDefinition]) -> list[tuple[str, str]]:
        contexts: list[tuple[str, str]] = []
        for prompt in prompts:
            content = app.library.read_prompt_result(rel_path, prompt.slug)  # type: ignore[attr-defined]
            if not content:
                continue
            contexts.append((prompt.name, content[:4000]))
            if len(contexts) >= 3:
                break
        return contexts

    def serialize_chat_message(message: Any) -> dict[str, Any]:
        payload = asdict(message) if hasattr(message, "__dataclass_fields__") else dict(message)
        payload["body_html"] = render_markdown(str(payload.get("body", "")))
        payload["avatar_url"] = avatar_url_for_rel_path(payload.get("avatar_rel_path"))
        payload["avatar_label"] = avatar_initials(str(payload.get("display_name", "")))
        return payload

    def serialize_chat_context(context: dict[str, Any]) -> dict[str, Any]:
        return {
            visibility: {
                "count": int(thread.get("count", 0)),
                "visibility": thread.get("visibility", visibility),
                "messages": [serialize_chat_message(message) for message in thread.get("messages", [])],
            }
            for visibility, thread in context.items()
        }

    app.chat_queue = PaperChatQueue(  # type: ignore[attr-defined]
        library=app.library,
        prompt_store=app.prompt_store,
        team_store=app.team_store,
        prompt_context_loader=load_chat_prompt_contexts,
    )
    migrate_legacy_done_documents()

    def build_batch_papers(
        *,
        folder: str,
        query: str,
        sort_by: str,
        show_done: bool,
        batch_show_done: bool,
        selected_rel_path: str = "",
    ) -> list[PaperRecord]:
        scan = app.library.scan()  # type: ignore[attr-defined]
        app.team_store.sync_papers(scan.papers)  # type: ignore[attr-defined]
        papers = apply_user_done_state(scan.papers)
        metadata_matches = app.team_store.search_rel_paths(query) if query.strip() else set()  # type: ignore[attr-defined]
        batch_papers = filter_and_sort_papers(
            papers,
            folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
            metadata_matches=metadata_matches,
        )
        if not batch_show_done:
            batch_papers = [paper for paper in batch_papers if not paper.is_done]
        return batch_papers

    def current_page_state() -> dict[str, Any]:
        folder = request.args.get("folder", "")
        query = request.args.get("q", "")
        sort_by = request.args.get("sort", "date_desc")
        show_done = request.args.get("show_done", "").strip().lower() in {"1", "true", "yes", "on"}
        batch_show_done = request.args.get("batch_show_done", "").strip().lower() in {"1", "true", "yes", "on"}
        batch_page = parse_page(request.args.get("batch_page"))
        selected_rel_path = request.args.get("paper", "").strip("/")
        selected_tab = request.args.get("tab", "source")

        scan = app.library.scan()  # type: ignore[attr-defined]
        app.team_store.sync_papers(scan.papers)  # type: ignore[attr-defined]
        papers_with_user_state = apply_user_done_state(scan.papers)
        all_prompts = app.prompt_store.list_prompts()  # type: ignore[attr-defined]
        active_prompts = [prompt for prompt in all_prompts if prompt.enabled]
        metadata_matches = app.team_store.search_rel_paths(query) if query.strip() else set()  # type: ignore[attr-defined]

        papers = filter_and_sort_papers(
            papers_with_user_state,
            folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=show_done,
            metadata_matches=metadata_matches,
        )
        batch_papers = filter_and_sort_papers(
            papers_with_user_state,
            folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
            metadata_matches=metadata_matches,
        )
        if not batch_show_done:
            batch_papers = [paper for paper in batch_papers if not paper.is_done]
        batch_library_papers = filter_and_sort_papers(
            papers_with_user_state,
            folder="",
            query="",
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
            metadata_matches=set(),
        )
        if not batch_show_done:
            batch_library_papers = [paper for paper in batch_library_papers if not paper.is_done]
        sidebar_groups = build_sidebar_groups(papers, selected_rel_path)

        selected_paper = next((item for item in papers if item.rel_path == selected_rel_path), None)
        if selected_paper is None and papers:
            selected_paper = papers[0]
        if selected_paper is not None:
            ensure_prompt_run_metadata(selected_paper, all_prompts)

        return {
            "scan": scan,
            "all_prompts": all_prompts,
            "active_prompts": active_prompts,
            "folder": folder,
            "query": query,
            "sort_by": sort_by,
            "show_done": show_done,
            "batch_show_done": batch_show_done,
            "batch_page": batch_page,
            "selected_rel_path": selected_rel_path,
            "selected_tab": selected_tab,
            "papers": papers,
            "batch_papers": batch_papers,
            "batch_library_total": len(batch_library_papers),
            "sidebar_groups": sidebar_groups,
            "selected_paper": selected_paper,
        }

    @app.get("/")
    def index() -> str:
        state = current_page_state()
        all_prompts = state["all_prompts"]
        active_prompts = state["active_prompts"]
        folder = state["folder"]
        query = state["query"]
        sort_by = state["sort_by"]
        show_done = state["show_done"]
        batch_show_done = state["batch_show_done"]
        batch_page = state["batch_page"]
        selected_tab = state["selected_tab"]
        papers = state["papers"]
        batch_papers = state["batch_papers"]
        sidebar_groups = state["sidebar_groups"]
        selected_paper = state["selected_paper"]

        preview_paragraphs: list[str] = []
        selected_prompt: PromptDefinition | None = None
        selected_prompt_content: str | None = None
        selected_prompt_html: Markup | None = None
        selected_prompt_info: dict[str, Any] | None = None
        selected_prompt_job: dict[str, Any] | None = None
        selected_share_payload: dict[str, str] | None = None
        selected_web_html: Markup | None = None
        selected_screening: dict[str, str] | None = None
        viewer_tabs: list[dict[str, Any]] = []
        selected_arxiv_markdown_status: dict[str, Any] = {
            "available": False,
            "cached": False,
            "arxiv_id": None,
            "source_url": None,
            "markdown_url": None,
            "markdown_rel_path": None,
            "metadata_rel_path": None,
            "fetched_at": None,
            "char_count": None,
        }
        selected_paper_context: dict[str, Any] = {
            "recommendations": [],
            "tags": [],
            "comments": [],
            "like_count": 0,
            "liked_by_current_user": False,
            "recommended_by_current_user": False,
            "recommendation_count": 0,
            "comment_count": 0,
            "prompt_runs": [],
            "sources": [],
            "attachments": [],
            "current_user_recommendation_reason": "",
        }
        selected_chat_context: dict[str, Any] = {
            "shared": {"messages": [], "count": 0, "visibility": "shared"},
            "private": {"messages": [], "count": 0, "visibility": "private"},
        }

        if selected_paper:
            selected_share_payload = share_payload_for_paper(selected_paper)
            if selected_paper.preview_text:
                preview_paragraphs = [chunk.strip() for chunk in selected_paper.preview_text.split("\n\n") if chunk.strip()]
            if selected_paper.preview_kind in {"web", "markdown"}:
                try:
                    snapshot_markdown = app.library.resolve_relative_path(selected_paper.rel_path).read_text(encoding="utf-8")  # type: ignore[attr-defined]
                    selected_web_html = Markup(render_markdown(snapshot_markdown))
                    web_metadata = app.library.web_resources.metadata_for_rel_path(selected_paper.rel_path) or {}  # type: ignore[attr-defined]
                    raw_screening = web_metadata.get("screening")
                    if isinstance(raw_screening, dict):
                        selected_screening = {
                            "decision": str(raw_screening.get("decision") or ""),
                            "reason": str(raw_screening.get("reason") or ""),
                        }
                except OSError:
                    selected_web_html = None
            selected_arxiv_markdown_status = app.library.arxiv_markdown_status_for_rel_path(selected_paper.rel_path)  # type: ignore[attr-defined]
            selected_paper_context = app.team_store.paper_context(  # type: ignore[attr-defined]
                selected_paper.rel_path,
                current_user_id(),
            )
            selected_chat_context = serialize_chat_context(app.team_store.chat_context(  # type: ignore[attr-defined]
                selected_paper.rel_path,
                current_user_id(),
            ))
            for prompt in active_prompts:
                info = app.library.prompt_result_info(selected_paper.rel_path, prompt.slug)  # type: ignore[attr-defined]
                latest_job = app.job_queue.latest_job_for(selected_paper.rel_path, prompt.slug)  # type: ignore[attr-defined]
                viewer_tabs.append({
                    "slug": prompt.slug,
                    "name": prompt.name,
                    "model": prompt.model,
                    "exists": info["exists"],
                    "updated_at": info["updated_at"],
                    "job_status": latest_job.status if latest_job else None,
                    "job_progress": latest_job.progress if latest_job else None,
                })
            valid_tabs = {"source"} | {prompt.slug for prompt in active_prompts}
            if selected_tab not in valid_tabs:
                selected_tab = "source"
            if selected_tab != "source":
                selected_prompt = next((prompt for prompt in active_prompts if prompt.slug == selected_tab), None)
                if selected_prompt is not None:
                    selected_prompt_content = app.library.read_prompt_result(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    if selected_prompt_content:
                        selected_prompt_html = Markup(render_markdown(selected_prompt_content))
                    selected_prompt_info = app.library.prompt_result_info(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    latest_job = app.job_queue.latest_job_for(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    if latest_job is not None:
                        selected_prompt_job = asdict(latest_job)
        else:
            selected_tab = "source"

        recommendation_feed = []
        excluded_rel_paths = set() if show_done else user_done_rel_paths()
        for item in app.team_store.recent_recommendation_feed(limit=12, window_days=14):  # type: ignore[attr-defined]
            if item["rel_path"] in excluded_rel_paths:
                continue
            paper_url_params: dict[str, Any] = {
                "folder": folder,
                "q": query,
                "sort": sort_by,
                "paper": item["rel_path"],
                "tab": "source",
            }
            if show_done:
                paper_url_params["show_done"] = "1"
            recommendation_feed.append(
                {
                    **item,
                    "paper_url": url_for("index", **paper_url_params),
                }
            )
            if len(recommendation_feed) >= 5:
                break

        return render_template(
            "index.html",
            papers=papers,
            folders=state["scan"].folders,
            current_folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=show_done,
            batch_show_done=batch_show_done,
            batch_page=batch_page,
            selected_paper=selected_paper,
            selected_tab=selected_tab,
            preview_paragraphs=preview_paragraphs,
            sidebar_groups=sidebar_groups,
            viewer_tabs=viewer_tabs,
            selected_prompt=selected_prompt,
            selected_prompt_content=selected_prompt_content,
            selected_prompt_html=selected_prompt_html,
            selected_prompt_info=selected_prompt_info,
            selected_prompt_job=selected_prompt_job,
            selected_share_payload=selected_share_payload,
            selected_web_html=selected_web_html,
            selected_screening=selected_screening,
            selected_arxiv_markdown_status=selected_arxiv_markdown_status,
            selected_paper_context=selected_paper_context,
            selected_chat_context=selected_chat_context,
            recommendation_feed=recommendation_feed,
            active_prompts=active_prompts,
            active_prompt_count=len(active_prompts),
            library_root=app.config["LIBRARY_ROOT"],
            initial_job_snapshot=app.job_queue.snapshot(),  # type: ignore[attr-defined]
            initial_action_snapshot=app.action_queue.snapshot(),  # type: ignore[attr-defined]
        )

    @app.get("/tool-panels/<panel_name>")
    def tool_panel_route(panel_name: str) -> Any:
        state = current_page_state()
        context = {
            "current_folder": state["folder"],
            "query": state["query"],
            "sort_by": state["sort_by"],
            "show_done": state["show_done"],
            "batch_show_done": state["batch_show_done"],
            "batch_page": state["batch_page"],
            "selected_paper": state["selected_paper"],
            "selected_tab": state["selected_tab"],
            "active_prompt_count": len(state["active_prompts"]),
        }

        if panel_name == "prompt-manager":
            return render_template(
                "panels/prompt_manager.html",
                **context,
                all_prompts=state["all_prompts"],
                can_manage_prompts=is_admin_user(),
                tag_prompt=app.prompt_store.get_tag_prompt(),  # type: ignore[attr-defined]
                new_prompt_defaults={
                    "name": "",
                    "slug": "",
                    "model": DEFAULT_MODEL,
                    "user_prompt": DEFAULT_USER_PROMPT,
                    "enabled": True,
                    "auto_run": True,
                },
            )

        if panel_name == "offline-package":
            return render_template(
                "panels/offline_package.html",
                **context,
                filtered_papers=state["batch_papers"],
            )

        if panel_name == "batch-run":
            pagination = paginate_items(state["batch_papers"], state["batch_page"], DEFAULT_BATCH_PANEL_PAGE_SIZE)
            return render_template(
                "panels/batch_run.html",
                **context,
                filtered_papers=pagination["items"],
                batch_pagination=pagination,
                batch_library_total=state["batch_library_total"],
                all_prompts=state["all_prompts"],
            )

        if panel_name == "team-admin":
            return render_template(
                "panels/team_admin.html",
                **context,
                can_manage_team=is_admin_user(),
                team_users=app.team_store.list_users(),  # type: ignore[attr-defined]
            )

        abort(404)

    @app.post("/upload")
    def upload() -> Any:
        files = request.files.getlist("files")
        actor = current_user()
        target_folder = resolve_upload_target_folder(actor, request.form.get("target_folder", ""))
        current_folder = request.form.get("folder", target_folder or "").strip().strip("/")
        if actor is not None and actor.role != "admin":
            current_folder = target_folder
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        if not files or not any(file.filename for file in files):
            flash("先选至少一个文件，再开始上传。", "error")
            return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)

        staged_files: list[dict[str, Any]] = []
        error_count = 0
        for file in files:
            try:
                staged_files.append(stage_uploaded_file(file))
            except ValueError as exc:
                error_count += 1
                flash(str(exc), "error")

        if not staged_files:
            flash("没有可处理的上传文件。", "error")
            return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)

        task = submit_upload_import_action(staged_files, target_folder, actor)
        flash(f"后台上传整理任务已开始：{task.title}。上传和分析已经拆开，页面不会再卡住。", "success")
        if error_count:
            flash(f"有 {error_count} 个文件因为格式问题没有进入后台整理。", "error")
        return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)

    @app.post("/upload-file")
    def upload_file() -> Any:
        file = request.files.get("file")
        actor = current_user()
        target_folder = resolve_upload_target_folder(actor, request.form.get("target_folder", ""))
        current_folder = request.form.get("folder", target_folder or "").strip().strip("/")
        if actor is not None and actor.role != "admin":
            current_folder = target_folder
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))

        if file is None or not file.filename:
            return {"status": "error", "message": "先选一个文件再上传。"}, 400
        if actor is None:
            return {"status": "error", "message": "请先登录。"}, 401

        try:
            staged = stage_uploaded_file(file)
            task = submit_upload_import_action([staged], target_folder, actor)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}, 400
        except Exception as exc:
            return {"status": "error", "message": f"上传没成功：{exc}"}, 500

        return {
            "status": "queued",
            "message": "文件已上传到服务器，后台正在查重、入库并安排分析。",
            "task_id": task.id,
            "task_title": task.title,
            "saved_rel_path": None,
            "visible_in_current_view": False,
        }, 200

    @app.post("/folders")
    def create_folder_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        folder_name = request.form.get("new_folder", "").strip()
        parent_folder = request.form.get("parent_folder", "").strip().strip("/")
        target = "/".join(part for part in [parent_folder, folder_name] if part)
        try:
            app.library.create_folder(target)  # type: ignore[attr-defined]
            flash(f"已经新建文件夹：{target}", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/rename")
    def rename_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        tab = request.form.get("tab", "source")
        rel_path = request.form.get("rel_path", "").strip("/")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        new_name = request.form.get("new_name", "").strip()
        try:
            new_rel_path = app.library.rename_file(rel_path, new_name)  # type: ignore[attr-defined]
            active_prompt_slugs = [prompt.slug for prompt in app.prompt_store.active_prompts()]  # type: ignore[attr-defined]
            moved_paper = app.library.build_record_for_rel_path(new_rel_path, active_prompt_slugs)  # type: ignore[attr-defined]
            app.team_store.rename_paper(rel_path, moved_paper)  # type: ignore[attr-defined]
            flash("文件名已经改好了。", "success")
            return redirect_to_index(current_folder, query, sort_by, new_rel_path, tab, show_done=show_done)
        except (FileNotFoundError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/delete")
    def delete_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, "source", show_done=show_done)
        try:
            app.library.delete_file(rel_path)  # type: ignore[attr-defined]
            app.team_store.delete_paper(rel_path)  # type: ignore[attr-defined]
            flash("这篇论文已经删除。", "success")
        except FileNotFoundError:
            flash("这篇论文已经找不到了。", "error")
        return redirect_to_index(current_folder, query, sort_by, show_done=show_done)

    @app.post("/done-toggle")
    def done_toggle_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            ensure_paper_metadata(rel_path)
            is_done = app.team_store.toggle_done_state(rel_path, actor.id)  # type: ignore[attr-defined]
            flash("这篇论文已经标记为已读。" if is_done else "这篇论文已经恢复为未读。", "success")
            selected_rel_path = rel_path if (show_done or not is_done) else None
            next_tab = tab if selected_rel_path else "source"
            return redirect_to_index(
                current_folder,
                query,
                sort_by,
                selected_rel_path,
                next_tab,
                show_done=show_done,
            )
        except (FileNotFoundError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/paper-date")
    def paper_date_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        precision = request.form.get("precision", "month").strip().lower()
        date_value = request.form.get(f"date_{precision}", "").strip()
        if not date_value:
            date_value = request.form.get("date_value", "").strip() or request.form.get("year_month", "").strip()
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            if not rel_path:
                raise FileNotFoundError("")
            if app.library.resolve_arxiv_id_for(rel_path) is not None:  # type: ignore[attr-defined]
                raise ValueError("这篇论文已有 arXiv ID，暂不支持手动修改日期。")
            app.library.set_manual_date(rel_path, precision, date_value, updated_by_user_id=actor.id)  # type: ignore[attr-defined]
            ensure_paper_metadata(rel_path)
            flash("论文日期已更新。", "success")
        except FileNotFoundError:
            flash("论文不存在。", "error")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/prompt-run")
    def prompt_run_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        prompt_slug = request.form.get("prompt_slug", "").strip()
        force = parse_checkbox(request.form.get("force"))
        prompt = app.prompt_store.get_prompt(prompt_slug)  # type: ignore[attr-defined]
        if prompt is None:
            flash("这份分析模板不存在。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, "source", show_done=show_done)
        actor = current_user()
        submission = app.job_queue.submit(  # type: ignore[attr-defined]
            [rel_path],
            [prompt.slug],
            force=force,
            source="manual",
            requested_by_user_id=actor.id if actor else None,
            requested_by_display_name=actor.display_name if actor else None,
        )
        flash_submission_summary(submission, action_label=f"《{prompt.name}》这份解读已经加入处理队列")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, prompt_slug, show_done=show_done)

    @app.post("/prompt-save")
    def prompt_save_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        existing_slug = request.form.get("existing_slug", "").strip() or None
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        try:
            prompt = app.prompt_store.save_prompt(  # type: ignore[attr-defined]
                existing_slug=existing_slug,
                name=request.form.get("name", ""),
                slug=request.form.get("slug", ""),
                user_prompt=request.form.get("user_prompt", ""),
                model=request.form.get("model", DEFAULT_MODEL),
                enabled=parse_checkbox(request.form.get("enabled")),
                auto_run=parse_checkbox(request.form.get("auto_run")),
                admin_only=True,
                created_by_user_id=admin_user.id,
            )
            flash(f"分析模板《{prompt.name}》已经保存。", "success")
            app.library.invalidate_scan_cache()  # type: ignore[attr-defined]
            if tab != "source" and tab == prompt.slug and not prompt.enabled:
                tab = "source"
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/tag-prompt-save")
    def tag_prompt_save_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        try:
            tag_prompt = app.prompt_store.save_tag_prompt(  # type: ignore[attr-defined]
                user_prompt=request.form.get("user_prompt", ""),
                model=request.form.get("model", DEFAULT_MODEL),
                enabled=parse_checkbox(request.form.get("enabled")),
                updated_by_user_id=admin_user.id,
            )
            if tag_prompt.enabled:
                flash("标签生成 Prompt 已保存；重新运行“核心解读”后会自动刷新 AI 标签。", "success")
            else:
                flash("标签生成 Prompt 已保存；当前处于关闭状态。", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/tags/generate-ai")
    def tag_generate_ai_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            ensure_paper_metadata(rel_path)
            task = submit_ai_tag_action(rel_path, admin_user)
            flash(f"AI 标签刷新任务已开始：{task.title}", "success")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            flash(f"AI 标签生成失败：{exc}", "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/prompt-delete")
    def prompt_delete_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        prompt_slug = request.form.get("prompt_slug", "").strip()
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        try:
            removed = app.prompt_store.delete_prompt(prompt_slug)  # type: ignore[attr-defined]
            flash(f"分析模板《{removed.name}》已经删除。", "success")
            app.library.invalidate_scan_cache()  # type: ignore[attr-defined]
            if tab == prompt_slug:
                tab = "source"
        except FileNotFoundError:
            flash("这份分析模板不存在。", "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/team/users/save")
    def team_user_save_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        try:
            user = app.team_store.create_user(  # type: ignore[attr-defined]
                request.form.get("username", ""),
                request.form.get("display_name", ""),
                request.form.get("password", ""),
                request.form.get("role", "member") or "member",
            )
            flash(f"账号 {user.display_name} 已经创建。", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/team/users/update")
    def team_user_update_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        try:
            user_id = int(request.form.get("user_id", "0") or 0)
            updated_user = app.team_store.update_user(  # type: ignore[attr-defined]
                user_id,
                display_name=request.form.get("display_name", ""),
                role_slug=request.form.get("role", "member") or "member",
                is_active=parse_checkbox(request.form.get("is_active")),
                password=request.form.get("new_password", ""),
                acting_user_id=admin_user.id,
            )
            flash(f"账号 {updated_user.display_name} 已经更新。", "success")
        except (FileNotFoundError, ValueError) as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/recommend")
    def recommend_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            ensure_paper_metadata(rel_path)
            mode = request.form.get("mode", "toggle").strip().lower() or "toggle"
            reason = request.form.get("reason", "")
            if mode == "save":
                app.team_store.add_recommendation(rel_path, actor.id, reason)  # type: ignore[attr-defined]
                flash("推荐信息已经保存；这篇论文会继续出现在团队推荐里。", "success")
            else:
                recommended = app.team_store.toggle_recommendation(rel_path, actor.id)  # type: ignore[attr-defined]
                flash("这篇论文已经加入推荐。" if recommended else "这篇论文已经从推荐里移除。", "success")
        except (ValueError, FileNotFoundError) as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/like-toggle")
    def like_toggle_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            ensure_paper_metadata(rel_path)
            recommended = app.team_store.toggle_recommendation(rel_path, actor.id)  # type: ignore[attr-defined]
            flash(
                "点赞已经并入推荐；这篇论文已经加入推荐。"
                if recommended
                else "点赞已经并入推荐；这篇论文已经从推荐里移除。",
                "success",
            )
        except FileNotFoundError:
            flash("论文不存在。", "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/comments")
    def comment_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        parent_id_value = request.form.get("parent_id", "").strip()
        try:
            ensure_paper_metadata(rel_path)
            parent_id = int(parent_id_value) if parent_id_value else None
            app.team_store.add_comment(rel_path, actor.id, request.form.get("body", ""), parent_id=parent_id)  # type: ignore[attr-defined]
            flash("评论已发布。", "success")
        except (ValueError, FileNotFoundError) as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.get("/chat/context")
    def chat_context_route() -> Any:
        rel_path = request.args.get("paper", "").strip("/")
        actor = current_user()
        if actor is None:
            return {"error": "unauthorized"}, 401
        if not rel_path:
            return {"error": "missing paper"}, 400
        return serialize_chat_context(app.team_store.chat_context(rel_path, actor.id))  # type: ignore[attr-defined]

    @app.post("/chat/send")
    def chat_send_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        visibility = request.form.get("visibility", "shared").strip().lower() or "shared"
        wants_json = request.headers.get("X-Requested-With") == "fetch"
        actor = current_user()
        if actor is None:
            if wants_json:
                return {"error": "unauthorized"}, 401
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        message_body = request.form.get("body", "")
        try:
            ensure_paper_metadata(rel_path)
            user_message_id = app.team_store.add_chat_message(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                body=message_body,
                user_id=actor.id,
                role="user",
            )
            history = app.team_store.chat_history(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                current_user_id=actor.id,
                limit=10,
            )
            assistant_message_id = app.team_store.add_chat_message(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                body="Paper Bot 正在思考...",
                user_id=actor.id if visibility == "private" else None,
                role="assistant",
                status="pending",
                model=runtime_config.llm_model,
            )
            app.chat_queue.submit(  # type: ignore[attr-defined]
                rel_path=rel_path,
                visibility=visibility,
                user_id=actor.id,
                display_name=actor.display_name,
                question=message_body.strip(),
                history=history,
                assistant_message_id=assistant_message_id,
                model=runtime_config.llm_model,
            )
            if wants_json:
                return {
                    "ok": True,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                    "context": serialize_chat_context(app.team_store.chat_context(rel_path, actor.id)),  # type: ignore[attr-defined]
                }
            flash("消息已经发出，Paper Bot 正在回复。", "success")
        except (FileNotFoundError, ValueError) as exc:
            if wants_json:
                return {"error": str(exc)}, 400
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/chat/send-stream")
    def chat_send_stream_route() -> Any:
        rel_path = request.form.get("rel_path", "").strip("/")
        visibility = request.form.get("visibility", "shared").strip().lower() or "shared"
        actor = current_user()
        if actor is None:
            return {"error": "unauthorized"}, 401
        message_body = request.form.get("body", "")
        try:
            ensure_paper_metadata(rel_path)
            user_message_id = app.team_store.add_chat_message(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                body=message_body,
                user_id=actor.id,
                role="user",
            )
            history = app.team_store.chat_history(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                current_user_id=actor.id,
                limit=10,
            )
            assistant_message_id = app.team_store.add_chat_message(  # type: ignore[attr-defined]
                rel_path,
                visibility=visibility,
                body="Paper Bot 正在思考...",
                user_id=actor.id if visibility == "private" else None,
                role="assistant",
                status="pending",
                model=runtime_config.llm_model,
            )
        except (FileNotFoundError, ValueError) as exc:
            return {"error": str(exc)}, 400

        def event_line(event: str, **payload: Any) -> str:
            return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"

        @stream_with_context
        def generate() -> Any:
            results: queue.Queue[tuple[str, str]] = queue.Queue()

            def run_answer() -> None:
                try:
                    document_path = Path(app.library.resolve_relative_path(rel_path))  # type: ignore[attr-defined]
                    arxiv_markdown = app.library.ensure_arxiv_markdown_for_rel_path(rel_path)  # type: ignore[attr-defined]
                    prompt_contexts = load_chat_prompt_contexts(rel_path, app.prompt_store.list_prompts())  # type: ignore[attr-defined]
                    for chunk in stream_answer_question_about_document(
                        document_path,
                        question=message_body.strip(),
                        visibility=visibility,
                        history=history,
                        prompt_contexts=prompt_contexts,
                        arxiv_markdown_path=(arxiv_markdown.markdown_path if arxiv_markdown is not None else None),
                        attachment_paths=app.team_store.attachment_paths_for_rel_path(rel_path),  # type: ignore[attr-defined]
                        model=runtime_config.llm_model,
                        session_id=f"paper-reader-chat-{assistant_message_id}",
                    ):
                        results.put(("partial", chunk))
                    results.put(("done", ""))
                except Exception as exc:
                    results.put(("error", str(exc)))

            worker = threading.Thread(target=run_answer, name=f"paper-reader-chat-stream-{assistant_message_id}", daemon=True)
            worker.start()

            yield event_line(
                "init",
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                context=serialize_chat_context(app.team_store.chat_context(rel_path, actor.id)),  # type: ignore[attr-defined]
            )
            yield event_line("status", assistant_message_id=assistant_message_id, text="Paper Bot 正在读取论文和历史对话。")

            partial = ""
            while True:
                try:
                    kind, value = results.get(timeout=2.0)
                except queue.Empty:
                    yield event_line("status", assistant_message_id=assistant_message_id, text="Paper Bot 正在生成回复。")
                    continue
                if kind == "partial":
                    partial += value
                    app.team_store.update_chat_message(  # type: ignore[attr-defined]
                        assistant_message_id,
                        body=partial,
                        status="pending",
                        model=runtime_config.llm_model,
                    )
                    yield event_line(
                        "partial",
                        assistant_message_id=assistant_message_id,
                        body=partial,
                        body_html=render_markdown(partial),
                    )
                    continue
                if kind == "error":
                    if "429" in value or "Too Many Requests" in value:
                        body = "Paper Bot 现在有点忙，刚刚碰到了速率限制。请等十几秒，再把这条问题发一次。"
                    else:
                        body = f"这次回复没成功：{value}"
                    app.team_store.update_chat_message(  # type: ignore[attr-defined]
                        assistant_message_id,
                        body=body,
                        status="failed",
                        model=runtime_config.llm_model,
                    )
                    yield event_line(
                        "error",
                        assistant_message_id=assistant_message_id,
                        message=body,
                        body_html=render_markdown(body),
                        context=serialize_chat_context(app.team_store.chat_context(rel_path, actor.id)),  # type: ignore[attr-defined]
                    )
                    return
                if not partial:
                    body = "这次回复没有生成内容，请稍后重试。"
                    app.team_store.update_chat_message(  # type: ignore[attr-defined]
                        assistant_message_id, body=body, status="failed", model=runtime_config.llm_model
                    )
                    yield event_line("error", assistant_message_id=assistant_message_id, message=body, body_html=render_markdown(body))
                    return
                app.team_store.update_chat_message(  # type: ignore[attr-defined]
                    assistant_message_id,
                    body=partial,
                    status="completed",
                    model=runtime_config.llm_model,
                )
                yield event_line(
                    "done",
                    assistant_message_id=assistant_message_id,
                    context=serialize_chat_context(app.team_store.chat_context(rel_path, actor.id)),  # type: ignore[attr-defined]
                )
                return

        return Response(generate(), mimetype="application/x-ndjson")

    @app.post("/tags/add")
    def tag_add_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        source_type = "official" if actor.role == "admin" else "manual"
        try:
            ensure_paper_metadata(rel_path)
            app.team_store.add_tag(  # type: ignore[attr-defined]
                rel_path,
                request.form.get("tag_name", ""),
                actor.id,
                source_type=source_type,
                is_locked=(actor.role == "admin"),
            )
            flash("标签已添加。", "success")
        except (ValueError, FileNotFoundError) as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/tags/remove")
    def tag_remove_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)
        try:
            ensure_paper_metadata(rel_path)
            tag_id = int(request.form.get("tag_id", "0") or 0)
            app.team_store.remove_tag(rel_path, tag_id, is_admin=(actor.role == "admin"), acting_user_id=actor.id)  # type: ignore[attr-defined]
            flash("标签已移除。", "success")
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/import-link")
    def import_link_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        target = request.form.get("import_target", "")
        recommendation_reason = request.form.get("recommendation_reason", "")
        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect_to_index(current_folder, query, sort_by, show_done=show_done)
        try:
            task = submit_remote_import_action(target, recommendation_reason, actor)
            flash(f"后台导入任务已开始：{task.title}。页面不会再卡住，导入结果会显示在“后台任务”里。", "success")
            return redirect_to_index(current_folder, query, sort_by, show_done=show_done)
        except Exception as exc:
            flash(f"导入没成功：{exc}", "error")
            return redirect_to_index(current_folder, query, sort_by, show_done=show_done)

    @app.post("/resource/retry")
    def resource_retry_route() -> Any:
        rel_path = request.form.get("rel_path", "").strip("/")
        actor = current_user()
        if actor is None:
            return {"error": "unauthorized"}, 401
        metadata = app.library.web_resources.metadata_for_rel_path(rel_path)  # type: ignore[attr-defined]
        if not metadata or not metadata.get("canonical_url"):
            flash("这个条目没有可重试的网页来源。", "error")
            return redirect_to_index("", "", "date_desc", rel_path or None, "source")
        try:
            task = submit_remote_import_action(str(metadata["canonical_url"]), "", actor)
            flash(f"网页重试任务已开始：{task.title}", "success")
        except Exception as exc:
            flash(f"网页重试没成功：{exc}", "error")
        return redirect_to_index("", "", "date_desc", rel_path or None, "source", show_done=True)

    @app.post("/prompt-batch-run")
    def prompt_batch_run_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        batch_show_done = parse_checkbox(request.form.get("batch_show_done"))
        batch_page = parse_page(request.form.get("batch_page"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        rel_paths = request.form.getlist("rel_paths")
        prompt_slugs = request.form.getlist("prompt_slugs")
        select_all_filtered = parse_checkbox(request.form.get("select_all_filtered"))
        force = parse_checkbox(request.form.get("force"))

        if select_all_filtered:
            rel_paths = [
                paper.rel_path
                for paper in build_batch_papers(
                    folder=current_folder,
                    query=query,
                    sort_by=sort_by,
                    show_done=show_done,
                    batch_show_done=batch_show_done,
                    selected_rel_path=(selected_paper or ""),
                )
            ]

        if not rel_paths:
            flash("先选至少一篇论文。", "error")
            return redirect(
                url_for(
                    "index",
                    folder=current_folder,
                    q=query,
                    sort=sort_by,
                    paper=selected_paper,
                    tab=tab,
                    show_done="1" if show_done else None,
                    batch_show_done="1" if batch_show_done else None,
                    batch_page=batch_page,
                )
            )
        if not prompt_slugs:
            flash("先选至少一份分析模板。", "error")
            return redirect(
                url_for(
                    "index",
                    folder=current_folder,
                    q=query,
                    sort=sort_by,
                    paper=selected_paper,
                    tab=tab,
                    show_done="1" if show_done else None,
                    batch_show_done="1" if batch_show_done else None,
                    batch_page=batch_page,
                )
            )

        actor = current_user()
        submission = app.job_queue.submit(  # type: ignore[attr-defined]
            rel_paths,
            prompt_slugs,
            force=force,
            source="batch",
            requested_by_user_id=actor.id if actor else None,
            requested_by_display_name=actor.display_name if actor else None,
        )
        flash_submission_summary(submission, action_label="批量生成已经开始")
        return redirect(
            url_for(
                "index",
                folder=current_folder,
                q=query,
                sort=sort_by,
                paper=(selected_paper or (rel_paths[0] if rel_paths else None)),
                tab=tab,
                show_done="1" if show_done else None,
                batch_show_done="1" if batch_show_done else None,
                batch_page=batch_page,
            )
        )

    @app.post("/offline-package")
    def offline_package_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        rel_paths = list(dict.fromkeys(path.strip("/") for path in request.form.getlist("rel_paths") if path.strip("/")))

        if not rel_paths:
            flash("先选至少一篇论文，再生成离线阅读包。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        scan = app.library.scan()  # type: ignore[attr-defined]
        app.team_store.sync_papers(scan.papers)  # type: ignore[attr-defined]
        paper_map = {paper.rel_path: paper for paper in apply_user_done_state(scan.papers)}  # type: ignore[attr-defined]
        selected_records: list[PaperRecord] = []
        for rel_path in rel_paths:
            paper = paper_map.get(rel_path)
            if paper is None or paper.is_done:
                continue
            selected_records.append(paper)

        if not selected_records:
            flash("当前选择中没有可导出的未完成论文。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        manifest = build_offline_manifest(app.library, app.prompt_store, selected_records)  # type: ignore[attr-defined]
        manifest_json = build_manifest_json(manifest)
        html = render_template("offline_reader.html", manifest_json=manifest_json)

        static_root = Path(app.static_folder or "")
        with tempfile.NamedTemporaryFile(prefix="paper-reader-offline-", suffix=".zip", delete=False) as handle:
            zip_path = Path(handle.name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("index.html", html.encode("utf-8"))
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            archive.writestr("assets/style.css", (static_root / "style.css").read_bytes())
            archive.writestr("assets/tokens.css", (static_root / "tokens.css").read_bytes())
            archive.writestr("assets/app.css", (static_root / "app.css").read_bytes())
            archive.writestr("assets/offline-reader.css", (static_root / "offline-reader.css").read_bytes())
            archive.writestr("assets/offline-reader.js", (static_root / "offline-reader.js").read_bytes())
            for font_name in (
                "Inter-400.woff2",
                "Inter-600.woff2",
                "Inter-700.woff2",
                "SpaceGrotesk-500.woff2",
                "SpaceGrotesk-700.woff2",
            ):
                archive.writestr(f"assets/fonts/{font_name}", (static_root / "fonts" / font_name).read_bytes())
            archive.writestr(
                "assets/vendor/mathjax/tex-svg.js",
                (static_root / "vendor" / "mathjax" / "tex-svg.js").read_bytes(),
            )

            for paper in selected_records:
                source_path = app.library.resolve_relative_path(paper.rel_path)  # type: ignore[attr-defined]
                archive.write(source_path, offline_source_arcname(paper.rel_path))
                for prompt_slug in app.library.list_existing_prompt_slugs(paper.rel_path):  # type: ignore[attr-defined]
                    result_path = app.library.existing_prompt_result_path(paper.rel_path, prompt_slug)  # type: ignore[attr-defined]
                    if result_path is None:
                        continue
                    archive.write(result_path, offline_prompt_arcname(paper.rel_path, prompt_slug))

        package_name = f"paper-reader-offline-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.zip"
        response = send_file(zip_path, as_attachment=True, download_name=package_name, mimetype="application/zip")
        response.call_on_close(lambda: zip_path.unlink(missing_ok=True))
        return response

    @app.get("/discover")
    def discover_index() -> Any:
        return render_template("discover.html")

    @app.get("/sources")
    def sources_index() -> Any:
        source_root = Path(app.config["SOURCE_ARCHIVE_ROOT"])
        day_records = load_source_days(source_root)
        source_groups = build_source_groups(day_records)
        return render_template(
            "sources.html",
            source_root=source_root,
            source_groups=source_groups,
            source_day_count=len(day_records),
            initial_action_snapshot=app.action_queue.snapshot(),  # type: ignore[attr-defined]
        )

    @app.get("/insights")
    def insights_index() -> Any:
        payload = app.history_store.load_history()  # type: ignore[attr-defined]
        status = app.history_store.status_snapshot()  # type: ignore[attr-defined]
        momentum_payload = app.momentum_store.load_dashboard()  # type: ignore[attr-defined]
        momentum_status = app.momentum_store.status_snapshot()  # type: ignore[attr-defined]
        opportunity_payload = app.opportunity_store.load_map()  # type: ignore[attr-defined]
        opportunity_status = app.opportunity_store.status_snapshot()  # type: ignore[attr-defined]
        return render_template(
            "insights.html",
            insights_payload=payload,
            insights_status=status,
            momentum_payload=momentum_payload,
            momentum_status=momentum_status,
            opportunity_payload=opportunity_payload,
            opportunity_status=opportunity_status,
        )

    @app.post("/insights/history/rebuild")
    def insights_history_rebuild_route() -> Any:
        started = app.history_store.start_or_resume()  # type: ignore[attr-defined]
        if started:
            flash("已开始/继续重建历史脉络；系统会在后台读取核心解读并持续保存进度。", "success")
        else:
            flash("历史脉络正在生成中，请稍后刷新页面。", "success")
        return redirect(url_for("insights_index"))

    @app.post("/insights/history/stop")
    def insights_history_stop_route() -> Any:
        stopped = app.history_store.request_stop()  # type: ignore[attr-defined]
        if stopped:
            flash("已请求停止历史脉络任务；当前阶段完成后会尽快停下。", "success")
        else:
            flash("当前没有正在运行的历史脉络任务。", "success")
        return redirect(url_for("insights_index"))

    @app.get("/insights/history/status")
    def insights_history_status_route() -> Any:
        return app.history_store.status_snapshot()  # type: ignore[attr-defined]

    @app.post("/insights/momentum/rebuild")
    def insights_momentum_rebuild_route() -> Any:
        started = app.momentum_store.start_or_resume()  # type: ignore[attr-defined]
        if started:
            flash("已开始/继续生成 Momentum Radar；系统会在后台读取核心解读并持续保存进度。", "success")
        else:
            flash("Momentum Radar 正在生成中，请稍后刷新页面。", "success")
        return redirect(url_for("insights_index"))

    @app.post("/insights/momentum/stop")
    def insights_momentum_stop_route() -> Any:
        stopped = app.momentum_store.request_stop()  # type: ignore[attr-defined]
        if stopped:
            flash("已请求停止 Momentum Radar 任务；当前阶段完成后会尽快停下。", "success")
        else:
            flash("当前没有正在运行的 Momentum Radar 任务。", "success")
        return redirect(url_for("insights_index"))

    @app.get("/insights/momentum/status")
    def insights_momentum_status_route() -> Any:
        return app.momentum_store.status_snapshot()  # type: ignore[attr-defined]

    @app.post("/insights/opportunity/rebuild")
    def insights_opportunity_rebuild_route() -> Any:
        started = app.opportunity_store.start_or_resume()  # type: ignore[attr-defined]
        if started:
            flash("已开始/继续生成 Opportunity Map；系统会在后台读取核心解读并持续保存进度。", "success")
        else:
            flash("Opportunity Map 正在生成中，请稍后刷新页面。", "success")
        return redirect(url_for("insights_index"))

    @app.post("/insights/opportunity/stop")
    def insights_opportunity_stop_route() -> Any:
        stopped = app.opportunity_store.request_stop()  # type: ignore[attr-defined]
        if stopped:
            flash("已请求停止 Opportunity Map 任务；当前阶段完成后会尽快停下。", "success")
        else:
            flash("当前没有正在运行的 Opportunity Map 任务。", "success")
        return redirect(url_for("insights_index"))

    @app.get("/insights/opportunity/status")
    def insights_opportunity_status_route() -> Any:
        return app.opportunity_store.status_snapshot()  # type: ignore[attr-defined]

    @app.get("/sources/open/<run_date>/<paper_id>")
    def source_pdf_route(run_date: str, paper_id: str) -> Any:
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            abort(404)
        paper = day_paper_map(day_record).get(paper_id)
        if paper is None:
            abort(404)
        source_pdf_path = local_pdf_path_for(day_record, paper)
        if source_pdf_path is None or not source_pdf_path.exists() or not source_pdf_path.is_file():
            abort(404)
        return send_file(source_pdf_path, as_attachment=False, download_name=paper.pdf_file_name or source_pdf_path.name)

    @app.post("/sources/download-zip")
    def source_download_zip_route() -> Any:
        run_date = request.form.get("run_date", "").strip()
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            flash("没有找到这一天的来源归档。", "error")
            return redirect(url_for("sources_index"))

        selected_papers = selected_source_papers(day_record, request.form.getlist("paper_ids"))
        valid_papers = []
        for paper in selected_papers:
            source_pdf_path = local_pdf_path_for(day_record, paper)
            if source_pdf_path is None or not source_pdf_path.exists() or not source_pdf_path.is_file():
                continue
            valid_papers.append((paper, source_pdf_path))

        if not valid_papers:
            flash("当前选择中没有可打包的本地 PDF。", "error")
            return redirect(url_for("sources_index"))

        with tempfile.NamedTemporaryFile(prefix="paper-reader-source-", suffix=".zip", delete=False) as handle:
            zip_path = Path(handle.name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(day_record.manifest_path, "manifest.json")
            for index, (paper, source_pdf_path) in enumerate(valid_papers, start=1):
                base_name = paper.pdf_file_name or f"{paper.paper_id}.pdf"
                archive_name = safe_download_name(
                    f"{index:02d}-{paper.paper_id}-{paper.title}.pdf",
                    fallback=base_name,
                )
                archive.write(source_pdf_path, archive_name)

        package_name = f"paper-reader-source-{day_record.run_date}.zip"
        response = send_file(zip_path, as_attachment=True, download_name=package_name, mimetype="application/zip")
        response.call_on_close(lambda: zip_path.unlink(missing_ok=True))
        return response

    @app.post("/sources/import")
    def source_import_route() -> Any:
        run_date = request.form.get("run_date", "").strip()
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            flash("没有找到这一天的来源归档。", "error")
            return redirect(url_for("sources_index"))

        selected = selected_source_papers(day_record, request.form.getlist("paper_ids"))
        if not selected:
            flash("先选至少一篇论文。", "error")
            return redirect(url_for("sources_index"))

        actor = current_user()
        if actor is None:
            flash("请先登录。", "error")
            return redirect(url_for("sources_index"))

        task = submit_source_import_action(day_record, selected, actor)
        flash(f"后台导入任务已开始：{task.title}。页面不会卡住，结果会显示在“后台任务”里。", "success")
        return redirect(url_for("sources_index"))

    @app.post("/ai-summary")
    def ai_summary_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        prompt = app.prompt_store.get_prompt(DEFAULT_PROMPT_SLUG)  # type: ignore[attr-defined]
        if prompt is None:
            prompt = app.prompt_store.save_prompt(  # type: ignore[attr-defined]
                existing_slug=None,
                name="核心解读",
                slug=DEFAULT_PROMPT_SLUG,
                user_prompt=DEFAULT_USER_PROMPT,
                model=request.form.get("model", DEFAULT_MODEL),
                enabled=True,
                auto_run=True,
            )
        submission = app.job_queue.submit([rel_path], [prompt.slug], force=True, source="legacy-ai-summary")  # type: ignore[attr-defined]
        flash_submission_summary(submission, action_label="这份核心解读已经开始生成")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, DEFAULT_PROMPT_SLUG, show_done=show_done)

    @app.get("/jobs/status")
    def jobs_status() -> Any:
        rel_path = request.args.get("paper", "").strip("/") or None
        return app.job_queue.snapshot(rel_path=rel_path)  # type: ignore[attr-defined]

    @app.get("/actions/status")
    def actions_status() -> Any:
        return app.action_queue.snapshot()  # type: ignore[attr-defined]

    @app.post("/jobs/config")
    def jobs_config_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        requested = request.form.get("max_concurrency", "")
        try:
            value = app.settings_store.save_max_concurrency(int(requested))  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            flash("同时处理数量要填 1 到 32 之间的整数。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        app.job_queue.update_max_concurrency(value)  # type: ignore[attr-defined]
        flash(f"同时处理数量已经更新为 {value}。", "success")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/jobs/stop-all")
    def jobs_stop_all_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        summary = app.job_queue.stop_all()  # type: ignore[attr-defined]
        interrupted = summary["queued"] + summary["running"]
        if interrupted:
            flash(f"已经请求停止 {interrupted} 个任务（正在处理 {summary['running']}，等待处理 {summary['queued']}）。", "success")
        else:
            flash("现在没有可停止的任务。", "success")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.get("/files/<path:rel_path>")
    def serve_file(rel_path: str) -> Any:
        rel_path = rel_path.strip("/")
        try:
            absolute = app.library.resolve_relative_path(rel_path)  # type: ignore[attr-defined]
        except ValueError:
            abort(404)
        if not absolute.exists() or not absolute.is_file():
            abort(404)
        safe_rel_path = absolute.relative_to(app.config["LIBRARY_ROOT"]).as_posix()
        return send_from_directory(app.config["LIBRARY_ROOT"], safe_rel_path, as_attachment=False)

    @app.post("/reindex")
    def reindex() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        admin_user = require_admin_user()
        if admin_user is None:
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)
        migrated_done_count = migrate_legacy_done_documents()
        active_records = app.library.rebuild_active_index(lightweight=True)  # type: ignore[attr-defined]
        done_records = app.library.rebuild_done_index(lightweight=True)  # type: ignore[attr-defined]
        migration_note = f"；另外还迁移了 {migrated_done_count} 篇旧版 DONE 论文到个人已读状态" if migrated_done_count else ""
        flash(
            f"论文列表已经快速同步：普通目录 {len(active_records)} 篇，旧版 DONE 目录 {len(done_records)} 篇{migration_note}；这次没有重新跑分析。",
            "success",
        )
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    return app
