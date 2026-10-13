"""Classification, extraction, and filesystem snapshots for web resources."""

from __future__ import annotations

import hashlib
import html
import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from trafilatura import extract, extract_metadata

from .arxiv_markdown import base_arxiv_id, normalize_arxiv_id
from .netutils import FetchedResponse, fetch_public_url

WEB_ROOT_NAME = "Web"
WEB_METADATA_SUFFIX = ".json"
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_NAMES = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref_src"})
_NEWS_DOMAINS = frozenset(
    {
        "bbc.com",
        "bbc.co.uk",
        "bloomberg.com",
        "cnn.com",
        "ft.com",
        "nytimes.com",
        "reuters.com",
        "technologyreview.com",
        "theguardian.com",
        "venturebeat.com",
        "wired.com",
    }
)
_PROJECT_DOMAINS = frozenset({"github.com", "gitlab.com", "huggingface.co", "replicate.com"})
_PROJECT_PATH_RE = re.compile(r"(?:^|[-_/])(project|projects|demo|demos|showcase|repository|repo)(?:[-_/]|$)", re.IGNORECASE)


class WebExtractionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ImportTarget:
    raw_url: str
    canonical_url: str
    route: str
    resource_kind: str
    source_type: str
    source_value: str
    source_url: str
    download_url: str | None = None
    target_folder: str = WEB_ROOT_NAME
    preferred_name: str | None = None


@dataclass(frozen=True, slots=True)
class WebSnapshot:
    rel_path: str
    metadata_rel_path: str
    canonical_url: str
    original_url: str
    final_url: str | None
    resource_kind: str
    title: str
    author: str | None
    published_date: str | None
    site_name: str | None
    source_domain: str
    fetch_status: str
    fetched_at: str
    error: str | None


def canonicalize_url(url: str) -> str:
    value = str(url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A full HTTP or HTTPS URL is required.")
    scheme = parsed.scheme.casefold()
    host = parsed.hostname.rstrip(".").casefold()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid.") from exc
    if scheme == "http" and port in {None, 80}:
        scheme = "https"
        port = None
    if port is not None and port != (443 if scheme == "https" else 80):
        host = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    elif ":" in host:
        host = f"[{host}]"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_QUERY_NAMES
        and not any(key.casefold().startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES)
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, host, path, query, ""))


def source_domain_for(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").casefold()


def classify_web_kind(url: str) -> str:
    domain = source_domain_for(url)
    registrable_hint = ".".join(domain.split(".")[-2:]) if domain.count(".") else domain
    if domain in _PROJECT_DOMAINS or registrable_hint in _PROJECT_DOMAINS or _PROJECT_PATH_RE.search(urlsplit(url).path):
        return "project"
    if domain in _NEWS_DOMAINS or registrable_hint in _NEWS_DOMAINS or "/news/" in urlsplit(url).path.casefold():
        return "news"
    return "article"


def _arxiv_target(raw: str, arxiv_id: str) -> ImportTarget:
    stable_id = base_arxiv_id(arxiv_id)
    canonical = f"https://arxiv.org/abs/{stable_id}"
    return ImportTarget(
        raw_url=raw,
        canonical_url=canonical,
        route="pdf",
        resource_kind="paper",
        source_type="arxiv",
        source_value=stable_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        download_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        target_folder="Imports/arXiv",
        preferred_name=f"{stable_id.replace('/', '-')}.pdf",
    )


def classify_import_target(raw_target: str) -> ImportTarget:
    target = str(raw_target or "").strip()
    if not target:
        raise ValueError("Please provide an arXiv ID or HTTP/HTTPS URL.")

    arxiv_id = normalize_arxiv_id(target)
    parsed_candidate = urlsplit(target)
    if arxiv_id and (not parsed_candidate.scheme or (parsed_candidate.hostname or "").casefold().endswith("arxiv.org")):
        return _arxiv_target(target, arxiv_id)

    if parsed_candidate.scheme.casefold() not in {"http", "https"} or not parsed_candidate.hostname:
        raise ValueError("Only arXiv IDs and full HTTP/HTTPS URLs are supported.")
    canonical = canonicalize_url(target)
    host = (parsed_candidate.hostname or "").casefold()
    path = parsed_candidate.path or ""
    query = dict(parse_qsl(parsed_candidate.query, keep_blank_values=True))

    if host.endswith("openreview.net"):
        openreview_id = str(query.get("id") or "").strip()
        if openreview_id and path.rstrip("/") in {"/forum", "/pdf"}:
            canonical_landing = f"https://openreview.net/forum?id={openreview_id}"
            return ImportTarget(
                raw_url=target,
                canonical_url=canonical_landing,
                route="pdf",
                resource_kind="paper",
                source_type="openreview",
                source_value=openreview_id,
                source_url=canonical_landing,
                download_url=f"https://openreview.net/pdf?id={openreview_id}",
                target_folder="Imports/OpenReview",
                preferred_name=f"{_safe_slug(openreview_id, fallback='openreview-paper')}.pdf",
            )

    if path.casefold().endswith(".pdf"):
        preferred_name = Path(path).name or "paper.pdf"
        return ImportTarget(
            raw_url=target,
            canonical_url=canonical,
            route="pdf",
            resource_kind="paper",
            source_type="pdf_url",
            source_value=canonical,
            source_url=canonical,
            download_url=target,
            target_folder="Imports/Links",
            preferred_name=preferred_name,
        )

    return ImportTarget(
        raw_url=target,
        canonical_url=canonical,
        route="web",
        resource_kind=classify_web_kind(canonical),
        source_type="canonical_url",
        source_value=canonical,
        source_url=canonical,
    )


def _safe_slug(value: str, *, fallback: str = "resource") -> str:
    ascii_value = value.encode("ascii", "ignore").decode("ascii").casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return (slug[:80].rstrip("-") or fallback)


class _FallbackTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._title_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._title_depth += 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "br"} and not self._skip_depth:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title" and self._title_depth:
            self._title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        if self._title_depth:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


def _fallback_extract(html_text: str) -> tuple[str, str]:
    parser = _FallbackTextExtractor()
    parser.feed(html_text)
    title = " ".join(parser.title_parts).strip()
    text = "\n\n".join(
        line.strip() for line in re.sub(r"\n\s*\n+", "\n", " ".join(parser.text_parts)).splitlines() if line.strip()
    )
    return title, html.unescape(text).strip()


def extract_web_content(response: FetchedResponse, *, resource_kind: str) -> tuple[str, dict[str, str | None]]:
    content_type = response.content_type
    if content_type and not (content_type in {"text/html", "application/xhtml+xml"} or content_type.startswith("text/")):
        raise WebExtractionError(f"Expected an HTML page, received {content_type}.")
    html_text = response.body.decode("utf-8", errors="replace")
    metadata_obj = extract_metadata(html_text, default_url=response.final_url)
    metadata = {
        "title": str(getattr(metadata_obj, "title", "") or "").strip() or None,
        "author": str(getattr(metadata_obj, "author", "") or "").strip() or None,
        "date": str(getattr(metadata_obj, "date", "") or "").strip() or None,
        "sitename": str(getattr(metadata_obj, "sitename", "") or "").strip() or None,
    }
    markdown = extract(
        html_text,
        url=response.final_url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        with_metadata=False,
        favor_recall=(resource_kind == "project"),
    )
    fallback_title, fallback_text = _fallback_extract(html_text)
    content = str(markdown or "").strip() or fallback_text
    if len(re.sub(r"\s+", " ", content)) < 40:
        raise WebExtractionError("No meaningful main content could be extracted from the page.")
    metadata["title"] = metadata["title"] or fallback_title or None
    return content, metadata


def _published_sort_date(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else None


def _resource_markdown(*, title: str, metadata: dict[str, object], content: str) -> str:
    lines = [f"# {title}", ""]
    author = str(metadata.get("author") or "").strip()
    published_date = str(metadata.get("published_date") or "").strip()
    site_name = str(metadata.get("site_name") or "").strip()
    if author:
        lines.append(f"**Author:** {author}")
    if published_date:
        lines.append(f"**Published:** {published_date}")
    if site_name:
        lines.append(f"**Site:** {site_name}")
    lines.extend([f"**Original:** {metadata['canonical_url']}", "", "---", "", content.strip(), ""])
    return "\n".join(lines)


def _failed_markdown(*, title: str, url: str, status: str, error: str) -> str:
    return (
        f"# {title}\n\n"
        f"**Status:** {status}\n\n"
        f"**Original:** {url}\n\n"
        "The page snapshot could not be extracted. Use **Retry fetch** after the source becomes available.\n\n"
        f"Technical detail: `{error}`\n"
    )


class WebResourceStore:
    def __init__(self, library_root: Path, *, fetch_max_bytes: int = 25 * 1024 * 1024):
        self.library_root = library_root.resolve()
        self.fetch_max_bytes = fetch_max_bytes
        self.web_root = self.library_root / WEB_ROOT_NAME
        self.web_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def metadata_path_for(self, markdown_path: Path) -> Path:
        return markdown_path.with_suffix(WEB_METADATA_SUFFIX)

    def metadata_for_rel_path(self, rel_path: str) -> dict[str, object] | None:
        path = (self.library_root / rel_path).resolve()
        if self.library_root not in path.parents or path.suffix.casefold() != ".md":
            return None
        metadata_path = self.metadata_path_for(path)
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) and payload.get("resource_kind") else None

    def _path_for(self, target: ImportTarget, *, existing_rel_path: str | None = None, title: str | None = None) -> Path:
        if existing_rel_path:
            path = (self.library_root / existing_rel_path).resolve()
            if self.library_root not in path.parents:
                raise ValueError("Existing resource path escapes the library root.")
            return path
        domain = _safe_slug(source_domain_for(target.canonical_url), fallback="web")
        source_hint = title or Path(urlsplit(target.canonical_url).path).stem or domain
        digest = hashlib.sha256(target.canonical_url.encode("utf-8")).hexdigest()[:12]
        return self.web_root / domain / f"{_safe_slug(source_hint)}-{digest}.md"

    def _write(self, path: Path, *, markdown: str, metadata: dict[str, object]) -> WebSnapshot:
        metadata_path = self.metadata_path_for(path)
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            markdown_temp = path.with_name(f".{path.name}.tmp")
            metadata_temp = metadata_path.with_name(f".{metadata_path.name}.tmp")
            markdown_temp.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            markdown_temp.replace(path)
            metadata_temp.replace(metadata_path)
        rel_path = path.relative_to(self.library_root).as_posix()
        return WebSnapshot(
            rel_path=rel_path,
            metadata_rel_path=metadata_path.relative_to(self.library_root).as_posix(),
            canonical_url=str(metadata["canonical_url"]),
            original_url=str(metadata["original_url"]),
            final_url=(str(metadata["final_url"]) if metadata.get("final_url") else None),
            resource_kind=str(metadata["resource_kind"]),
            title=str(metadata["title"]),
            author=(str(metadata["author"]) if metadata.get("author") else None),
            published_date=(str(metadata["published_date"]) if metadata.get("published_date") else None),
            site_name=(str(metadata["site_name"]) if metadata.get("site_name") else None),
            source_domain=str(metadata["source_domain"]),
            fetch_status=str(metadata["fetch_status"]),
            fetched_at=str(metadata["fetched_at"]),
            error=(str(metadata["error"]) if metadata.get("error") else None),
        )

    def fetch_and_store(self, target: ImportTarget, *, existing_rel_path: str | None = None) -> WebSnapshot:
        if target.route != "web":
            raise ValueError("Only web targets can be snapshot-extracted.")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        title = source_domain_for(target.canonical_url) or "Web resource"
        final_url: str | None = None
        author: str | None = None
        published_date: str | None = None
        site_name: str | None = None
        error: str | None = None
        content = ""
        status = "ok"
        try:
            response = fetch_public_url(target.raw_url, max_bytes=self.fetch_max_bytes)
            final_url = response.final_url
            content, extracted = extract_web_content(response, resource_kind=target.resource_kind)
            title = str(extracted.get("title") or title).strip()
            author = str(extracted.get("author") or "").strip() or None
            published_date = str(extracted.get("date") or "").strip() or None
            site_name = str(extracted.get("sitename") or "").strip() or None
        except Exception as exc:
            error = str(exc)
            lowered = error.casefold()
            status = "paywalled" if "http 401" in lowered or "http 402" in lowered or "http 403" in lowered else "failed"

        path = self._path_for(target, existing_rel_path=existing_rel_path, title=title)
        metadata: dict[str, object] = {
            "schema_version": 1,
            "canonical_url": target.canonical_url,
            "original_url": target.raw_url,
            "final_url": final_url,
            "resource_kind": target.resource_kind,
            "title": title,
            "author": author,
            "published_date": published_date,
            "sort_date": _published_sort_date(published_date),
            "site_name": site_name,
            "source_domain": source_domain_for(final_url or target.canonical_url),
            "fetch_status": status,
            "fetched_at": fetched_at,
            "error": error,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None,
            "content_chars": len(content),
        }
        markdown = (
            _resource_markdown(title=title, metadata=metadata, content=content)
            if status == "ok"
            else _failed_markdown(title=title, url=target.canonical_url, status=status, error=error or "unknown error")
        )
        return self._write(path, markdown=markdown, metadata=metadata)


def snapshot_asdict(snapshot: WebSnapshot) -> dict[str, object]:
    return asdict(snapshot)
