"""Daily, interest-aware paper discovery for the shared reader inbox."""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .arxiv_markdown import base_arxiv_id, normalize_arxiv_id
from .config import PaperReaderConfig
from .llm import get_llm_client
from .netutils import fetch_public_url
from .team_store import TeamStore

LOGGER = logging.getLogger(__name__)
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
HUGGING_FACE_PAPERS_URL = "https://huggingface.co/papers"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("cs.RO", "cs.CV", "cs.LG", "cs.AI")
ARXIV_PER_CATEGORY = 10
SCORING_BATCH_SIZE = 20
PROFILE_CACHE_NAME = "discovery_interest_profile.json"
PROFILE_CACHE_TTL = timedelta(hours=24)
_DAILY_PAPERS_PROPS_RE = re.compile(r'data-target="DailyPapers"\s+data-props="([^"]+)"')
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class CompletionClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, model: str, session_id: str | None = None) -> str: ...


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    source: str
    external_id: str
    title: str
    url: str
    abstract: str
    published_at: str | None = None

    @property
    def arxiv_id(self) -> str | None:
        return normalize_arxiv_id(self.url) or normalize_arxiv_id(self.external_id)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: DiscoveryCandidate
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryRunResult:
    fetched: int
    eligible: int
    surfaced: int
    profile: str


def _text(element: ET.Element | None) -> str:
    return " ".join((element.text or "").split()) if element is not None else ""


def _canonical_arxiv_url(raw_id: str) -> tuple[str, str] | None:
    normalized = normalize_arxiv_id(raw_id)
    if normalized is None:
        return None
    base_id = base_arxiv_id(normalized)
    return base_id, f"https://arxiv.org/abs/{base_id}"


def parse_huggingface_daily(html_text: str) -> list[DiscoveryCandidate]:
    """Parse HF's embedded DailyPapers JSON without issuing another request."""
    match = _DAILY_PAPERS_PROPS_RE.search(html_text)
    if match is None:
        raise ValueError("Hugging Face DailyPapers payload was not found.")
    try:
        payload = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as exc:
        raise ValueError("Hugging Face DailyPapers payload was invalid JSON.") from exc

    candidates: list[DiscoveryCandidate] = []
    for item in payload.get("dailyPapers", []):
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        raw_id = str(paper.get("id") or item.get("id") or "").strip()
        title = str(item.get("title") or paper.get("title") or "").strip()
        if not raw_id or not title:
            continue
        canonical = _canonical_arxiv_url(raw_id)
        external_id, url = canonical if canonical else (raw_id, f"{HUGGING_FACE_PAPERS_URL}/{quote(raw_id, safe='')}")
        candidates.append(
            DiscoveryCandidate(
                source="huggingface",
                external_id=external_id,
                title=title,
                url=url,
                abstract=str(item.get("summary") or paper.get("summary") or "").strip(),
                published_at=str(paper.get("publishedAt") or item.get("publishedAt") or "").strip() or None,
            )
        )
    return candidates


def fetch_huggingface_daily() -> list[DiscoveryCandidate]:
    response = fetch_public_url(HUGGING_FACE_PAPERS_URL, accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1")
    return parse_huggingface_daily(response.body.decode("utf-8", "replace"))


def parse_arxiv_atom(xml_text: str, *, category: str) -> list[DiscoveryCandidate]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"arXiv {category} returned malformed Atom XML.") from exc
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    candidates: list[DiscoveryCandidate] = []
    for entry in root.findall("atom:entry", namespace):
        raw_url = _text(entry.find("atom:id", namespace))
        title = _text(entry.find("atom:title", namespace))
        if not raw_url or not title:
            continue
        canonical = _canonical_arxiv_url(raw_url)
        if canonical is None:
            continue
        arxiv_id, url = canonical
        candidates.append(
            DiscoveryCandidate(
                source=f"arxiv:{category}",
                external_id=arxiv_id,
                title=title,
                url=url,
                abstract=_text(entry.find("atom:summary", namespace)),
                published_at=_text(entry.find("atom:published", namespace)) or None,
            )
        )
    return candidates


def fetch_arxiv_category(category: str) -> list[DiscoveryCandidate]:
    query = (
        f"{ARXIV_API_URL}?search_query={quote(f'cat:{category}', safe=':')}&start=0"
        f"&max_results={ARXIV_PER_CATEGORY}&sortBy=submittedDate&sortOrder=descending"
    )
    response = fetch_public_url(query, accept="application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1")
    return parse_arxiv_atom(response.body.decode("utf-8", "replace"), category=category)


def parse_scoring_output(output: str, candidates: Iterable[DiscoveryCandidate]) -> dict[str, tuple[float, str]]:
    """Accept a strict JSON response, retaining only ids supplied in this batch."""
    expected = {candidate.external_id for candidate in candidates}
    payloads = [output.strip(), *[block.strip() for block in _CODE_FENCE_RE.findall(output) if block.strip()]]
    parsed: Any = None
    for payload in payloads:
        try:
            parsed = json.loads(payload)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(parsed, dict):
        parsed = parsed.get("items") or parsed.get("scores") or []
    if not isinstance(parsed, list):
        raise ValueError("Discovery scorer did not return a JSON list.")

    scores: dict[str, tuple[float, str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id") or item.get("external_id") or "").strip()
        if identifier not in expected:
            continue
        try:
            score = max(0.0, min(10.0, float(item.get("score"))))
        except (TypeError, ValueError):
            continue
        reason = " ".join(str(item.get("reason") or "").split())[:180]
        if reason:
            scores[identifier] = (score, reason)
    return scores


class DiscoveryService:
    def __init__(
        self,
        *,
        store: TeamStore,
        config: PaperReaderConfig,
        client: CompletionClient | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.client = client
        self.cache_path = cache_path or (config.data_dir / PROFILE_CACHE_NAME)

    def interest_profile(self, *, force: bool = False) -> str:
        if not force:
            cached = self._load_cached_profile()
            if cached:
                return cached
        rows = self.store.library_interest_rows()
        compact_library = [
            {"title": row["title"][:220], "tags": row["tags"][:12], "kind": row["kind"]}
            for row in rows
        ]
        if not compact_library:
            profile = "实验室当前书库为空；优先关注机器人、视觉、机器学习与通用人工智能中具有明确方法贡献的研究。"
        else:
            prompt = (
                "你是 DexGEM Lab 的研究编辑。根据下面书库中保留条目的标题、标签和类型，写一份中文兴趣画像，"
                "用于筛选每日新论文。只描述反复出现的研究主题、方法偏好、应用边界和不感兴趣的泛化方向；"
                "不要列举每篇论文。严格不超过 500 tokens。\n\n书库信号：\n"
                + json.dumps(compact_library, ensure_ascii=False, separators=(",", ":"))
            )
            try:
                profile = self._client().complete(
                    [{"role": "user", "content": prompt}],
                    model=self.config.llm_cheap_model,
                    session_id="paper-reader-discovery-profile",
                ).strip()
            except Exception as exc:
                LOGGER.warning("Discovery interest profile fallback: %s", exc)
                profile = self._heuristic_profile(rows)
        profile = profile[:2000].strip()
        self._write_cached_profile(profile)
        return profile

    def collect_candidates(self) -> list[DiscoveryCandidate]:
        candidates = fetch_huggingface_daily()
        for category in ARXIV_CATEGORIES:
            candidates.extend(fetch_arxiv_category(category))
        return candidates

    def eligible_candidates(self, candidates: Iterable[DiscoveryCandidate]) -> list[DiscoveryCandidate]:
        eligible: list[DiscoveryCandidate] = []
        seen_urls: set[str] = set()
        for candidate in candidates:
            if candidate.url in seen_urls:
                continue
            seen_urls.add(candidate.url)
            if self.store.discovery_exists(source=candidate.source, external_id=candidate.external_id, url=candidate.url):
                continue
            if self.store.discovery_is_in_library(url=candidate.url, arxiv_id=candidate.arxiv_id):
                continue
            eligible.append(candidate)
        return eligible

    def score_candidates(self, candidates: list[DiscoveryCandidate], profile: str) -> list[ScoredCandidate]:
        scored: list[ScoredCandidate] = []
        for start in range(0, len(candidates), SCORING_BATCH_SIZE):
            batch = candidates[start : start + SCORING_BATCH_SIZE]
            payload = [
                {"id": item.external_id, "title": item.title, "abstract": item.abstract[:1200], "source": item.source}
                for item in batch
            ]
            prompt = (
                "你是 DexGEM Lab 的论文推荐编辑。根据兴趣画像，为每篇候选论文给出相关度 0-10 和一行中文理由。"
                "理由必须指出它和画像的具体交集或关键偏离，避免空泛褒奖。严格只返回 JSON 数组，"
                "每项为 {\"id\":...,\"score\":0-10,\"reason\":\"...\"}。\n\n兴趣画像：\n"
                f"{profile}\n\n候选：\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            )
            try:
                returned = self._client().complete(
                    [{"role": "user", "content": prompt}],
                    model=self.config.llm_cheap_model,
                    session_id="paper-reader-discovery-score",
                )
                parsed = parse_scoring_output(returned, batch)
            except Exception as exc:
                LOGGER.warning("Discovery scoring failed for batch %s: %s", start // SCORING_BATCH_SIZE + 1, exc)
                parsed = {}
            for item in batch:
                score, reason = parsed.get(item.external_id, (0.0, "与当前兴趣画像的匹配度尚未确认。"))
                scored.append(ScoredCandidate(candidate=item, score=score, reason=reason))
        return scored

    def run(self) -> DiscoveryRunResult:
        profile = self.interest_profile()
        fetched = self.collect_candidates()
        eligible = self.eligible_candidates(fetched)
        scored = self.score_candidates(eligible, profile)
        surfaced_at = datetime.now(BEIJING_TZ).isoformat(timespec="seconds")
        inserted = self.store.add_discoveries(
            [
                {
                    **asdict(entry.candidate),
                    "score": entry.score,
                    "reason": entry.reason,
                    "surfaced_at": surfaced_at,
                }
                for entry in scored
            ]
        )
        return DiscoveryRunResult(fetched=len(fetched), eligible=len(eligible), surfaced=inserted, profile=profile)

    def _client(self) -> CompletionClient:
        return self.client or get_llm_client(self.config)

    def _load_cached_profile(self) -> str | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(payload.get("created_at") or ""))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created_at > PROFILE_CACHE_TTL:
                return None
            profile = str(payload.get("profile") or "").strip()
            return profile or None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_cached_profile(self, profile: str) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "profile": profile}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    @staticmethod
    def _heuristic_profile(rows: list[dict[str, Any]]) -> str:
        tags: list[str] = []
        for row in rows:
            tags.extend(str(tag) for tag in row["tags"])
        recurring = list(dict.fromkeys(tags))[:16]
        label = "、".join(recurring) if recurring else "近期书库主题"
        return f"实验室近期持续关注：{label}。优先推荐有清晰技术贡献、可复现实验或研究系统价值的论文。"


class DailyDiscoveryScheduler:
    """Small in-process scheduler matching the reader's existing local queues."""

    def __init__(self, service: DiscoveryService, *, hour: int, poll_seconds: float = 60.0) -> None:
        self.service = service
        self.hour = hour
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run_date: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._serve, name="paper-reader-discovery", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def should_run(self, now: datetime) -> bool:
        local = now.astimezone(BEIJING_TZ)
        return local.hour >= self.hour and self._last_run_date != local.date().isoformat()

    def run_if_due(self, now: datetime | None = None) -> DiscoveryRunResult | None:
        now = now or datetime.now(BEIJING_TZ)
        if not self.should_run(now):
            return None
        result = self.service.run()
        self._last_run_date = now.astimezone(BEIJING_TZ).date().isoformat()
        return result

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_if_due()
            except Exception:  # pragma: no cover - operational boundary
                LOGGER.exception("Daily discovery run failed")
            self._stop.wait(self.poll_seconds)
