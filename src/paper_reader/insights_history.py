from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ai_summary import run_text_prompt

INSIGHTS_DIR_NAME = ".paper-reader-insights"
HISTORY_CACHE_FILE_NAME = "history_timeline.json"
HISTORY_STATUS_FILE_NAME = "history_status.json"
CORE_PROMPT_SLUG = "core-zh"
MIN_THEME_PAPERS = 5
MAX_THEME_COUNT = 6
MAX_THEME_PREVALENCE = 0.55
MAX_CONTEXT_PAPERS = 24
MAX_YEAR_PICKS = 3
SCAN_PROGRESS = 8
THEME_SELECTION_PROGRESS = 16
WRITE_PROGRESS = 98
SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])\s+|\n+")
_ASCII_TERM_RE = re.compile(r"^[a-z0-9][a-z0-9 ._+/-]*$")


@dataclass(slots=True)
class HistoryPaper:
    rel_path: str
    title: str
    sort_date: str | None
    summary: str
    methods: list[str]
    signals: list[str]
    novelty_score: int
    turning_score: int

    @property
    def year(self) -> str:
        if self.sort_date and len(self.sort_date) >= 4:
            return self.sort_date[:4]
        return "unknown"


@dataclass(frozen=True, slots=True)
class ThemeDefinition:
    slug: str
    label: str
    description: str
    keywords: tuple[str, ...]


THEMES = (
    ThemeDefinition("agent", "Agent", "从工具调用走向长期任务和真实工作流的智能体路线。", ("agent", "agents", "智能体", "web agent", "browser agent", "assistant agent", "tool use")),
    ThemeDefinition("benchmark", "Benchmark/Evaluation", "评测从静态问答转向真实环境、长期交互和可执行验证。", ("benchmark", "arena", "bench", "evaluation", "评测", "测评", "executable checks", "workspace")),
    ThemeDefinition("memory", "Memory", "长期上下文、记忆管理、外部记忆和历史经验回放。", ("memory", "long-horizon", "context management", "记忆", "长期上下文", "externalization")),
    ThemeDefinition("reasoning", "Reasoning", "推理、test-time compute、规划和长链路决策。", ("reasoning", "推理", "test-time", "chain-of-thought", "planning", "planner")),
    ThemeDefinition("alignment", "Alignment/Safety", "对齐、安全、reward hacking 和高权限行为风险。", ("alignment", "misalignment", "reward hacking", "safety", "对齐", "安全", "constitutional")),
    ThemeDefinition("rl", "Reinforcement Learning", "RLHF、RLOO、DPO、策略优化和经验驱动改进。", ("reinforcement learning", "rlhf", "rloo", "dpo", "policy", "强化学习")),
    ThemeDefinition("multimodal", "Multimodal/VLM", "视觉语言、多模态理解、视频与跨模态任务。", ("multimodal", "vision-language", "vlm", "video", "视觉语言", "多模态")),
    ThemeDefinition("robotics", "Embodied/Robotics", "具身智能、机器人、VLA 与真实控制。", ("embodied", "robotics", "robot", "vla", "具身", "机器人")),
    ThemeDefinition("world_model", "World Model/Simulation", "环境建模、行为模拟、世界模型和虚拟用户。", ("world model", "simulation", "simulator", "行为模拟", "世界模型", "life simulator")),
    ThemeDefinition("data", "Data/Dataset", "数据工程、数据集构建、synthetic data 和 benchmark dataset。", ("dataset", "data engine", "data curation", "synthetic data", "数据集", "数据引擎")),
    ThemeDefinition("diffusion", "Diffusion/Generation", "扩散模型、图像/视频生成及其控制能力。", ("diffusion", "text-to-image", "text-to-video", "stable diffusion", "video diffusion")),
    ThemeDefinition("interpretability", "Interpretability", "机制解释、稀疏特征、可解释性和内部表征。", ("interpretability", "sparse autoencoder", "feature", "monosemantic", "可解释", "机制解释")),
)

METHOD_DEFINITIONS = {
    "skill_library": ("skill library", "skills", "skill injection", "技能库", "procedural skill"),
    "planning": ("planning", "planner", "plan", "规划"),
    "retrieval": ("retrieval", "retriever", "rag", "检索"),
    "memory": ("memory", "context management", "记忆"),
    "verification": ("verification", "executable checks", "grounded", "验证"),
    "benchmark_design": ("benchmark", "arena", "evaluation setup", "评测设计"),
    "distillation": ("distillation", "蒸馏"),
    "synthetic_data": ("synthetic data", "合成数据", "generated data"),
    "workflow": ("workflow", "pipeline", "tool workflow", "工作流"),
    "self_evolution": ("self-evolving", "collective evolution", "自我演化"),
}

SIGNAL_DEFINITIONS = {
    "foundation": ("we introduce", "we present", "we propose", "提出", "首次", "benchmark", "arena", "framework"),
    "turning": ("real-world", "dynamic", "belief revision", "long-horizon", "个性化", "externalization", "迁移"),
    "mainstream": ("widely used", "become mainstream", "主流", "follow", "adopted"),
}


class HistoricalInsightsStore:
    def __init__(self, library_root: Path):
        self.library_root = library_root.resolve()
        self.cache_dir = self.library_root / INSIGHTS_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_cache_path = self.cache_dir / HISTORY_CACHE_FILE_NAME
        self.status_path = self.cache_dir / HISTORY_STATUS_FILE_NAME
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def load_history(self) -> dict[str, Any] | None:
        payload = self._load_json(self.history_cache_path)
        return payload if payload else None

    def status_snapshot(self) -> dict[str, Any]:
        payload = self._load_json(self.status_path)
        if not payload:
            return self._default_status()

        state = str(payload.get("state") or "idle")
        if state in {"running", "stopping"} and not self._is_running():
            # The browser may have closed or the service may have restarted. Mark the task resumable.
            payload["state"] = "stopped"
            payload["message"] = payload.get("message") or "任务已中断，可继续生成历史脉络。"
            payload["updated_at"] = _timestamp()
            payload["stop_requested"] = False
            self._write_json(self.status_path, payload)

        payload.setdefault("progress", 0)
        payload.setdefault("current_theme_index", 0)
        payload.setdefault("total_themes", 0)
        payload.setdefault("completed_themes", 0)
        payload.setdefault("stage", "idle")
        payload.setdefault("message", "")
        payload.setdefault("error", None)
        payload.setdefault("scanned_core_papers", 0)
        payload.setdefault("total_core_papers", 0)
        payload["cache_exists"] = self.history_cache_path.exists()
        payload["thread_alive"] = self._is_running()
        payload["can_stop"] = payload["state"] in {"running", "stopping"}
        payload["can_continue"] = payload["state"] in {"stopped", "failed"}
        payload["can_start"] = payload["state"] in {"idle", "ready"} and not self._is_running()
        return payload

    def start_or_resume(self, *, model: str | None = None, max_themes: int = MAX_THEME_COUNT) -> bool:
        with self._lock:
            if self._is_running():
                return False
            now = _timestamp()
            existing = self._load_json(self.status_path)
            started_at = str(existing.get("started_at") or now)
            progress = int(existing.get("progress") or 0) if existing.get("state") in {"stopped", "failed"} else 0
            completed = int(existing.get("completed_themes") or 0) if existing.get("state") in {"stopped", "failed"} else 0
            total = int(existing.get("total_themes") or 0) if existing.get("state") in {"stopped", "failed"} else 0
            current_index = int(existing.get("current_theme_index") or completed) if existing.get("state") in {"stopped", "failed"} else 0
            self._stop_requested = False
            self._write_json(
                self.status_path,
                {
                    "state": "running",
                    "stage": existing.get("stage") if existing.get("state") in {"stopped", "failed"} else "scan",
                    "started_at": started_at,
                    "updated_at": now,
                    "message": "正在读取核心解读并重建历史脉络。",
                    "error": None,
                    "progress": progress,
                    "current_theme_index": current_index,
                    "current_theme_slug": existing.get("current_theme_slug"),
                    "current_theme_label": existing.get("current_theme_label"),
                    "completed_themes": completed,
                    "total_themes": total,
                    "scanned_core_papers": int(existing.get("scanned_core_papers") or 0),
                    "total_core_papers": int(existing.get("total_core_papers") or 0),
                    "stop_requested": False,
                    "model": model,
                },
            )
            self._thread = threading.Thread(
                target=self._rebuild_worker,
                kwargs={"model": model, "max_themes": max_themes},
                daemon=True,
                name="paper-reader-history-insights",
            )
            self._thread.start()
            return True

    def request_stop(self) -> bool:
        with self._lock:
            if not self._is_running():
                return False
            self._stop_requested = True
            status = self._load_json(self.status_path)
            status["state"] = "stopping"
            status["stop_requested"] = True
            status["updated_at"] = _timestamp()
            status["message"] = "正在停止历史脉络生成任务..."
            self._write_json(self.status_path, status)
        return True

    def rebuild_sync(self, *, model: str | None = None, max_themes: int = MAX_THEME_COUNT) -> dict[str, Any]:
        payload = build_history_payload(self.library_root, model=model, max_themes=max_themes)
        self._write_json(self.history_cache_path, payload)
        self._write_json(
            self.status_path,
            {
                "state": "ready",
                "stage": "complete",
                "started_at": _timestamp(),
                "updated_at": _timestamp(),
                "message": "历史脉络已生成。",
                "error": None,
                "progress": 100,
                "current_theme_index": payload.get("theme_count", 0),
                "completed_themes": payload.get("theme_count", 0),
                "total_themes": payload.get("theme_count", 0),
                "scanned_core_papers": payload.get("source_paper_count", 0),
                "total_core_papers": payload.get("source_paper_count", 0),
                "stop_requested": False,
            },
        )
        return payload

    def _is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _default_status(self) -> dict[str, Any]:
        return {
            "state": "idle",
            "stage": "idle",
            "updated_at": None,
            "started_at": None,
            "error": None,
            "cache_exists": self.history_cache_path.exists(),
            "thread_alive": False,
            "message": "",
            "progress": 0,
            "current_theme_index": 0,
            "current_theme_slug": None,
            "current_theme_label": None,
            "completed_themes": 0,
            "total_themes": 0,
            "scanned_core_papers": 0,
            "total_core_papers": 0,
            "can_stop": False,
            "can_continue": False,
            "can_start": True,
        }

    def _rebuild_worker(self, *, model: str, max_themes: int) -> None:
        try:
            self._execute_rebuild(model=model, max_themes=max_themes)
        except InterruptedError:
            self._mark_stopped("历史脉络任务已停止，可稍后继续。")
        except Exception as exc:
            status = self._load_json(self.status_path)
            status.update(
                {
                    "state": "failed",
                    "stage": "failed",
                    "updated_at": _timestamp(),
                    "message": "历史脉络生成失败。",
                    "error": str(exc),
                    "stop_requested": False,
                }
            )
            self._write_json(self.status_path, status)
        finally:
            with self._lock:
                self._thread = None
                self._stop_requested = False

    def _execute_rebuild(self, *, model: str, max_themes: int) -> None:
        self._update_status(
            stage="scan",
            progress=0,
            message="正在扫描已有核心解读。",
            scanned_core_papers=0,
            total_core_papers=0,
        )
        papers = load_core_history_papers(self.library_root, progress_callback=self._scan_progress_callback())
        self._update_status(
            stage="scan",
            progress=SCAN_PROGRESS,
            message=f"已扫描 {len(papers)}/{len(papers)} 篇核心解读。",
            scanned_core_papers=len(papers),
            total_core_papers=len(papers),
        )
        self._check_stop()

        self._update_status(stage="theme_selection", progress=THEME_SELECTION_PROGRESS, message="正在筛选历史主线主题。")
        theme_groups = select_theme_groups(papers, max_themes=max_themes)
        total_themes = len(theme_groups)
        self._update_status(total_themes=total_themes, current_theme_index=0, completed_themes=0)
        self._check_stop()

        existing_payload = self.load_history() or {}
        existing_themes = {
            str(theme.get("slug")): theme
            for theme in existing_payload.get("themes", [])
            if isinstance(theme, dict) and theme.get("slug")
        }
        result_themes: list[dict[str, Any]] = []

        for index, (theme, theme_papers) in enumerate(theme_groups, start=1):
            current_progress = THEME_SELECTION_PROGRESS + int((index - 1) / max(1, total_themes) * (WRITE_PROGRESS - THEME_SELECTION_PROGRESS))
            self._update_status(
                stage="theme_generation",
                progress=current_progress,
                message=f"正在生成第 {index}/{total_themes} 条历史主线：{theme.label}",
                current_theme_index=index,
                current_theme_slug=theme.slug,
                current_theme_label=theme.label,
                total_themes=total_themes,
            )
            self._check_stop()

            cached_theme = existing_themes.get(theme.slug)
            if cached_theme and cached_theme.get("paper_count") == len(theme_papers):
                theme_payload = cached_theme
            else:
                theme_payload = build_theme_payload(
                    theme,
                    theme_papers,
                    library_root=self.library_root,
                    model=model,
                    should_abort=self._should_abort,
                    progress_callback=self._theme_progress_callback(index=index, total=total_themes, label=theme.label),
                )
            result_themes.append(theme_payload)
            partial_payload = {
                "generated_at": _timestamp(),
                "source_paper_count": len(papers),
                "theme_count": len(result_themes),
                "total_theme_count": total_themes,
                "themes": result_themes,
            }
            self._write_json(self.history_cache_path, partial_payload)
            next_progress = THEME_SELECTION_PROGRESS + int(index / max(1, total_themes) * (WRITE_PROGRESS - THEME_SELECTION_PROGRESS))
            self._update_status(
                completed_themes=index,
                progress=next_progress,
                message=f"已完成 {index}/{total_themes} 条历史主线。",
            )
            self._check_stop()

        final_payload = {
            "generated_at": _timestamp(),
            "source_paper_count": len(papers),
            "theme_count": len(result_themes),
            "themes": result_themes,
        }
        self._update_status(stage="writing", progress=WRITE_PROGRESS, message="正在写入最终 Insights 结果。")
        self._write_json(self.history_cache_path, final_payload)
        self._write_json(
            self.status_path,
            {
                "state": "ready",
                "stage": "complete",
                "started_at": self._load_json(self.status_path).get("started_at") or _timestamp(),
                "updated_at": _timestamp(),
                "message": "历史脉络已生成。",
                "error": None,
                "progress": 100,
                "current_theme_index": total_themes,
                "current_theme_slug": result_themes[-1]["slug"] if result_themes else None,
                "current_theme_label": result_themes[-1]["label"] if result_themes else None,
                "completed_themes": total_themes,
                "total_themes": total_themes,
                "scanned_core_papers": len(papers),
                "total_core_papers": len(papers),
                "stop_requested": False,
                "model": model,
            },
        )

    def _theme_progress_callback(self, *, index: int, total: int, label: str):
        base = THEME_SELECTION_PROGRESS + (index - 1) / max(1, total) * (WRITE_PROGRESS - THEME_SELECTION_PROGRESS)
        width = (WRITE_PROGRESS - THEME_SELECTION_PROGRESS) / max(1, total)

        def callback(step_progress: int, message: str) -> None:
            normalized = max(0, min(step_progress, 99)) / 100
            overall = int(base + width * normalized)
            self._update_status(
                progress=max(overall, 1),
                stage="theme_generation",
                message=f"{label}：{message}",
            )
            self._check_stop()

        return callback

    def _scan_progress_callback(self):
        def callback(scanned: int, total: int, title: str) -> None:
            progress = int(scanned / max(1, total) * SCAN_PROGRESS) if total else SCAN_PROGRESS
            self._update_status(
                stage="scan",
                progress=max(progress, 1 if scanned else 0),
                message=f"正在扫描核心解读 {scanned}/{total}：{title}" if total else "正在扫描已有核心解读。",
                scanned_core_papers=scanned,
                total_core_papers=total,
            )
            self._check_stop()

        return callback

    def _should_abort(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _check_stop(self) -> None:
        if self._should_abort():
            raise InterruptedError("History insights task interrupted.")

    def _mark_stopped(self, message: str) -> None:
        status = self._load_json(self.status_path)
        status.update(
            {
                "state": "stopped",
                "stage": "stopped",
                "updated_at": _timestamp(),
                "message": message,
                "error": None,
                "stop_requested": False,
            }
        )
        self._write_json(self.status_path, status)

    def _update_status(self, **changes: Any) -> None:
        status = self._load_json(self.status_path)
        if not status:
            status = self._default_status()
        status.update(changes)
        status["updated_at"] = _timestamp()
        self._write_json(self.status_path, status)

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_history_payload(library_root: Path, *, model: str | None = None, max_themes: int = MAX_THEME_COUNT) -> dict[str, Any]:
    papers = load_core_history_papers(library_root)
    theme_groups = select_theme_groups(papers, max_themes=max_themes)
    theme_payloads: list[dict[str, Any]] = []
    for theme, theme_papers in theme_groups:
        theme_payloads.append(build_theme_payload(theme, theme_papers, library_root=library_root, model=model))
    return {
        "generated_at": _timestamp(),
        "source_paper_count": len(papers),
        "theme_count": len(theme_payloads),
        "themes": theme_payloads,
    }


def load_core_history_papers(
    library_root: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[HistoryPaper]:
    library_root = library_root.resolve()
    prompt_root = library_root / ".paper-reader-ai"
    sources = collect_core_history_sources(library_root, prompt_root)
    records: list[HistoryPaper] = []
    total = len(sources)
    for index, (item, prompt_path) in enumerate(sources, start=1):
        rel_path = str(item.get("rel_path") or "").strip()
        body = _parse_prompt_body(prompt_path.read_text(encoding="utf-8"))
        summary = extract_digest(body)
        methods = match_terms(summary + "\n" + body, METHOD_DEFINITIONS)
        signals = match_terms(summary + "\n" + body, SIGNAL_DEFINITIONS)
        novelty_score = len(methods) + len(signals) + count_cues(body, SIGNAL_DEFINITIONS["foundation"])
        turning_score = len(signals) + count_cues(body, SIGNAL_DEFINITIONS["turning"])
        title = str(item.get("display_title") or item.get("title") or Path(rel_path).stem)
        records.append(
            HistoryPaper(
                rel_path=rel_path,
                title=title,
                sort_date=item.get("sort_date"),
                summary=summary,
                methods=methods,
                signals=signals,
                novelty_score=novelty_score,
                turning_score=turning_score,
            )
        )
        if progress_callback:
            progress_callback(index, total, title)
    return sorted(records, key=lambda paper: (paper.sort_date or "0000-00-00", paper.title.lower()))


def collect_core_history_sources(library_root: Path, prompt_root: Path | None = None) -> list[tuple[dict[str, Any], Path]]:
    library_root = library_root.resolve()
    prompt_root = prompt_root or (library_root / ".paper-reader-ai")
    seen: set[str] = set()
    sources: list[tuple[dict[str, Any], Path]] = []
    for index_name in (".paper_reader_index.json", ".paper_reader_done_index.json"):
        index_path = library_root / index_name
        if not index_path.exists():
            continue
        payload = _load_json(index_path)
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("rel_path") or "").strip()
            if not rel_path or rel_path in seen:
                continue
            prompt_slugs = [str(slug) for slug in item.get("prompt_result_slugs", []) if slug]
            if CORE_PROMPT_SLUG not in prompt_slugs:
                continue
            prompt_path = prompt_root / Path(rel_path) / f"{CORE_PROMPT_SLUG}.md"
            if not prompt_path.exists():
                prompt_path = prompt_root / Path(rel_path).parent / f"{Path(rel_path).stem}.explained.zh.md"
            if not prompt_path.exists():
                continue
            seen.add(rel_path)
            sources.append((item, prompt_path))
    sources.sort(
        key=lambda pair: (
            str(pair[0].get("sort_date") or "0000-00-00"),
            str(pair[0].get("display_title") or pair[0].get("title") or pair[0].get("rel_path") or "").lower(),
        )
    )
    return sources


def select_theme_groups(papers: list[HistoryPaper], *, max_themes: int) -> list[tuple[ThemeDefinition, list[HistoryPaper]]]:
    assignments: dict[str, list[HistoryPaper]] = defaultdict(list)
    for paper in papers:
        text = f"{paper.title}\n{paper.summary}"
        matched = [theme.slug for theme in THEMES if theme_matches(text, theme)]
        for slug in matched:
            assignments[slug].append(paper)

    total = max(1, len(papers))
    priority = {
        slug: index
        for index, slug in enumerate(
            [
                "agent",
                "benchmark",
                "memory",
                "reasoning",
                "alignment",
                "rl",
                "world_model",
                "robotics",
                "multimodal",
                "data",
                "interpretability",
                "diffusion",
            ]
        )
    }
    candidates: list[tuple[ThemeDefinition, list[HistoryPaper], int]] = []
    for theme in THEMES:
        group = assignments.get(theme.slug, [])
        prevalence = len(group) / total
        if len(group) < MIN_THEME_PAPERS or prevalence > MAX_THEME_PREVALENCE:
            continue
        score = len(group) * 10 + sum(paper.turning_score + paper.novelty_score for paper in group[:30])
        candidates.append((theme, sorted(group, key=lambda paper: (paper.sort_date or "0000-00-00", paper.title.lower())), score))
    candidates.sort(key=lambda item: (-item[2], priority.get(item[0].slug, 99), item[0].slug))
    return [(theme, group) for theme, group, _ in candidates[:max_themes]]


def build_theme_payload(
    theme: ThemeDefinition,
    papers: list[HistoryPaper],
    *,
    library_root: Path,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    year_counts = Counter(paper.year for paper in papers if paper.year != "unknown")
    selected = select_context_papers(papers)
    llm_payload = synthesize_theme_history(
        theme,
        papers,
        selected,
        library_root=library_root,
        model=model,
        should_abort=should_abort,
        progress_callback=progress_callback,
    )
    representative = [paper_to_dict(paper, paper_id=f"P{index:02d}") for index, paper in enumerate(selected, start=1)]
    paper_by_id = {paper_entry["paper_id"]: paper_entry for paper_entry in representative}

    milestones = []
    for item in llm_payload.get("milestones", []):
        paper_ref = paper_by_id.get(item.get("paper_id", ""))
        if paper_ref is None:
            continue
        milestones.append(
            {
                "paper_id": paper_ref["paper_id"],
                "title": paper_ref["title"],
                "sort_date": paper_ref["sort_date"],
                "role": item.get("role", "milestone"),
                "why": item.get("why", ""),
                "rel_path": paper_ref["rel_path"],
            }
        )

    phases = []
    for phase in llm_payload.get("phase_cards", []):
        phase_papers = [paper_by_id[paper_id] for paper_id in phase.get("paper_ids", []) if paper_id in paper_by_id]
        if not phase_papers:
            phase_papers = [paper for paper in phase.get("papers", []) if isinstance(paper, dict)]
        phases.append(
            {
                "phase": phase.get("phase", "阶段"),
                "label": phase.get("label", "阶段"),
                "summary": phase.get("summary", ""),
                "main_shift": phase.get("main_shift", ""),
                "papers": phase_papers,
            }
        )

    start_date = next((paper.sort_date for paper in papers if paper.sort_date), None)
    end_date = next((paper.sort_date for paper in reversed(papers) if paper.sort_date), None)
    return {
        "slug": theme.slug,
        "label": theme.label,
        "description": theme.description,
        "paper_count": len(papers),
        "start_date": start_date,
        "end_date": end_date,
        "year_counts": [{"year": year, "count": count} for year, count in sorted(year_counts.items())],
        "history_summary": llm_payload.get("history_summary") or fallback_history_summary(theme, papers),
        "why_it_emerged": llm_payload.get("why_it_emerged") or theme.description,
        "phase_cards": phases or fallback_phase_cards(papers, representative),
        "milestones": milestones or fallback_milestones(representative),
        "paradigm_shifts": llm_payload.get("paradigm_shifts", []),
        "mainstreaming": llm_payload.get("mainstreaming", []),
        "representative_papers": representative,
        "llm_used": bool(llm_payload.get("llm_used")),
    }


def synthesize_theme_history(
    theme: ThemeDefinition,
    papers: list[HistoryPaper],
    selected: list[HistoryPaper],
    *,
    library_root: Path,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    context = build_theme_context(theme, papers, selected)
    prompt = (
        "你是 AutoResearch 的历史脉络分析器。\n"
        "你的任务不是总结单篇论文，而是基于给定的候选论文，梳理一个主题是如何演化的。\n\n"
        "要求：\n"
        "1. 只使用给定材料，不要编造论文。\n"
        "2. 输出必须是 JSON object，不要带 Markdown 代码块。\n"
        "3. `paper_id` 只能使用输入里给出的 ID。\n"
        "4. phase_cards 保持 3-5 段，milestones 保持 4-8 个。\n"
        "5. 重点回答：这个方向为什么出现、哪些论文是奠基点、哪些论文改写了路线、哪些只是把旧路线推成主流。\n\n"
        "输出 JSON schema:\n"
        "{\n"
        '  "history_summary": "...",\n'
        '  "why_it_emerged": "...",\n'
        '  "phase_cards": [{"phase": "...", "label": "...", "summary": "...", "main_shift": "...", "paper_ids": ["P01", "P02"]}],\n'
        '  "milestones": [{"paper_id": "P01", "role": "foundation|turning_point|mainstreaming", "why": "..."}],\n'
        '  "paradigm_shifts": ["..."],\n'
        '  "mainstreaming": ["..."]\n'
        "}\n\n"
        f"主题：{theme.label}\n"
        f"主题说明：{theme.description}\n\n"
        f"材料：\n{context}\n"
    )
    try:
        raw = run_text_prompt(
            prompt,
            workdir=library_root,
            model=model,
            progress_callback=progress_callback,
            should_abort=should_abort,
        )
        parsed = extract_json_object(raw)
        if isinstance(parsed, dict):
            parsed["llm_used"] = True
            return parsed
    except Exception:
        if should_abort and should_abort():
            raise InterruptedError("Theme synthesis interrupted.")
    return {
        "history_summary": fallback_history_summary(theme, papers),
        "why_it_emerged": theme.description,
        "phase_cards": fallback_phase_cards(papers, [paper_to_dict(paper) for paper in selected]),
        "milestones": fallback_milestones([paper_to_dict(paper) for paper in selected]),
        "paradigm_shifts": [],
        "mainstreaming": [],
        "llm_used": False,
    }


def build_theme_context(theme: ThemeDefinition, papers: list[HistoryPaper], selected: list[HistoryPaper]) -> str:
    total_years = Counter(paper.year for paper in papers if paper.year != "unknown")
    method_counts = Counter(method for paper in papers for method in set(paper.methods))
    lines = [
        f"总论文数：{len(papers)}",
        "年度分布：" + ", ".join(f"{year}:{count}" for year, count in sorted(total_years.items())),
        "方法信号：" + ", ".join(f"{method}:{count}" for method, count in method_counts.most_common(8)),
        "候选代表论文：",
    ]
    for index, paper in enumerate(selected, start=1):
        paper_id = f"P{index:02d}"
        methods = ", ".join(paper.methods[:4]) or "none"
        signals = ", ".join(paper.signals[:4]) or "none"
        lines.extend(
            [
                f"- {paper_id}",
                f"  date: {paper.sort_date or 'unknown'}",
                f"  title: {paper.title}",
                f"  methods: {methods}",
                f"  signals: {signals}",
                f"  summary: {paper.summary}",
            ]
        )
    return "\n".join(lines)


def select_context_papers(papers: list[HistoryPaper]) -> list[HistoryPaper]:
    by_year: dict[str, list[HistoryPaper]] = defaultdict(list)
    for paper in papers:
        by_year[paper.year].append(paper)

    selected: list[HistoryPaper] = []
    for year in sorted(by_year):
        year_papers = sorted(
            by_year[year],
            key=lambda item: (item.turning_score, item.novelty_score, item.sort_date or "0000-00-00"),
            reverse=True,
        )
        picks = dedupe_papers(year_papers[:MAX_YEAR_PICKS])
        selected.extend(picks)

    selected = dedupe_papers(sorted(selected, key=lambda item: (item.sort_date or "0000-00-00", item.title.lower())))
    if len(selected) > MAX_CONTEXT_PAPERS:
        ranked = sorted(selected, key=lambda item: (item.turning_score, item.novelty_score, item.sort_date or "0000-00-00"), reverse=True)
        kept = ranked[:MAX_CONTEXT_PAPERS]
        selected = dedupe_papers(sorted(kept, key=lambda item: (item.sort_date or "0000-00-00", item.title.lower())))
    return selected


def fallback_history_summary(theme: ThemeDefinition, papers: list[HistoryPaper]) -> str:
    start = next((paper.sort_date for paper in papers if paper.sort_date), "unknown")
    end = next((paper.sort_date for paper in reversed(papers) if paper.sort_date), "unknown")
    method_counts = Counter(method for paper in papers for method in paper.methods)
    top_methods = ", ".join(method for method, _ in method_counts.most_common(3)) or "mixed signals"
    return f"{theme.label} 这条线从 {start} 到 {end} 持续扩展，核心变化是研究重点逐渐从早期问题定义，转向 {top_methods} 这类更可落地、更可评测的组织方式。"


def fallback_phase_cards(papers: list[HistoryPaper], representative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in representative:
        year = str(paper.get("sort_date") or "unknown")[:4]
        buckets[year].append(paper)
    phases: list[dict[str, Any]] = []
    for year in sorted(buckets)[:5]:
        phases.append(
            {
                "phase": year,
                "label": year,
                "summary": f"这一阶段开始积累与 {year} 年相关的代表论文。",
                "main_shift": "从定义问题逐渐走向具体机制和评测落地。",
                "papers": buckets[year][:3],
            }
        )
    return phases[:4]


def fallback_milestones(representative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    milestones: list[dict[str, Any]] = []
    for index, paper in enumerate(representative[:6]):
        role = "foundation" if index < 2 else "turning_point" if index < 4 else "mainstreaming"
        milestones.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "sort_date": paper["sort_date"],
                "role": role,
                "why": "它在当前候选集合里覆盖了一个关键阶段。",
                "rel_path": paper["rel_path"],
            }
        )
    return milestones


def paper_to_dict(paper: HistoryPaper, *, paper_id: str | None = None) -> dict[str, Any]:
    return {
        "paper_id": paper_id or "",
        "rel_path": paper.rel_path,
        "title": paper.title,
        "sort_date": paper.sort_date,
        "summary": paper.summary,
        "methods": paper.methods,
        "signals": paper.signals,
    }


def dedupe_papers(papers: list[HistoryPaper]) -> list[HistoryPaper]:
    seen: set[str] = set()
    output: list[HistoryPaper] = []
    for paper in papers:
        if paper.rel_path in seen:
            continue
        seen.add(paper.rel_path)
        output.append(paper)
    return output


def extract_digest(body: str) -> str:
    paragraphs = [clean_text(chunk) for chunk in body.split("\n\n") if clean_text(chunk)]
    preferred = [paragraph for paragraph in paragraphs if len(paragraph) >= 20 and not paragraph.startswith("#")]
    if preferred:
        return preferred[0][:360]
    sentences = [clean_text(part) for part in SENTENCE_RE.split(body) if clean_text(part)]
    for sentence in sentences:
        if len(sentence) >= 20:
            return sentence[:360]
    return clean_text(body)[:360]


def _parse_prompt_body(text: str) -> str:
    marker = "\n---\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def theme_matches(text: str, theme: ThemeDefinition) -> bool:
    return any(term_in_text(text, keyword) for keyword in theme.keywords)


def match_terms(text: str, definitions: dict[str, tuple[str, ...]]) -> list[str]:
    matches: list[str] = []
    for slug, keywords in definitions.items():
        if any(term_in_text(text, keyword) for keyword in keywords):
            matches.append(slug)
    return matches


def count_cues(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if term_in_text(text, keyword))


def term_in_text(text: str, keyword: str) -> bool:
    lowered = text.lower()
    term = keyword.lower().strip()
    if not term:
        return False
    if _ASCII_TERM_RE.match(term):
        pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        return re.search(pattern, lowered) is not None
    return term in lowered


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).strip(" -*")


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
