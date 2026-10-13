from __future__ import annotations

import json
import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from .ai_summary import run_text_prompt
from .insights_history import (
    INSIGHTS_DIR_NAME,
    SIGNAL_DEFINITIONS,
    THEMES,
    collect_core_history_sources,
    count_cues,
    extract_digest,
    extract_json_object,
    term_in_text,
)

MOMENTUM_CACHE_FILE_NAME = "momentum_dashboard.json"
MOMENTUM_STATUS_FILE_NAME = "momentum_status.json"
MOMENTUM_WINDOWS = (30, 60, 90)
SCAN_PROGRESS = 10
WINDOW_SELECTION_PROGRESS = 18
WRITE_PROGRESS = 98
MAX_WINDOW_ITEMS = 4
MAX_WINDOW_SUPPORT_PAPERS = 3


@dataclass(slots=True)
class MomentumPaper:
    rel_path: str
    title: str
    sort_date: str | None
    summary: str
    topics: list[str]
    methods: list[str]
    benchmark_signals: list[str]
    novelty_score: int
    turning_score: int

    @property
    def parsed_date(self) -> date | None:
        if not self.sort_date:
            return None
        try:
            return datetime.fromisoformat(str(self.sort_date)[:10]).date()
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class SignalDefinition:
    label: str
    keywords: tuple[str, ...]


METHOD_SIGNAL_DEFINITIONS = (
    SignalDefinition("Planning/Planner", ("planning", "planner", "plan", "规划")),
    SignalDefinition("Memory/Context", ("memory", "context management", "记忆", "长期上下文")),
    SignalDefinition("Retrieval/RAG", ("retrieval", "retriever", "rag", "检索")),
    SignalDefinition("Verification/Executable Checks", ("verification", "executable checks", "verifier", "grounded", "验证")),
    SignalDefinition("Synthetic Data", ("synthetic data", "generated data", "合成数据")),
    SignalDefinition("Workflow/Tool Use", ("workflow", "tool workflow", "tool use", "agent workflow", "工作流")),
    SignalDefinition("Self-Evolution", ("self-evolving", "collective evolution", "self-improvement", "自我演化")),
    SignalDefinition("Distillation", ("distillation", "蒸馏")),
    SignalDefinition("Reasoning/Test-Time Compute", ("test-time", "reasoning", "chain-of-thought", "推理")),
)

BENCHMARK_SIGNAL_DEFINITIONS = (
    SignalDefinition("Arena/Benchmark", ("benchmark", "bench", "arena", "leaderboard", "评测", "测评")),
    SignalDefinition("Workspace/Real-World Eval", ("workspace", "real-world", "browser", "computer use", "真实环境", "动态环境")),
    SignalDefinition("Simulation/User Model Eval", ("simulation", "simulator", "synthetic user", "life simulator", "虚拟用户", "世界模型")),
    SignalDefinition("Dataset/Data Engine", ("dataset", "corpus", "data engine", "数据集", "数据引擎")),
    SignalDefinition("Evaluation Setup/Executable Judge", ("evaluation setup", "judge", "unit test", "tool execution", "executable checks", "执行验证")),
)


class MomentumInsightsStore:
    def __init__(self, library_root: Path):
        self.library_root = library_root.resolve()
        self.cache_dir = self.library_root / INSIGHTS_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_path = self.cache_dir / MOMENTUM_CACHE_FILE_NAME
        self.status_path = self.cache_dir / MOMENTUM_STATUS_FILE_NAME
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def load_dashboard(self) -> dict[str, Any] | None:
        payload = self._load_json(self.dashboard_path)
        return payload if payload else None

    def status_snapshot(self) -> dict[str, Any]:
        payload = self._load_json(self.status_path)
        if not payload:
            return self._default_status()

        state = str(payload.get("state") or "idle")
        if state in {"running", "stopping"} and not self._is_running():
            payload["state"] = "stopped"
            payload["message"] = payload.get("message") or "Momentum 任务已中断，可继续生成。"
            payload["updated_at"] = _timestamp()
            payload["stop_requested"] = False
            self._write_json(self.status_path, payload)

        payload.setdefault("progress", 0)
        payload.setdefault("completed_windows", 0)
        payload.setdefault("total_windows", 0)
        payload.setdefault("current_window_days", None)
        payload.setdefault("current_window_label", None)
        payload.setdefault("stage", "idle")
        payload.setdefault("message", "")
        payload.setdefault("error", None)
        payload.setdefault("scanned_core_papers", 0)
        payload.setdefault("total_core_papers", 0)
        payload["cache_exists"] = self.dashboard_path.exists()
        payload["thread_alive"] = self._is_running()
        payload["can_stop"] = payload["state"] in {"running", "stopping"}
        payload["can_continue"] = payload["state"] in {"stopped", "failed"}
        payload["can_start"] = payload["state"] in {"idle", "ready"} and not self._is_running()
        return payload

    def start_or_resume(self, *, model: str | None = None, windows: tuple[int, ...] = MOMENTUM_WINDOWS) -> bool:
        with self._lock:
            if self._is_running():
                return False
            now = _timestamp()
            existing = self._load_json(self.status_path)
            resumable = existing.get("state") in {"stopped", "failed"}
            self._stop_requested = False
            self._write_json(
                self.status_path,
                {
                    "state": "running",
                    "stage": existing.get("stage") if resumable else "scan",
                    "started_at": str(existing.get("started_at") or now),
                    "updated_at": now,
                    "message": "正在扫描核心解读并构建 Momentum Radar。",
                    "error": None,
                    "progress": int(existing.get("progress") or 0) if resumable else 0,
                    "completed_windows": int(existing.get("completed_windows") or 0) if resumable else 0,
                    "total_windows": int(existing.get("total_windows") or 0) if resumable else 0,
                    "current_window_days": existing.get("current_window_days"),
                    "current_window_label": existing.get("current_window_label"),
                    "scanned_core_papers": int(existing.get("scanned_core_papers") or 0),
                    "total_core_papers": int(existing.get("total_core_papers") or 0),
                    "stop_requested": False,
                    "model": model,
                },
            )
            self._thread = threading.Thread(
                target=self._rebuild_worker,
                kwargs={"model": model, "windows": windows},
                daemon=True,
                name="paper-reader-momentum-insights",
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
            status["message"] = "正在停止 Momentum Radar 生成任务..."
            self._write_json(self.status_path, status)
        return True

    def _is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _default_status(self) -> dict[str, Any]:
        return {
            "state": "idle",
            "stage": "idle",
            "updated_at": None,
            "started_at": None,
            "error": None,
            "cache_exists": self.dashboard_path.exists(),
            "thread_alive": False,
            "message": "",
            "progress": 0,
            "completed_windows": 0,
            "total_windows": 0,
            "current_window_days": None,
            "current_window_label": None,
            "scanned_core_papers": 0,
            "total_core_papers": 0,
            "can_stop": False,
            "can_continue": False,
            "can_start": True,
        }

    def _rebuild_worker(self, *, model: str, windows: tuple[int, ...]) -> None:
        try:
            self._execute_rebuild(model=model, windows=windows)
        except InterruptedError:
            self._mark_stopped("Momentum Radar 任务已停止，可稍后继续。")
        except Exception as exc:
            status = self._load_json(self.status_path)
            status.update(
                {
                    "state": "failed",
                    "stage": "failed",
                    "updated_at": _timestamp(),
                    "message": "Momentum Radar 生成失败。",
                    "error": str(exc),
                    "stop_requested": False,
                }
            )
            self._write_json(self.status_path, status)
        finally:
            with self._lock:
                self._thread = None
                self._stop_requested = False

    def _execute_rebuild(self, *, model: str, windows: tuple[int, ...]) -> None:
        self._update_status(
            stage="scan",
            progress=0,
            message="正在扫描已有核心解读。",
            scanned_core_papers=0,
            total_core_papers=0,
        )
        papers = load_momentum_papers(self.library_root, progress_callback=self._scan_progress_callback())
        anchor_date = select_anchor_date(papers)
        self._update_status(
            stage="scan",
            progress=SCAN_PROGRESS,
            message=f"已扫描 {len(papers)}/{len(papers)} 篇核心解读。",
            scanned_core_papers=len(papers),
            total_core_papers=len(papers),
        )
        self._check_stop()

        ordered_windows = tuple(sorted(set(int(days) for days in windows)))
        self._update_status(
            stage="window_selection",
            progress=WINDOW_SELECTION_PROGRESS,
            total_windows=len(ordered_windows),
            completed_windows=0,
            message="正在组织 30/60/90 天 Momentum 窗口。",
        )
        self._check_stop()

        existing_payload = self.load_dashboard() or {}
        existing_windows = {
            int(item.get("days")): item
            for item in existing_payload.get("windows", [])
            if isinstance(item, dict) and item.get("days")
        }
        result_windows: list[dict[str, Any]] = []

        for index, days in enumerate(ordered_windows, start=1):
            current_papers = papers_for_window(papers, anchor_date, days)
            previous_papers = papers_for_previous_window(papers, anchor_date, days)
            current_progress = WINDOW_SELECTION_PROGRESS + int((index - 1) / max(1, len(ordered_windows)) * (WRITE_PROGRESS - WINDOW_SELECTION_PROGRESS))
            self._update_status(
                stage="window_generation",
                progress=current_progress,
                message=f"正在生成过去 {days} 天的 Momentum Radar。",
                current_window_days=days,
                current_window_label=window_label(days),
                total_windows=len(ordered_windows),
            )
            self._check_stop()

            cached = existing_windows.get(days)
            if (
                cached
                and str(cached.get("anchor_date") or "") == anchor_date.isoformat()
                and int(cached.get("paper_count") or 0) == len(current_papers)
                and int(cached.get("previous_paper_count") or 0) == len(previous_papers)
            ):
                window_payload = cached
            else:
                window_payload = build_momentum_window_payload(
                    days,
                    papers,
                    anchor_date=anchor_date,
                    model=model,
                    should_abort=self._should_abort,
                    progress_callback=self._window_progress_callback(index=index, total=len(ordered_windows), days=days),
                )
            result_windows.append(window_payload)
            partial_payload = {
                "generated_at": _timestamp(),
                "anchor_date": anchor_date.isoformat(),
                "source_paper_count": len(papers),
                "window_count": len(result_windows),
                "windows": result_windows,
            }
            self._write_json(self.dashboard_path, partial_payload)
            next_progress = WINDOW_SELECTION_PROGRESS + int(index / max(1, len(ordered_windows)) * (WRITE_PROGRESS - WINDOW_SELECTION_PROGRESS))
            self._update_status(
                completed_windows=index,
                progress=next_progress,
                message=f"已完成 {index}/{len(ordered_windows)} 个 Momentum 窗口。",
            )
            self._check_stop()

        final_payload = {
            "generated_at": _timestamp(),
            "anchor_date": anchor_date.isoformat(),
            "source_paper_count": len(papers),
            "window_count": len(result_windows),
            "windows": result_windows,
        }
        self._update_status(stage="writing", progress=WRITE_PROGRESS, message="正在写入 Momentum Radar 结果。")
        self._write_json(self.dashboard_path, final_payload)
        self._write_json(
            self.status_path,
            {
                "state": "ready",
                "stage": "complete",
                "started_at": self._load_json(self.status_path).get("started_at") or _timestamp(),
                "updated_at": _timestamp(),
                "message": "Momentum Radar 已生成。",
                "error": None,
                "progress": 100,
                "completed_windows": len(ordered_windows),
                "total_windows": len(ordered_windows),
                "current_window_days": ordered_windows[-1] if ordered_windows else None,
                "current_window_label": window_label(ordered_windows[-1]) if ordered_windows else None,
                "scanned_core_papers": len(papers),
                "total_core_papers": len(papers),
                "stop_requested": False,
                "model": model,
            },
        )

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

    def _window_progress_callback(self, *, index: int, total: int, days: int):
        base = WINDOW_SELECTION_PROGRESS + (index - 1) / max(1, total) * (WRITE_PROGRESS - WINDOW_SELECTION_PROGRESS)
        width = (WRITE_PROGRESS - WINDOW_SELECTION_PROGRESS) / max(1, total)

        def callback(step_progress: int, message: str) -> None:
            normalized = max(0, min(step_progress, 99)) / 100
            overall = int(base + width * normalized)
            self._update_status(
                progress=max(overall, 1),
                stage="window_generation",
                current_window_days=days,
                current_window_label=window_label(days),
                message=f"过去 {days} 天：{message}",
            )
            self._check_stop()

        return callback

    def _should_abort(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _check_stop(self) -> None:
        if self._should_abort():
            raise InterruptedError("Momentum insights task interrupted.")

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


def load_momentum_papers(
    library_root: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[MomentumPaper]:
    library_root = library_root.resolve()
    prompt_root = library_root / ".paper-reader-ai"
    sources = collect_core_history_sources(library_root, prompt_root)
    records: list[MomentumPaper] = []
    total = len(sources)
    for index, (item, prompt_path) in enumerate(sources, start=1):
        rel_path = str(item.get("rel_path") or "").strip()
        title = str(item.get("display_title") or item.get("title") or Path(rel_path).stem)
        body = parse_prompt_body(prompt_path.read_text(encoding="utf-8"))
        summary = extract_digest(body)
        text = f"{title}\n{summary}\n{body}"
        topics = [theme.label for theme in THEMES if any(term_in_text(text, keyword) for keyword in theme.keywords)]
        methods = match_signal_labels(text, METHOD_SIGNAL_DEFINITIONS)
        benchmarks = match_signal_labels(text, BENCHMARK_SIGNAL_DEFINITIONS)
        novelty_score = len(topics) + len(methods) + len(benchmarks) + count_cues(body, SIGNAL_DEFINITIONS["foundation"])
        turning_score = len(benchmarks) + len(methods) + count_cues(body, SIGNAL_DEFINITIONS["turning"])
        records.append(
            MomentumPaper(
                rel_path=rel_path,
                title=title,
                sort_date=item.get("sort_date"),
                summary=summary,
                topics=topics,
                methods=methods,
                benchmark_signals=benchmarks,
                novelty_score=novelty_score,
                turning_score=turning_score,
            )
        )
        if progress_callback:
            progress_callback(index, total, title)
    return sorted(records, key=lambda paper: (paper.sort_date or "0000-00-00", paper.title.lower()))


def build_momentum_window_payload(
    days: int,
    papers: list[MomentumPaper],
    *,
    anchor_date: date,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    current_papers = papers_for_window(papers, anchor_date, days)
    previous_papers = papers_for_previous_window(papers, anchor_date, days)
    hot_topics = rank_signal_items(current_papers, previous_papers, papers, lambda paper: paper.topics, top_n=MAX_WINDOW_ITEMS, min_current=2)
    method_routes = rank_signal_items(current_papers, previous_papers, papers, lambda paper: paper.methods, top_n=MAX_WINDOW_ITEMS, min_current=2)
    benchmark_focus = rank_signal_items(current_papers, previous_papers, papers, lambda paper: paper.benchmark_signals, top_n=MAX_WINDOW_ITEMS, min_current=1)
    emerging_edges = rank_emerging_edges(current_papers, previous_papers, papers)

    llm_payload = synthesize_window_momentum(
        days,
        anchor_date,
        current_papers,
        previous_papers,
        hot_topics,
        method_routes,
        benchmark_focus,
        emerging_edges,
        model=model,
        should_abort=should_abort,
        progress_callback=progress_callback,
    )

    hot_topics = merge_llm_reasons(hot_topics, llm_payload.get("hot_topics", []))
    method_routes = merge_llm_reasons(method_routes, llm_payload.get("method_routes", []))
    benchmark_focus = merge_llm_reasons(benchmark_focus, llm_payload.get("benchmark_focus", []))
    emerging_edges = merge_llm_reasons(emerging_edges, llm_payload.get("emerging_edges", []))

    return {
        "days": days,
        "label": window_label(days),
        "anchor_date": anchor_date.isoformat(),
        "start_date": (anchor_date - timedelta(days=max(days - 1, 0))).isoformat(),
        "paper_count": len(current_papers),
        "previous_paper_count": len(previous_papers),
        "summary": llm_payload.get("summary") or fallback_window_summary(days, hot_topics, method_routes, benchmark_focus, emerging_edges, len(current_papers)),
        "hot_topics": hot_topics,
        "method_routes": method_routes,
        "benchmark_focus": benchmark_focus,
        "emerging_edges": emerging_edges,
        "llm_used": bool(llm_payload.get("llm_used")),
    }


def synthesize_window_momentum(
    days: int,
    anchor_date: date,
    current_papers: list[MomentumPaper],
    previous_papers: list[MomentumPaper],
    hot_topics: list[dict[str, Any]],
    method_routes: list[dict[str, Any]],
    benchmark_focus: list[dict[str, Any]],
    emerging_edges: list[dict[str, Any]],
    *,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if not current_papers:
        return {
            "summary": f"过去 {days} 天没有足够新的核心解读，暂时无法形成 Momentum Radar。",
            "hot_topics": [],
            "method_routes": [],
            "benchmark_focus": [],
            "emerging_edges": [],
            "llm_used": False,
        }

    candidate_sections = [
        ("hot_topics", hot_topics),
        ("method_routes", method_routes),
        ("benchmark_focus", benchmark_focus),
        ("emerging_edges", emerging_edges),
    ]
    if not any(items for _, items in candidate_sections):
        return {
            "summary": fallback_window_summary(days, hot_topics, method_routes, benchmark_focus, emerging_edges, len(current_papers)),
            "hot_topics": [],
            "method_routes": [],
            "benchmark_focus": [],
            "emerging_edges": [],
            "llm_used": False,
        }

    selected_papers = select_prompt_papers(hot_topics, method_routes, benchmark_focus, emerging_edges)
    paper_ids = {paper.rel_path: f"P{index:02d}" for index, paper in enumerate(selected_papers, start=1)}
    paper_lines = []
    for paper in selected_papers:
        paper_lines.extend(
            [
                f"- {paper_ids[paper.rel_path]}",
                f"  date: {paper.sort_date or 'unknown'}",
                f"  title: {paper.title}",
                f"  summary: {paper.summary}",
            ]
        )

    candidate_lines = []
    for section_name, items in candidate_sections:
        candidate_lines.append(f"[{section_name}]")
        if not items:
            candidate_lines.append("- none")
            continue
        for item in items:
            refs = ", ".join(paper_ids.get(paper["rel_path"], "") for paper in item.get("papers", []) if paper_ids.get(paper["rel_path"]))
            candidate_lines.append(
                f"- {item['name']} | current:{item['count']} | previous:{item['previous_count']} | score:{item['momentum_score']} | refs:{refs or 'none'}"
            )
    candidate_block = "\n".join(candidate_lines)
    paper_block = "\n".join(paper_lines)

    prompt = (
        "你是 AutoResearch 的 Momentum Radar 分析器。\n"
        "任务：基于最近窗口里的论文核心解读，解释哪些 topic / method / benchmark signals 正在升温。\n\n"
        "要求：\n"
        "1. 只能使用给定候选名字，不要发明新的 signal 名称。\n"
        "2. 只根据材料写原因，不要编造数据。\n"
        "3. 输出必须是 JSON object，不要带 Markdown 代码块。\n"
        "4. 每条 why 只写 1-2 句，聚焦‘为什么现在在升温’。\n\n"
        "输出 JSON schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "hot_topics": [{"name": "...", "why": "..."}],\n'
        '  "method_routes": [{"name": "...", "why": "..."}],\n'
        '  "benchmark_focus": [{"name": "...", "why": "..."}],\n'
        '  "emerging_edges": [{"name": "...", "why": "..."}]\n'
        "}\n\n"
        f"窗口：过去 {days} 天\n"
        f"锚点日期：{anchor_date.isoformat()}\n"
        f"当前窗口论文数：{len(current_papers)}\n"
        f"前一窗口论文数：{len(previous_papers)}\n\n"
        f"候选信号：\n{candidate_block}\n\n"
        f"支撑论文：\n{paper_block}\n"
    )

    try:
        raw = run_text_prompt(
            prompt,
            workdir=Path.cwd(),
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
            raise InterruptedError("Momentum synthesis interrupted.")

    return {
        "summary": fallback_window_summary(days, hot_topics, method_routes, benchmark_focus, emerging_edges, len(current_papers)),
        "hot_topics": [],
        "method_routes": [],
        "benchmark_focus": [],
        "emerging_edges": [],
        "llm_used": False,
    }


def select_anchor_date(papers: list[MomentumPaper]) -> date:
    dated = [paper.parsed_date for paper in papers if paper.parsed_date is not None]
    return max(dated) if dated else datetime.utcnow().date()


def papers_for_window(papers: list[MomentumPaper], anchor_date: date, days: int) -> list[MomentumPaper]:
    lower = anchor_date - timedelta(days=max(days - 1, 0))
    return [paper for paper in papers if paper.parsed_date is not None and lower <= paper.parsed_date <= anchor_date]


def papers_for_previous_window(papers: list[MomentumPaper], anchor_date: date, days: int) -> list[MomentumPaper]:
    current_lower = anchor_date - timedelta(days=max(days - 1, 0))
    previous_upper = current_lower - timedelta(days=1)
    previous_lower = previous_upper - timedelta(days=max(days - 1, 0))
    return [paper for paper in papers if paper.parsed_date is not None and previous_lower <= paper.parsed_date <= previous_upper]


def rank_signal_items(
    current_papers: list[MomentumPaper],
    previous_papers: list[MomentumPaper],
    all_papers: list[MomentumPaper],
    extractor: Callable[[MomentumPaper], Iterable[str]],
    *,
    top_n: int,
    min_current: int,
) -> list[dict[str, Any]]:
    current_counts, current_support = signal_counts(current_papers, extractor)
    previous_counts, _ = signal_counts(previous_papers, extractor)
    lifetime_counts, _ = signal_counts(all_papers, extractor)
    items: list[dict[str, Any]] = []

    for name, count in current_counts.items():
        previous_count = previous_counts.get(name, 0)
        current_share = count / max(1, len(current_papers))
        previous_share = previous_count / max(1, len(previous_papers)) if previous_papers else 0
        delta_share = current_share - previous_share
        if count < min_current and previous_count == 0:
            continue
        if delta_share <= 0 and count <= previous_count:
            continue
        momentum_score = round(delta_share * 100 + math.log1p(count) * 8 + (count - previous_count) * 2, 2)
        papers = [momentum_paper_to_view(paper) for paper in select_support_papers(current_support.get(name, []), limit=MAX_WINDOW_SUPPORT_PAPERS)]
        items.append(
            {
                "name": name,
                "count": count,
                "previous_count": previous_count,
                "lifetime_count": lifetime_counts.get(name, count),
                "momentum_score": momentum_score,
                "why": build_signal_description(name, count, previous_count),
                "papers": papers,
            }
        )

    items.sort(key=lambda item: (-item["momentum_score"], -item["count"], item["name"].lower()))
    return items[:top_n]


def rank_emerging_edges(
    current_papers: list[MomentumPaper],
    previous_papers: list[MomentumPaper],
    all_papers: list[MomentumPaper],
) -> list[dict[str, Any]]:
    def combined_signals(paper: MomentumPaper) -> list[str]:
        return [f"Topic::{name}" for name in paper.topics] + [f"Method::{name}" for name in paper.methods]

    current_counts, current_support = signal_counts(current_papers, combined_signals)
    previous_counts, _ = signal_counts(previous_papers, combined_signals)
    lifetime_counts, _ = signal_counts(all_papers, combined_signals)
    total = max(1, len(all_papers))
    items: list[dict[str, Any]] = []

    for raw_name, count in current_counts.items():
        lifetime_count = lifetime_counts.get(raw_name, count)
        if lifetime_count > max(6, int(total * 0.12)):
            continue
        previous_count = previous_counts.get(raw_name, 0)
        current_share = count / max(1, len(current_papers))
        previous_share = previous_count / max(1, len(previous_papers)) if previous_papers else 0
        delta_share = current_share - previous_share
        if delta_share <= 0 and count <= previous_count:
            continue
        signal_type, name = raw_name.split("::", 1)
        scarcity_bonus = max(0.0, 1 - lifetime_count / max(1, total)) * 10
        momentum_score = round(delta_share * 100 + scarcity_bonus + (count - previous_count) * 2, 2)
        papers = [momentum_paper_to_view(paper) for paper in select_support_papers(current_support.get(raw_name, []), limit=MAX_WINDOW_SUPPORT_PAPERS)]
        items.append(
            {
                "name": name,
                "kind": signal_type.lower(),
                "count": count,
                "previous_count": previous_count,
                "lifetime_count": lifetime_count,
                "momentum_score": momentum_score,
                "why": build_emerging_description(name, signal_type, count, previous_count, lifetime_count),
                "papers": papers,
            }
        )

    items.sort(key=lambda item: (-item["momentum_score"], -item["count"], item["name"].lower()))
    return items[:MAX_WINDOW_ITEMS]


def signal_counts(
    papers: list[MomentumPaper],
    extractor: Callable[[MomentumPaper], Iterable[str]],
) -> tuple[Counter[str], dict[str, list[MomentumPaper]]]:
    counts: Counter[str] = Counter()
    support: dict[str, list[MomentumPaper]] = defaultdict(list)
    for paper in papers:
        for signal_name in dict.fromkeys(item for item in extractor(paper) if item):
            counts[signal_name] += 1
            support[signal_name].append(paper)
    return counts, support


def select_support_papers(papers: list[MomentumPaper], *, limit: int) -> list[MomentumPaper]:
    ranked = sorted(
        papers,
        key=lambda paper: (
            paper.sort_date or "0000-00-00",
            paper.turning_score,
            paper.novelty_score,
            paper.title.lower(),
        ),
        reverse=True,
    )
    return ranked[:limit]


def select_prompt_papers(*signal_groups: list[dict[str, Any]]) -> list[MomentumPaper]:
    paper_map: dict[str, MomentumPaper] = {}
    for group in signal_groups:
        for item in group:
            for paper in item.get("papers", []):
                rel_path = str(paper.get("rel_path") or "")
                if not rel_path or rel_path in paper_map:
                    continue
                paper_map[rel_path] = MomentumPaper(
                    rel_path=rel_path,
                    title=str(paper.get("title") or Path(rel_path).stem),
                    sort_date=paper.get("sort_date"),
                    summary=str(paper.get("summary") or ""),
                    topics=[],
                    methods=[],
                    benchmark_signals=[],
                    novelty_score=0,
                    turning_score=0,
                )
    return sorted(paper_map.values(), key=lambda paper: (paper.sort_date or "0000-00-00", paper.title.lower()), reverse=True)[:18]


def merge_llm_reasons(items: list[dict[str, Any]], llm_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    why_by_name = {
        str(item.get("name")): str(item.get("why") or "").strip()
        for item in llm_items
        if isinstance(item, dict) and item.get("name")
    }
    merged: list[dict[str, Any]] = []
    for item in items:
        updated = dict(item)
        reason = why_by_name.get(item["name"])
        if reason:
            updated["why"] = reason
        merged.append(updated)
    return merged


def match_signal_labels(text: str, definitions: tuple[SignalDefinition, ...]) -> list[str]:
    matches: list[str] = []
    for definition in definitions:
        if any(term_in_text(text, keyword) for keyword in definition.keywords):
            matches.append(definition.label)
    return matches


def build_signal_description(name: str, count: int, previous_count: int) -> str:
    if previous_count == 0:
        return f"{name} 在最近窗口里开始集中出现，目前已累计到 {count} 篇，是一个明显的新热区。"
    return f"{name} 在最近窗口里出现 {count} 篇，高于前一窗口的 {previous_count} 篇，说明 follow 信号正在增多。"


def build_emerging_description(name: str, signal_type: str, count: int, previous_count: int, lifetime_count: int) -> str:
    return f"{signal_type} 方向里的 {name} 目前总体样本还不多（累计 {lifetime_count} 篇），但最近窗口已出现 {count} 篇，说明边缘信号正在聚集。"


def fallback_window_summary(
    days: int,
    hot_topics: list[dict[str, Any]],
    method_routes: list[dict[str, Any]],
    benchmark_focus: list[dict[str, Any]],
    emerging_edges: list[dict[str, Any]],
    paper_count: int,
) -> str:
    if paper_count == 0:
        return f"过去 {days} 天没有足够新的核心解读，暂时看不到明显的 Momentum 信号。"
    pieces = [f"过去 {days} 天一共看到 {paper_count} 篇带核心解读的论文。"]
    if hot_topics:
        pieces.append(f"增长最快的话题集中在 {', '.join(item['name'] for item in hot_topics[:2])}。")
    if method_routes:
        pieces.append(f"方法路线里升温最明显的是 {', '.join(item['name'] for item in method_routes[:2])}。")
    if benchmark_focus:
        pieces.append(f"评测与数据焦点开始转向 {', '.join(item['name'] for item in benchmark_focus[:2])}。")
    if emerging_edges:
        pieces.append(f"边缘但开始聚集的信号包括 {', '.join(item['name'] for item in emerging_edges[:2])}。")
    return "".join(pieces)


def parse_prompt_body(text: str) -> str:
    marker = "\n---\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def momentum_paper_to_view(paper: MomentumPaper) -> dict[str, Any]:
    return {
        "paper_id": "",
        "rel_path": paper.rel_path,
        "title": paper.title,
        "sort_date": paper.sort_date,
        "summary": paper.summary,
        "methods": paper.methods,
        "signals": paper.benchmark_signals,
    }


def window_label(days: int) -> str:
    return f"过去 {days} 天"


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
