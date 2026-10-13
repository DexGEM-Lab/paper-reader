from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ai_summary import run_text_prompt
from .insights_history import INSIGHTS_DIR_NAME, THEMES, extract_json_object
from .insights_momentum import (
    BENCHMARK_SIGNAL_DEFINITIONS,
    METHOD_SIGNAL_DEFINITIONS,
    MOMENTUM_CACHE_FILE_NAME,
    MomentumPaper,
    load_momentum_papers,
    match_signal_labels,
    momentum_paper_to_view,
    papers_for_window,
    select_anchor_date,
    select_support_papers,
)

OPPORTUNITY_CACHE_FILE_NAME = "opportunity_map.json"
OPPORTUNITY_STATUS_FILE_NAME = "opportunity_status.json"
SCAN_PROGRESS = 10
SECTION_SELECTION_PROGRESS = 18
WRITE_PROGRESS = 98
MAX_SECTION_ITEMS = 4
MAX_SECTION_SUPPORT_PAPERS = 4
DEFAULT_SECTION_COUNT = 5


@dataclass(frozen=True, slots=True)
class OpportunitySectionDefinition:
    slug: str
    label: str
    description: str
    prompt_focus: str


OPPORTUNITY_SECTIONS = (
    OpportunitySectionDefinition(
        "unresolved_problems",
        "未解问题",
        "哪些问题被不断提起，但到今天仍没有被真正解决。",
        "从候选项里挑出最值得继续打的未解问题，解释为什么它还没有被解决，以及下一步最值得下注的方向。",
    ),
    OpportunitySectionDefinition(
        "weak_claims",
        "证据薄弱的 Claim",
        "哪些 claim 被反复说，但证据体系仍然薄弱。",
        "从候选项里挑出那些已经被反复宣称、但评测或证据仍不够扎实的 claim，说明证据缺口在哪里。",
    ),
    OpportunitySectionDefinition(
        "crowded_spaces",
        "开始拥挤的方向",
        "哪些方向虽然火，但其实已经开始拥挤。",
        "从候选项里挑出最拥挤的热门方向，解释为什么现在再进入会变难，以及需要怎样的差异化。",
    ),
    OpportunitySectionDefinition(
        "high_upside_edges",
        "高回报低密度",
        "哪些方向研究少，但可能回报很高。",
        "从候选项里挑出研究密度还低、但值得继续押注的小方向，解释为什么它可能有高回报。",
    ),
    OpportunitySectionDefinition(
        "transfer_gaps",
        "可迁移的方法空白",
        "哪些不同领域之间存在方法可迁移但尚未迁移的空白。",
        "从候选项里挑出最有价值的方法迁移空白，说明应该把什么方法迁移到哪里，以及为什么现在值得做。",
    ),
)

CLAIM_DEFINITIONS = (
    ("Real-world readiness", ("real-world", "production", "deployment", "真实环境", "落地")),
    ("Long-horizon reliability", ("long-horizon", "long-term", "长期任务", "multi-step", "长链路")),
    ("Generalization/Transfer", ("generalization", "generalize", "transfer", "out-of-distribution", "泛化", "迁移")),
    ("Safety/Alignment robustness", ("safety", "safe", "alignment", "robust", "对齐", "安全", "鲁棒")),
    ("Autonomy", ("autonomous", "autonomy", "self-evolving", "自主", "自我演化")),
)


class OpportunityInsightsStore:
    def __init__(self, library_root: Path):
        self.library_root = library_root.resolve()
        self.cache_dir = self.library_root / INSIGHTS_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.map_path = self.cache_dir / OPPORTUNITY_CACHE_FILE_NAME
        self.status_path = self.cache_dir / OPPORTUNITY_STATUS_FILE_NAME
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def load_map(self) -> dict[str, Any] | None:
        payload = self._load_json(self.map_path)
        return payload if payload else None

    def status_snapshot(self) -> dict[str, Any]:
        payload = self._load_json(self.status_path)
        if not payload:
            return self._default_status()

        state = str(payload.get("state") or "idle")
        if state in {"running", "stopping"} and not self._is_running():
            payload["state"] = "stopped"
            payload["message"] = payload.get("message") or "Opportunity Map 任务已中断，可继续生成。"
            payload["updated_at"] = _timestamp()
            payload["stop_requested"] = False
            self._write_json(self.status_path, payload)

        payload.setdefault("progress", 0)
        payload.setdefault("completed_sections", 0)
        payload.setdefault("total_sections", 0)
        payload.setdefault("current_section_slug", None)
        payload.setdefault("current_section_label", None)
        payload.setdefault("stage", "idle")
        payload.setdefault("message", "")
        payload.setdefault("error", None)
        payload.setdefault("scanned_core_papers", 0)
        payload.setdefault("total_core_papers", 0)
        payload["cache_exists"] = self.map_path.exists()
        payload["thread_alive"] = self._is_running()
        payload["can_stop"] = payload["state"] in {"running", "stopping"}
        payload["can_continue"] = payload["state"] in {"stopped", "failed"}
        payload["can_start"] = payload["state"] in {"idle", "ready"} and not self._is_running()
        return payload

    def start_or_resume(self, *, model: str | None = None) -> bool:
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
                    "message": "正在扫描核心解读并构建 Opportunity Map。",
                    "error": None,
                    "progress": int(existing.get("progress") or 0) if resumable else 0,
                    "completed_sections": int(existing.get("completed_sections") or 0) if resumable else 0,
                    "total_sections": int(existing.get("total_sections") or 0) if resumable else 0,
                    "current_section_slug": existing.get("current_section_slug"),
                    "current_section_label": existing.get("current_section_label"),
                    "scanned_core_papers": int(existing.get("scanned_core_papers") or 0),
                    "total_core_papers": int(existing.get("total_core_papers") or 0),
                    "stop_requested": False,
                    "model": model,
                },
            )
            self._thread = threading.Thread(
                target=self._rebuild_worker,
                kwargs={"model": model},
                daemon=True,
                name="paper-reader-opportunity-insights",
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
            status["message"] = "正在停止 Opportunity Map 生成任务..."
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
            "cache_exists": self.map_path.exists(),
            "thread_alive": False,
            "message": "",
            "progress": 0,
            "completed_sections": 0,
            "total_sections": 0,
            "current_section_slug": None,
            "current_section_label": None,
            "scanned_core_papers": 0,
            "total_core_papers": 0,
            "can_stop": False,
            "can_continue": False,
            "can_start": True,
        }

    def _rebuild_worker(self, *, model: str) -> None:
        try:
            self._execute_rebuild(model=model)
        except InterruptedError:
            self._mark_stopped("Opportunity Map 任务已停止，可稍后继续。")
        except Exception as exc:
            status = self._load_json(self.status_path)
            status.update(
                {
                    "state": "failed",
                    "stage": "failed",
                    "updated_at": _timestamp(),
                    "message": "Opportunity Map 生成失败。",
                    "error": str(exc),
                    "stop_requested": False,
                }
            )
            self._write_json(self.status_path, status)
        finally:
            with self._lock:
                self._thread = None
                self._stop_requested = False

    def _execute_rebuild(self, *, model: str) -> None:
        self._update_status(
            stage="scan",
            progress=0,
            message="正在扫描已有核心解读。",
            scanned_core_papers=0,
            total_core_papers=0,
        )
        papers = load_momentum_papers(self.library_root, progress_callback=self._scan_progress_callback())
        anchor_date = select_anchor_date(papers)
        history_payload = self._load_json(self.cache_dir / "history_timeline.json")
        momentum_payload = self._load_json(self.cache_dir / MOMENTUM_CACHE_FILE_NAME)
        self._update_status(
            stage="scan",
            progress=SCAN_PROGRESS,
            message=f"已扫描 {len(papers)}/{len(papers)} 篇核心解读。",
            scanned_core_papers=len(papers),
            total_core_papers=len(papers),
        )
        self._check_stop()

        candidates_by_section = build_opportunity_candidates(papers, history_payload, momentum_payload, anchor_date=anchor_date)
        self._update_status(
            stage="section_selection",
            progress=SECTION_SELECTION_PROGRESS,
            completed_sections=0,
            total_sections=len(OPPORTUNITY_SECTIONS),
            message="正在组织机会地图的候选机会。",
        )
        self._check_stop()

        existing_payload = self.load_map() or {}
        existing_sections = {
            str(item.get("slug")): item
            for item in existing_payload.get("sections", [])
            if isinstance(item, dict) and item.get("slug")
        }
        result_sections: list[dict[str, Any]] = []

        for index, section in enumerate(OPPORTUNITY_SECTIONS, start=1):
            candidates = candidates_by_section.get(section.slug, [])
            current_progress = SECTION_SELECTION_PROGRESS + int((index - 1) / max(1, len(OPPORTUNITY_SECTIONS)) * (WRITE_PROGRESS - SECTION_SELECTION_PROGRESS))
            self._update_status(
                stage="section_generation",
                progress=current_progress,
                message=f"正在生成机会模块：{section.label}",
                current_section_slug=section.slug,
                current_section_label=section.label,
                total_sections=len(OPPORTUNITY_SECTIONS),
            )
            self._check_stop()

            cached = existing_sections.get(section.slug)
            if (
                cached
                and int(cached.get("candidate_count") or 0) == len(candidates)
                and str(cached.get("anchor_date") or "") == anchor_date.isoformat()
            ):
                section_payload = cached
            else:
                section_payload = build_opportunity_section_payload(
                    section,
                    candidates,
                    papers,
                    history_payload=history_payload,
                    momentum_payload=momentum_payload,
                    anchor_date=anchor_date,
                    library_root=self.library_root,
                    model=model,
                    should_abort=self._should_abort,
                    progress_callback=self._section_progress_callback(index=index, total=len(OPPORTUNITY_SECTIONS), label=section.label),
                )
            result_sections.append(section_payload)
            partial_payload = {
                "generated_at": _timestamp(),
                "anchor_date": anchor_date.isoformat(),
                "source_paper_count": len(papers),
                "section_count": len(result_sections),
                "history_theme_count": len(history_payload.get("themes", [])) if isinstance(history_payload.get("themes"), list) else 0,
                "momentum_window_count": len(momentum_payload.get("windows", [])) if isinstance(momentum_payload.get("windows"), list) else 0,
                "sections": result_sections,
            }
            self._write_json(self.map_path, partial_payload)
            next_progress = SECTION_SELECTION_PROGRESS + int(index / max(1, len(OPPORTUNITY_SECTIONS)) * (WRITE_PROGRESS - SECTION_SELECTION_PROGRESS))
            self._update_status(
                completed_sections=index,
                progress=next_progress,
                message=f"已完成 {index}/{len(OPPORTUNITY_SECTIONS)} 个机会模块。",
            )
            self._check_stop()

        final_payload = {
            "generated_at": _timestamp(),
            "anchor_date": anchor_date.isoformat(),
            "source_paper_count": len(papers),
            "section_count": len(result_sections),
            "history_theme_count": len(history_payload.get("themes", [])) if isinstance(history_payload.get("themes"), list) else 0,
            "momentum_window_count": len(momentum_payload.get("windows", [])) if isinstance(momentum_payload.get("windows"), list) else 0,
            "sections": result_sections,
        }
        self._update_status(stage="writing", progress=WRITE_PROGRESS, message="正在写入 Opportunity Map 结果。")
        self._write_json(self.map_path, final_payload)
        self._write_json(
            self.status_path,
            {
                "state": "ready",
                "stage": "complete",
                "started_at": self._load_json(self.status_path).get("started_at") or _timestamp(),
                "updated_at": _timestamp(),
                "message": "Opportunity Map 已生成。",
                "error": None,
                "progress": 100,
                "completed_sections": len(OPPORTUNITY_SECTIONS),
                "total_sections": len(OPPORTUNITY_SECTIONS),
                "current_section_slug": OPPORTUNITY_SECTIONS[-1].slug,
                "current_section_label": OPPORTUNITY_SECTIONS[-1].label,
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

    def _section_progress_callback(self, *, index: int, total: int, label: str):
        base = SECTION_SELECTION_PROGRESS + (index - 1) / max(1, total) * (WRITE_PROGRESS - SECTION_SELECTION_PROGRESS)
        width = (WRITE_PROGRESS - SECTION_SELECTION_PROGRESS) / max(1, total)

        def callback(step_progress: int, message: str) -> None:
            normalized = max(0, min(step_progress, 99)) / 100
            overall = int(base + width * normalized)
            self._update_status(
                progress=max(overall, 1),
                stage="section_generation",
                message=f"{label}：{message}",
            )
            self._check_stop()

        return callback

    def _should_abort(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _check_stop(self) -> None:
        if self._should_abort():
            raise InterruptedError("Opportunity insights task interrupted.")

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


def build_opportunity_candidates(
    papers: list[MomentumPaper],
    history_payload: dict[str, Any],
    momentum_payload: dict[str, Any],
    *,
    anchor_date: datetime.date,
) -> dict[str, list[dict[str, Any]]]:
    recent_90 = papers_for_window(papers, anchor_date, 90)
    recent_30 = papers_for_window(papers, anchor_date, 30)
    theme_support = build_theme_support(papers)
    history_themes = history_payload.get("themes") if isinstance(history_payload.get("themes"), list) else []
    momentum_windows = momentum_payload.get("windows") if isinstance(momentum_payload.get("windows"), list) else []

    return {
        "unresolved_problems": build_unresolved_candidates(history_themes, theme_support, recent_90),
        "weak_claims": build_weak_claim_candidates(recent_90),
        "crowded_spaces": build_crowded_candidates(papers, momentum_windows, recent_90),
        "high_upside_edges": build_high_upside_candidates(papers, momentum_windows, recent_30, recent_90),
        "transfer_gaps": build_transfer_gap_candidates(papers, theme_support, recent_90),
    }


def build_opportunity_section_payload(
    section: OpportunitySectionDefinition,
    candidates: list[dict[str, Any]],
    papers: list[MomentumPaper],
    *,
    history_payload: dict[str, Any],
    momentum_payload: dict[str, Any],
    anchor_date: datetime.date,
    library_root: Path,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    llm_payload = synthesize_section(
        section,
        candidates,
        history_payload=history_payload,
        momentum_payload=momentum_payload,
        anchor_date=anchor_date,
        library_root=library_root,
        model=model,
        should_abort=should_abort,
        progress_callback=progress_callback,
    )
    merged_items = merge_section_items(candidates, llm_payload.get("items", []))
    return {
        "slug": section.slug,
        "label": section.label,
        "description": section.description,
        "summary": llm_payload.get("summary") or fallback_section_summary(section, merged_items),
        "candidate_count": len(candidates),
        "item_count": len(merged_items),
        "items": merged_items,
        "anchor_date": anchor_date.isoformat(),
        "llm_used": bool(llm_payload.get("llm_used")),
    }


def synthesize_section(
    section: OpportunitySectionDefinition,
    candidates: list[dict[str, Any]],
    *,
    history_payload: dict[str, Any],
    momentum_payload: dict[str, Any],
    anchor_date: datetime.date,
    library_root: Path,
    model: str,
    should_abort: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {
            "summary": f"当前还没有足够信号来形成“{section.label}”的明确判断。",
            "items": [],
            "llm_used": False,
        }

    paper_pool = collect_section_prompt_papers(candidates)
    paper_ids = {paper.rel_path: f"P{index:02d}" for index, paper in enumerate(paper_pool, start=1)}
    paper_lines: list[str] = []
    for paper in paper_pool:
        paper_lines.extend(
            [
                f"- {paper_ids[paper.rel_path]}",
                f"  date: {paper.sort_date or 'unknown'}",
                f"  title: {paper.title}",
                f"  summary: {paper.summary}",
            ]
        )

    candidate_lines: list[str] = []
    for item in candidates:
        refs = ", ".join(paper_ids.get(paper["rel_path"], "") for paper in item.get("papers", []) if paper_ids.get(paper["rel_path"]))
        candidate_lines.extend(
            [
                f"- candidate_key: {item['key']}",
                f"  title: {item['title']}",
                f"  stats: {item.get('stats_line', 'none')}",
                f"  default_why: {item.get('why', '')}",
                f"  default_evidence: {item.get('evidence', '')}",
                f"  default_next_bet: {item.get('next_bet', '')}",
                f"  refs: {refs or 'none'}",
            ]
        )

    history_context = build_history_context(history_payload)
    momentum_context = build_momentum_context(momentum_payload)
    candidate_block = "\n".join(candidate_lines)
    paper_block = "\n".join(paper_lines)

    prompt = (
        "你是 AutoResearch 的 Opportunity Map 分析器。\n"
        "你的目标不是继续多读几篇，而是告诉研究者：接下来什么值得做。\n\n"
        f"当前模块：{section.label}\n"
        f"模块目标：{section.prompt_focus}\n\n"
        "要求：\n"
        "1. 只能从给定候选项里挑选，不要发明新的候选 key。\n"
        "2. 输出必须是 JSON object，不要带 Markdown 代码块。\n"
        "3. items 最多保留 4 个，优先选择最有价值的。\n"
        "4. why 解释为什么它是机会 / 风险 / gap；evidence 解释你依赖的信号；next_bet 给出下一步值得做的研究下注。\n\n"
        "输出 JSON schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "items": [{"candidate_key": "...", "why": "...", "evidence": "...", "next_bet": "..."}]\n'
        "}\n\n"
        f"历史脉络上下文：\n{history_context}\n\n"
        f"Momentum 上下文：\n{momentum_context}\n\n"
        f"机会候选：\n{candidate_block}\n\n"
        f"支撑论文：\n{paper_block}\n\n"
        f"锚点日期：{anchor_date.isoformat()}\n"
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
            raise InterruptedError("Opportunity synthesis interrupted.")

    return {
        "summary": fallback_section_summary(section, candidates),
        "items": [],
        "llm_used": False,
    }


def build_theme_support(papers: list[MomentumPaper]) -> dict[str, list[MomentumPaper]]:
    support: dict[str, list[MomentumPaper]] = {}
    for theme in THEMES:
        matched = [paper for paper in papers if theme.label in paper.topics]
        if matched:
            support[theme.label] = matched
    return support


def build_unresolved_candidates(
    history_themes: list[Any],
    theme_support: dict[str, list[MomentumPaper]],
    recent_90: list[MomentumPaper],
) -> list[dict[str, Any]]:
    recent_theme_counts: dict[str, int] = {
        label: sum(1 for paper in recent_90 if label in paper.topics)
        for label in theme_support
    }
    candidates: list[dict[str, Any]] = []
    for theme in history_themes:
        if not isinstance(theme, dict):
            continue
        label = str(theme.get("label") or "").strip()
        if not label:
            continue
        lifetime_count = int(theme.get("paper_count") or len(theme_support.get(label, [])))
        recent_count = recent_theme_counts.get(label, 0)
        if lifetime_count < 5:
            continue
        support = select_support_papers(theme_support.get(label, []), limit=MAX_SECTION_SUPPORT_PAPERS)
        candidates.append(
            {
                "key": f"unresolved-{slugify(label)}",
                "title": f"{label} 方向里的关键未解问题",
                "stats_line": f"历史累计 {lifetime_count} 篇；最近 90 天 {recent_count} 篇",
                "why": f"{label} 这条线已经积累了大量工作，但最近仍在持续升温，说明核心瓶颈并没有真正解决。",
                "evidence": theme.get("history_summary") or theme.get("why_it_emerged") or f"{label} 在历史脉络里长期存在。",
                "next_bet": f"优先找出 {label} 当前最容易被忽略、但直接限制真实落地的一环，并围绕它设计更硬的验证。",
                "score": recent_count * 4 + lifetime_count,
                "papers": [momentum_paper_to_view(paper) for paper in support],
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["title"]))
    return candidates[:MAX_SECTION_ITEMS]


def build_weak_claim_candidates(recent_90: list[MomentumPaper]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label, keywords in CLAIM_DEFINITIONS:
        claim_papers = [paper for paper in recent_90 if any(keyword.lower() in (paper.summary or "").lower() for keyword in keywords)]
        if len(claim_papers) < 2:
            continue
        benchmark_depth = sum(len(paper.benchmark_signals) for paper in claim_papers)
        evidence_gap = len(claim_papers) * 2 - benchmark_depth
        if evidence_gap <= 0:
            continue
        support = select_support_papers(claim_papers, limit=MAX_SECTION_SUPPORT_PAPERS)
        candidates.append(
            {
                "key": f"weak-claim-{slugify(label)}",
                "title": label,
                "stats_line": f"最近 90 天被提及 {len(claim_papers)} 篇；评测信号累计 {benchmark_depth}",
                "why": f"{label} 相关说法最近被频繁提起，但直接支撑它的评测或数据仍显得偏薄。",
                "evidence": f"最近 90 天共有 {len(claim_papers)} 篇论文重复触及这一 claim，但对应 benchmark / dataset / executable evidence 密度不高。",
                "next_bet": f"不要继续重复口号，而是先把 {label} 拆成可被独立验证的子命题，再补出缺的评测。",
                "score": evidence_gap * 5 + len(claim_papers),
                "papers": [momentum_paper_to_view(paper) for paper in support],
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["title"]))
    return candidates[:MAX_SECTION_ITEMS]


def build_crowded_candidates(
    papers: list[MomentumPaper],
    momentum_windows: list[Any],
    recent_90: list[MomentumPaper],
) -> list[dict[str, Any]]:
    lifetime_topic_counts: dict[str, int] = {}
    for theme in THEMES:
        lifetime_topic_counts[theme.label] = sum(1 for paper in papers if theme.label in paper.topics)
    recent_topic_counts: dict[str, int] = {label: sum(1 for paper in recent_90 if label in paper.topics) for label in lifetime_topic_counts}
    momentum_hot = {}
    for window in momentum_windows:
        if not isinstance(window, dict):
            continue
        for item in window.get("hot_topics", []):
            if isinstance(item, dict) and item.get("name"):
                momentum_hot[str(item.get("name"))] = max(momentum_hot.get(str(item.get("name")), 0), int(item.get("count") or 0))
        for item in window.get("method_routes", []):
            if isinstance(item, dict) and item.get("name"):
                momentum_hot[str(item.get("name"))] = max(momentum_hot.get(str(item.get("name")), 0), int(item.get("count") or 0))

    candidates: list[dict[str, Any]] = []
    for label, lifetime_count in lifetime_topic_counts.items():
        recent_count = recent_topic_counts.get(label, 0)
        hot_boost = momentum_hot.get(label, 0)
        if lifetime_count < 6 or recent_count <= 0:
            continue
        support = select_support_papers([paper for paper in recent_90 if label in paper.topics], limit=MAX_SECTION_SUPPORT_PAPERS)
        candidates.append(
            {
                "key": f"crowded-{slugify(label)}",
                "title": label,
                "stats_line": f"历史累计 {lifetime_count} 篇；最近 90 天 {recent_count} 篇；动量峰值 {hot_boost}",
                "why": f"{label} 既有长期积累，又在最近窗口继续升温，这通常意味着它已经进入竞争变密的阶段。",
                "evidence": f"历史上 {label} 已有大量论文，最近 90 天里仍不断出现 follow 工作。",
                "next_bet": f"如果继续做 {label}，最好不要只做局部优化，而要在问题设定、评测标准或跨模块耦合上做出明显差异。",
                "score": lifetime_count + recent_count * 4 + hot_boost * 3,
                "papers": [momentum_paper_to_view(paper) for paper in support],
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["title"]))
    return candidates[:MAX_SECTION_ITEMS]


def build_high_upside_candidates(
    papers: list[MomentumPaper],
    momentum_windows: list[Any],
    recent_30: list[MomentumPaper],
    recent_90: list[MomentumPaper],
) -> list[dict[str, Any]]:
    lifetime_topic_counts: dict[str, int] = {}
    for theme in THEMES:
        lifetime_topic_counts[theme.label] = sum(1 for paper in papers if theme.label in paper.topics)

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for window in momentum_windows:
        if not isinstance(window, dict):
            continue
        for item in window.get("emerging_edges", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item.get("name"))
            key = f"high-upside-{slugify(name)}"
            if key in seen:
                continue
            seen.add(key)
            support_rel_paths = {str(paper.get("rel_path") or "") for paper in item.get("papers", []) if isinstance(paper, dict)}
            support = [paper for paper in recent_90 if paper.rel_path in support_rel_paths]
            candidates.append(
                {
                    "key": key,
                    "title": name,
                    "stats_line": f"最近 {item.get('count', 0)} 篇；前窗 {item.get('previous_count', 0)}；累计 {item.get('lifetime_count', 0)}",
                    "why": str(item.get("why") or f"{name} 最近开始聚集，但总体密度仍低。"),
                    "evidence": f"最近窗口里 {name} 的信号开始连续出现，但历史累计量仍然不高。",
                    "next_bet": f"优先在 {name} 上做更硬的问题定义或验证闭环，在人还不多的时候形成先发优势。",
                    "score": float(item.get("momentum_score") or 0),
                    "papers": [momentum_paper_to_view(paper) for paper in select_support_papers(support, limit=MAX_SECTION_SUPPORT_PAPERS)],
                }
            )

    if len(candidates) < MAX_SECTION_ITEMS:
        for label, lifetime_count in lifetime_topic_counts.items():
            recent_count = sum(1 for paper in recent_30 if label in paper.topics)
            if lifetime_count == 0 or lifetime_count > 6 or recent_count == 0:
                continue
            key = f"high-upside-theme-{slugify(label)}"
            if key in seen:
                continue
            seen.add(key)
            support = [paper for paper in recent_90 if label in paper.topics]
            candidates.append(
                {
                    "key": key,
                    "title": label,
                    "stats_line": f"历史累计 {lifetime_count} 篇；最近 30 天 {recent_count} 篇",
                    "why": f"{label} 目前研究密度还不高，但最近开始持续出现，说明可能正处在早期窗口。",
                    "evidence": f"{label} 的历史累计样本不多，但近期已开始形成连续信号。",
                    "next_bet": f"在 {label} 方向尽快建立问题定义和早期 benchmark，有机会拿到更高的研究杠杆。",
                    "score": recent_count * 6 + max(0, 8 - lifetime_count),
                    "papers": [momentum_paper_to_view(paper) for paper in select_support_papers(support, limit=MAX_SECTION_SUPPORT_PAPERS)],
                }
            )

    candidates.sort(key=lambda item: (-item["score"], item["title"]))
    return candidates[:MAX_SECTION_ITEMS]


def build_transfer_gap_candidates(
    papers: list[MomentumPaper],
    theme_support: dict[str, list[MomentumPaper]],
    recent_90: list[MomentumPaper],
) -> list[dict[str, Any]]:
    method_labels = [definition.label for definition in METHOD_SIGNAL_DEFINITIONS]
    method_totals = {label: sum(1 for paper in papers if label in paper.methods) for label in method_labels}
    candidates: list[dict[str, Any]] = []
    for theme_label, theme_papers in theme_support.items():
        theme_total = len(theme_papers)
        theme_recent = sum(1 for paper in recent_90 if theme_label in paper.topics)
        if theme_total < 5:
            continue
        for method_label in method_labels:
            method_total = method_totals.get(method_label, 0)
            if method_total < 5:
                continue
            pair_count = sum(1 for paper in theme_papers if method_label in paper.methods)
            if pair_count > max(1, int(theme_total * 0.2)):
                continue
            donor_theme, donor_count = best_donor_theme(theme_support, method_label, exclude=theme_label)
            if donor_count < 3:
                continue
            donor_support = [paper for paper in theme_support.get(donor_theme, []) if method_label in paper.methods]
            target_support = [paper for paper in recent_90 if theme_label in paper.topics]
            merged_support = select_support_papers(donor_support + target_support, limit=MAX_SECTION_SUPPORT_PAPERS)
            candidates.append(
                {
                    "key": f"transfer-{slugify(method_label)}-to-{slugify(theme_label)}",
                    "title": f"把 {method_label} 迁移到 {theme_label}",
                    "stats_line": f"方法累计 {method_total} 篇；{theme_label} 累计 {theme_total} 篇；当前组合仅 {pair_count} 篇；最近 90 天目标主题 {theme_recent} 篇",
                    "why": f"{method_label} 已经在 {donor_theme} 里证明了价值，但在 {theme_label} 这条线里仍几乎没有被系统迁移。",
                    "evidence": f"方法本身已有大量工作，目标主题也持续活跃，但二者组合密度明显偏低。",
                    "next_bet": f"优先把 {method_label} 的核心机制移植到 {theme_label}，并设计能直接验证迁移收益的对照实验。",
                    "score": donor_count * 4 + method_total + theme_recent * 3 - pair_count * 6,
                    "papers": [momentum_paper_to_view(paper) for paper in merged_support],
                }
            )
    candidates.sort(key=lambda item: (-item["score"], item["title"]))
    return candidates[:MAX_SECTION_ITEMS]


def best_donor_theme(theme_support: dict[str, list[MomentumPaper]], method_label: str, *, exclude: str) -> tuple[str, int]:
    donor_theme = ""
    donor_count = 0
    for label, theme_papers in theme_support.items():
        if label == exclude:
            continue
        count = sum(1 for paper in theme_papers if method_label in paper.methods)
        if count > donor_count:
            donor_theme = label
            donor_count = count
    return donor_theme, donor_count


def collect_section_prompt_papers(candidates: list[dict[str, Any]]) -> list[MomentumPaper]:
    paper_map: dict[str, MomentumPaper] = {}
    for item in candidates:
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
                methods=list(paper.get("methods") or []),
                benchmark_signals=list(paper.get("signals") or []),
                novelty_score=0,
                turning_score=0,
            )
    return sorted(paper_map.values(), key=lambda paper: (paper.sort_date or "0000-00-00", paper.title.lower()), reverse=True)[:18]


def merge_section_items(candidates: list[dict[str, Any]], llm_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        str(item.get("candidate_key")): item
        for item in llm_items
        if isinstance(item, dict) and item.get("candidate_key")
    }
    merged: list[dict[str, Any]] = []
    for candidate in candidates[:MAX_SECTION_ITEMS]:
        item = dict(candidate)
        llm_item = by_key.get(candidate["key"])
        if llm_item:
            item["why"] = str(llm_item.get("why") or item.get("why") or "")
            item["evidence"] = str(llm_item.get("evidence") or item.get("evidence") or "")
            item["next_bet"] = str(llm_item.get("next_bet") or item.get("next_bet") or "")
        merged.append(item)
    return merged


def build_history_context(history_payload: dict[str, Any]) -> str:
    themes = history_payload.get("themes") if isinstance(history_payload.get("themes"), list) else []
    if not themes:
        return "暂无历史脉络缓存。"
    lines: list[str] = []
    for theme in themes[:6]:
        if not isinstance(theme, dict):
            continue
        lines.append(
            f"- {theme.get('label', 'unknown')}: {theme.get('paper_count', 0)} 篇；summary={theme.get('history_summary', '')}"
        )
    return "\n".join(lines) if lines else "暂无历史脉络缓存。"


def build_momentum_context(momentum_payload: dict[str, Any]) -> str:
    windows = momentum_payload.get("windows") if isinstance(momentum_payload.get("windows"), list) else []
    if not windows:
        return "暂无 Momentum Radar 缓存。"
    lines: list[str] = []
    for window in windows[:3]:
        if not isinstance(window, dict):
            continue
        hot = ", ".join(item.get("name", "") for item in window.get("hot_topics", [])[:2] if isinstance(item, dict) and item.get("name")) or "none"
        method = ", ".join(item.get("name", "") for item in window.get("method_routes", [])[:2] if isinstance(item, dict) and item.get("name")) or "none"
        edge = ", ".join(item.get("name", "") for item in window.get("emerging_edges", [])[:2] if isinstance(item, dict) and item.get("name")) or "none"
        lines.append(f"- {window.get('label', 'window')}: hot={hot}; method={method}; edge={edge}")
    return "\n".join(lines) if lines else "暂无 Momentum Radar 缓存。"


def fallback_section_summary(section: OpportunitySectionDefinition, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"当前还没有足够信号来形成“{section.label}”的明确判断。"
    titles = ", ".join(item["title"] for item in items[:2])
    return f"在“{section.label}”这个维度，当前最值得关注的机会集中在 {titles}。"


def slugify(text: str) -> str:
    lowered = text.lower()
    output = []
    for char in lowered:
        if char.isalnum():
            output.append(char)
        else:
            output.append("-")
    slug = "".join(output)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
