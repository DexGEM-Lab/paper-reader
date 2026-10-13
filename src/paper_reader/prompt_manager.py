from __future__ import annotations

import json
import sqlite3
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .ai_summary import DEFAULT_MODEL, DEFAULT_USER_PROMPT

PROMPT_STORE_NAME = ".paper-reader-prompts.json"
DEFAULT_PROMPT_SLUG = "core-zh"
LEGACY_DEFAULT_USER_PROMPT = (
    "请直接阅读本地论文文件 `{document_path}`，不要先把论文分段再汇总。\n\n"
    "请用通俗、准确、结构化的中文解释这篇论文，重点回答：\n"
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
)
_SAFE_BUILTIN_PROMPT_UPDATES: dict[str, tuple[str, ...]] = {
    DEFAULT_PROMPT_SLUG: (LEGACY_DEFAULT_USER_PROMPT,),
}
DEFAULT_TAG_PROMPT = (
    "请根据下面提供的论文正文生成 3 到 8 个适合团队检索的英文标签。\n"
    "要求：\n"
    "- 标签尽量短，小写英文\n"
    "- 优先提取方法、任务、模态、应用领域、机构或模型名\n"
    "- 可以包含连字符或空格，例如 `on-policy distillation`、`long-context`\n"
    "- 避免输出过于空泛的词，例如 model、models、training、paper、method、approach\n"
    "- 如果不确定，就不要硬猜\n\n"
    "只输出 JSON 数组，不要附加解释。示例：\n"
    "[\"lora\", \"finance\", \"coding\", \"deepseek\"]\n"
)


@dataclass
class PromptDefinition:
    slug: str
    name: str
    user_prompt: str
    model: str
    enabled: bool
    auto_run: bool
    created_at: str
    updated_at: str
    prompt_id: int | None = None
    version_id: int | None = None
    version: int = 1
    admin_only: bool = True


@dataclass
class TagPromptDefinition:
    user_prompt: str
    model: str
    enabled: bool
    created_at: str
    updated_at: str


class PromptStore:
    def __init__(self, library_root: Path, db_path: Path):
        self.library_root = library_root.resolve()
        self.store_path = self.library_root / PROMPT_STORE_NAME
        self.db_path = db_path.resolve()
        self._ensure_schema()
        self._migrate_legacy_prompts_if_needed()
        self._sync_builtin_prompts_if_safe()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    auto_run INTEGER NOT NULL DEFAULT 1,
                    admin_only INTEGER NOT NULL DEFAULT 1,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    user_prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE (prompt_id, version),
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tag_prompt_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL DEFAULT 0,
                    user_prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by_user_id INTEGER
                );
                """
            )

    def _migrate_legacy_prompts_if_needed(self) -> None:
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM prompts WHERE is_deleted = 0").fetchone()[0])
        if count > 0:
            return

        payload = self._load_legacy_payload()
        prompts = payload.get("prompts", []) if isinstance(payload.get("prompts"), list) else []
        migrated_any = False
        for item in prompts:
            prompt = self._coerce_legacy_prompt(item)
            if prompt is None:
                continue
            self._insert_prompt(prompt, created_by_user_id=None)
            migrated_any = True

        if not migrated_any:
            self._insert_prompt(self.default_prompt(), created_by_user_id=None)

    def _sync_builtin_prompts_if_safe(self) -> None:
        builtin_defaults = {
            DEFAULT_PROMPT_SLUG: self.default_prompt(),
        }
        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            for slug, default_prompt in builtin_defaults.items():
                legacy_versions = _SAFE_BUILTIN_PROMPT_UPDATES.get(slug, ())
                if not legacy_versions:
                    continue
                row = conn.execute(
                    """
                    SELECT p.id AS prompt_id,
                           p.name,
                           p.enabled,
                           p.auto_run,
                           p.admin_only,
                           p.created_at,
                           p.updated_at,
                           pv.version,
                           pv.user_prompt,
                           pv.model
                    FROM prompts p
                    JOIN prompt_versions pv
                      ON pv.id = (
                          SELECT latest.id
                          FROM prompt_versions latest
                          WHERE latest.prompt_id = p.id
                          ORDER BY latest.version DESC, latest.id DESC
                          LIMIT 1
                      )
                    WHERE p.slug = ? AND p.is_deleted = 0
                    """,
                    (slug,),
                ).fetchone()
                if row is None:
                    continue
                current_prompt = str(row["user_prompt"] or "")
                if current_prompt == default_prompt.user_prompt:
                    continue
                if current_prompt not in legacy_versions:
                    continue
                next_version = int(row["version"]) + 1
                prompt_id = int(row["prompt_id"])
                conn.execute(
                    "UPDATE prompts SET updated_at = ? WHERE id = ?",
                    (now, prompt_id),
                )
                conn.execute(
                    """
                    INSERT INTO prompt_versions(prompt_id, version, user_prompt, model, created_by_user_id, created_at)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        prompt_id,
                        next_version,
                        default_prompt.user_prompt,
                        default_prompt.model,
                        now,
                    ),
                )

    def list_prompts(self) -> list[PromptDefinition]:
        rows = self._list_prompt_rows()
        if rows:
            return rows
        default_prompt = self.default_prompt()
        self._insert_prompt(default_prompt, created_by_user_id=None)
        return self._list_prompt_rows()

    def default_prompt(self) -> PromptDefinition:
        now = datetime.utcnow().isoformat(timespec="seconds")
        return PromptDefinition(
            slug=DEFAULT_PROMPT_SLUG,
            name="核心解读",
            user_prompt=DEFAULT_USER_PROMPT,
            model=DEFAULT_MODEL,
            enabled=True,
            auto_run=True,
            created_at=now,
            updated_at=now,
            admin_only=True,
        )

    def get_prompt(self, slug: str) -> PromptDefinition | None:
        for prompt in self.list_prompts():
            if prompt.slug == slug:
                return prompt
        return None

    def default_tag_prompt(self) -> TagPromptDefinition:
        now = datetime.utcnow().isoformat(timespec="seconds")
        return TagPromptDefinition(
            user_prompt=DEFAULT_TAG_PROMPT,
            model=DEFAULT_MODEL,
            enabled=True,
            created_at=now,
            updated_at=now,
        )

    def get_tag_prompt(self) -> TagPromptDefinition:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT enabled, user_prompt, model, created_at, updated_at FROM tag_prompt_config WHERE id = 1"
            ).fetchone()
            if row is None:
                prompt = self.default_tag_prompt()
                conn.execute(
                    """
                    INSERT INTO tag_prompt_config(id, enabled, user_prompt, model, created_at, updated_at, updated_by_user_id)
                    VALUES (1, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        1 if prompt.enabled else 0,
                        prompt.user_prompt,
                        prompt.model,
                        prompt.created_at,
                        prompt.updated_at,
                    ),
                )
                return prompt
        return TagPromptDefinition(
            user_prompt=str(row["user_prompt"]),
            model=str(row["model"] or DEFAULT_MODEL),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def save_tag_prompt(
        self,
        *,
        user_prompt: str,
        model: str,
        enabled: bool,
        updated_by_user_id: int | None = None,
    ) -> TagPromptDefinition:
        user_prompt = user_prompt.strip()
        model = model.strip() or DEFAULT_MODEL
        if not user_prompt:
            raise ValueError("标签 Prompt 内容不能为空。")
        existing = self.get_tag_prompt()
        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tag_prompt_config(id, enabled, user_prompt, model, created_at, updated_at, updated_by_user_id)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled = excluded.enabled,
                    user_prompt = excluded.user_prompt,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    updated_by_user_id = excluded.updated_by_user_id
                """,
                (
                    1 if enabled else 0,
                    user_prompt,
                    model,
                    existing.created_at,
                    now,
                    updated_by_user_id,
                ),
            )
        return self.get_tag_prompt()

    def active_prompts(self) -> list[PromptDefinition]:
        return [prompt for prompt in self.list_prompts() if prompt.enabled]

    def auto_prompts(self) -> list[PromptDefinition]:
        return [prompt for prompt in self.active_prompts() if prompt.auto_run]

    def save_prompt(
        self,
        *,
        existing_slug: str | None,
        name: str,
        slug: str,
        user_prompt: str,
        model: str,
        enabled: bool,
        auto_run: bool,
        admin_only: bool = True,
        created_by_user_id: int | None = None,
    ) -> PromptDefinition:
        name = name.strip()
        requested_slug = slug.strip()
        user_prompt = user_prompt.strip()
        model = model.strip() or DEFAULT_MODEL
        if not name:
            raise ValueError("Prompt 名称不能为空。")
        if not user_prompt:
            raise ValueError("Prompt 内容不能为空。")

        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            active_rows = conn.execute(
                "SELECT slug FROM prompts WHERE is_deleted = 0"
            ).fetchall()
            used_slugs = {str(row["slug"]) for row in active_rows}
            if existing_slug:
                used_slugs.discard(existing_slug)
                resolved_slug = existing_slug.strip()
            else:
                resolved_slug = self._choose_slug(name=name, requested_slug=requested_slug, used_slugs=used_slugs)

            if existing_slug:
                prompt_row = conn.execute(
                    "SELECT id, created_at FROM prompts WHERE slug = ? AND is_deleted = 0",
                    (existing_slug,),
                ).fetchone()
                if prompt_row is None:
                    raise FileNotFoundError(existing_slug)
                prompt_id = int(prompt_row["id"])
                version_row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS version FROM prompt_versions WHERE prompt_id = ?",
                    (prompt_id,),
                ).fetchone()
                next_version = int(version_row["version"]) + 1
                conn.execute(
                    """
                    UPDATE prompts
                    SET name = ?, enabled = ?, auto_run = ?, admin_only = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, 1 if enabled else 0, 1 if auto_run else 0, 1 if admin_only else 0, now, prompt_id),
                )
                conn.execute(
                    """
                    INSERT INTO prompt_versions(prompt_id, version, user_prompt, model, created_by_user_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (prompt_id, next_version, user_prompt, model, created_by_user_id, now),
                )
            else:
                if resolved_slug in used_slugs:
                    raise ValueError("Prompt 标识已存在，请换一个。")
                cursor = conn.execute(
                    """
                    INSERT INTO prompts(slug, name, enabled, auto_run, admin_only, is_deleted, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (resolved_slug, name, 1 if enabled else 0, 1 if auto_run else 0, 1 if admin_only else 0, now, now),
                )
                prompt_id_raw = cursor.lastrowid
                if prompt_id_raw is None:
                    raise RuntimeError("Failed to persist prompt.")
                prompt_id = int(prompt_id_raw)
                conn.execute(
                    """
                    INSERT INTO prompt_versions(prompt_id, version, user_prompt, model, created_by_user_id, created_at)
                    VALUES (?, 1, ?, ?, ?, ?)
                    """,
                    (prompt_id, user_prompt, model, created_by_user_id, now),
                )
        prompt = self.get_prompt(existing_slug or resolved_slug)
        if prompt is None:
            raise RuntimeError("Failed to load prompt after save.")
        return prompt

    def delete_prompt(self, slug: str) -> PromptDefinition:
        prompt = self.get_prompt(slug)
        if prompt is None:
            raise FileNotFoundError(slug)
        with self._connect() as conn:
            conn.execute(
                "UPDATE prompts SET is_deleted = 1, enabled = 0, updated_at = ? WHERE slug = ?",
                (datetime.utcnow().isoformat(timespec="seconds"), slug),
            )
        return prompt

    def _list_prompt_rows(self) -> list[PromptDefinition]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.id AS prompt_id,
                    p.slug,
                    p.name,
                    p.enabled,
                    p.auto_run,
                    p.admin_only,
                    p.created_at,
                    p.updated_at,
                    pv.id AS version_id,
                    pv.version,
                    pv.user_prompt,
                    pv.model
                FROM prompts p
                JOIN prompt_versions pv
                  ON pv.id = (
                      SELECT latest.id
                      FROM prompt_versions latest
                      WHERE latest.prompt_id = p.id
                      ORDER BY latest.version DESC, latest.id DESC
                      LIMIT 1
                  )
                WHERE p.is_deleted = 0
                ORDER BY
                    CASE WHEN p.slug = ? THEN 0 ELSE 1 END,
                    CASE WHEN p.enabled = 1 THEN 0 ELSE 1 END,
                    LOWER(p.name),
                    p.created_at
                """,
                (DEFAULT_PROMPT_SLUG,),
            ).fetchall()
        return [self._row_to_prompt(row) for row in rows]

    def _row_to_prompt(self, row: sqlite3.Row) -> PromptDefinition:
        return PromptDefinition(
            slug=str(row["slug"]),
            name=str(row["name"]),
            user_prompt=str(row["user_prompt"]),
            model=str(row["model"] or DEFAULT_MODEL),
            enabled=bool(row["enabled"]),
            auto_run=bool(row["auto_run"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            prompt_id=int(row["prompt_id"]),
            version_id=int(row["version_id"]),
            version=int(row["version"]),
            admin_only=bool(row["admin_only"]),
        )

    def _insert_prompt(self, prompt: PromptDefinition, created_by_user_id: int | None) -> None:
        now = prompt.updated_at or datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompts(slug, name, enabled, auto_run, admin_only, is_deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    prompt.slug,
                    prompt.name,
                    1 if prompt.enabled else 0,
                    1 if prompt.auto_run else 0,
                    1 if prompt.admin_only else 0,
                    prompt.created_at or now,
                    now,
                ),
            )
            prompt_id_raw = cursor.lastrowid
            if prompt_id_raw is None:
                raise RuntimeError("Failed to migrate prompt.")
            conn.execute(
                """
                INSERT INTO prompt_versions(prompt_id, version, user_prompt, model, created_by_user_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (int(prompt_id_raw), max(1, int(prompt.version or 1)), prompt.user_prompt, prompt.model, created_by_user_id, now),
            )

    def _coerce_legacy_prompt(self, item: Any) -> PromptDefinition | None:
        if not isinstance(item, dict):
            return None
        try:
            slug = str(item.get("slug", "")).strip()
            name = str(item.get("name", "")).strip()
            user_prompt = str(item.get("user_prompt") or item.get("prompt") or DEFAULT_USER_PROMPT).strip()
            model = str(item.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
            enabled = bool(item.get("enabled", True))
            auto_run = bool(item.get("auto_run", True))
            created_at = str(item.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"))
            updated_at = str(item.get("updated_at") or created_at)
        except Exception:
            return None
        if not slug or not name or not user_prompt:
            return None
        return PromptDefinition(
            slug=slug,
            name=name,
            user_prompt=user_prompt,
            model=model,
            enabled=enabled,
            auto_run=auto_run,
            created_at=created_at,
            updated_at=updated_at,
            admin_only=True,
        )

    def _load_legacy_payload(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _choose_slug(self, *, name: str, requested_slug: str, used_slugs: set[str]) -> str:
        normalized_requested = slugify(requested_slug)
        if normalized_requested:
            if normalized_requested in used_slugs:
                raise ValueError("Prompt 标识已存在，请换一个。")
            return normalized_requested

        normalized_name = slugify(name)
        base = normalized_name or "prompt"
        slug = base
        counter = 2
        while slug in used_slugs:
            slug = f"{base}-{counter}"
            counter += 1
        return slug


_slug_pattern = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = _slug_pattern.sub("-", lowered).strip("-")
    return normalized[:80]


def parse_checkbox(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"on", "1", "true", "yes"}
