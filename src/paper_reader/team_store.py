from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from werkzeug.security import check_password_hash, generate_password_hash

from .arxiv_markdown import base_arxiv_id, normalize_arxiv_id


DEFAULT_DB_NAME = ".paper-reader-team.db"
_AUTO_TAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "application",
    "applications",
    "approach",
    "approaches",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "check",
    "data",
    "dataset",
    "datasets",
    "distillation",
    "for",
    "foundation",
    "framework",
    "frameworks",
    "from",
    "general",
    "in",
    "into",
    "is",
    "language",
    "large",
    "learning",
    "method",
    "methods",
    "model",
    "models",
    "new",
    "network",
    "networks",
    "of",
    "on",
    "or",
    "paper",
    "pdf",
    "reader",
    "research",
    "results",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "teams",
    "that",
    "the",
    "training",
    "ui",
    "using",
    "via",
    "with",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_TITLE_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]{2,}\b")
_LOW_PRIORITY_AUTO_TAGS = {"agent", "distillation", "reasoning"}
_SPECIALIZED_AUTO_TAG_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lora", (r"\blora\b", r"low[ -]?rank adaptation")),
    ("qlora", (r"\bqlora\b",)),
    ("diffusion", (r"\bdiffusion\b", r"\bdiffusion model")),
    ("rectified flow", (r"rectified flow",)),
    ("flow matching", (r"flow matching",)),
    ("on-policy distillation", (r"on[- ]policy distillation", r"onpolicy distillation")),
    ("distillation", (r"\bdistill(?:ation|ed)?\b",)),
    ("dpo", (r"\bdpo\b", r"direct preference optimization")),
    ("grpo", (r"\bgrpo\b", r"group relative policy optimization")),
    ("ppo", (r"\bppo\b", r"proximal policy optimization")),
    ("rlhf", (r"\brlhf\b", r"reinforcement learning from human feedback")),
    ("sft", (r"\bsft\b", r"supervised fine[- ]tuning")),
    ("rag", (r"\brag\b", r"retrieval[- ]augmented generation")),
    ("moe", (r"\bmoe\b", r"mixture of experts")),
    ("long-context", (r"long[- ]context", r"long context")),
    ("reasoning", (r"\breasoning\b",)),
    ("agent", (r"\bagent(?:ic|s)?\b",)),
    ("multimodal", (r"\bmultimodal\b", r"vision[- ]language", r"vision language")),
    ("vision", (r"\bvision\b", r"computer vision", r"image generation", r"image understanding")),
    ("video", (r"\bvideo\b",)),
    ("audio", (r"\baudio\b",)),
    ("speech", (r"\bspeech\b",)),
    ("robotics", (r"\brobotics?\b",)),
    ("coding", (r"\bcoding\b", r"\bcode\b", r"program synthesis", r"software engineering")),
    ("math", (r"\bmath(?:ematical)?\b", r"\bgeometry\b", r"\balgebra\b", r"\btheorem\b")),
    ("finance", (r"\bfinanc(?:e|ial)\b", r"\btrading\b", r"\bquant\b")),
    ("healthcare", (r"\bmedical\b", r"\bclinical\b", r"\bhealthcare\b", r"\bpatient\b")),
    ("biology", (r"\bbiology\b", r"\bbiomedical\b", r"\bprotein\b", r"\bgene\b")),
    ("deepseek", (r"\bdeepseek\b",)),
    ("thu", (r"\btsinghua\b", r"\bthu\b")),
    ("nvidia", (r"\bnvidia\b",)),
    ("openai", (r"\bopenai\b",)),
    ("anthropic", (r"\banthropic\b",)),
    ("deepmind", (r"\bdeepmind\b",)),
    ("google", (r"\bgoogle\b",)),
    ("meta", (r"\bmeta\b",)),
    ("microsoft", (r"\bmicrosoft\b",)),
    ("alibaba", (r"\balibaba\b",)),
    ("bytedance", (r"\bbytedance\b",)),
    ("moonshot", (r"\bmoonshot\b",)),
)


@dataclass(slots=True)
class TeamUser:
    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
    created_at: str
    avatar_rel_path: str | None = None


@dataclass(slots=True)
class TeamTag:
    id: int
    name: str
    slug: str
    source_type: str
    added_by_display_name: str | None
    created_at: str
    is_locked: bool


@dataclass(slots=True)
class TeamRecommendation:
    id: int
    user_id: int
    display_name: str
    reason: str
    created_at: str


@dataclass(slots=True)
class TeamComment:
    id: int
    user_id: int
    display_name: str
    avatar_rel_path: str | None
    body: str
    created_at: str
    parent_id: int | None
    replies: list["TeamComment"]


@dataclass(slots=True)
class TeamPromptRun:
    prompt_slug: str
    prompt_name: str
    prompt_version: int
    model: str
    generated_by_display_name: str | None
    generated_at: str | None
    result_rel_path: str | None
    status: str


@dataclass(slots=True)
class TeamChatMessage:
    id: int
    role: str
    display_name: str
    avatar_rel_path: str | None
    body: str
    created_at: str
    status: str
    model: str | None


@dataclass(slots=True)
class PaperAttachment:
    id: int
    kind: str
    title: str
    local_path: str | None
    source_url: str
    created_at: str


@dataclass(frozen=True, slots=True)
class DiscoveryRecord:
    id: int
    source: str
    external_id: str
    title: str
    url: str
    abstract: str
    score: float
    reason: str
    status: str
    surfaced_at: str
    decided_by: int | None
    decided_at: str | None


class TeamStore:
    def __init__(
        self,
        root: Path,
        db_name: str = DEFAULT_DB_NAME,
        *,
        admin_usernames: Iterable[str] = (),
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / db_name
        self.admin_usernames = frozenset(item.strip().casefold() for item in admin_usernames if item and item.strip())
        self._write_lock = threading.Lock()
        self._ensure_schema()
        self._migrate_likes_into_recommendations()
        self._ensure_configured_admins()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    avatar_rel_path TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_roles (
                    user_id INTEGER NOT NULL,
                    role_id INTEGER NOT NULL,
                    PRIMARY KEY (user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rel_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    title TEXT NOT NULL,
                    display_title TEXT NOT NULL,
                    preview_text TEXT NOT NULL DEFAULT '',
                    extracted_date TEXT,
                    sort_date TEXT,
                    date_precision TEXT,
                    date_source TEXT,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    modified_at TEXT NOT NULL,
                    preview_kind TEXT NOT NULL DEFAULT 'file',
                    resource_kind TEXT NOT NULL DEFAULT 'paper',
                    canonical_url TEXT,
                    source_domain TEXT,
                    fetch_status TEXT NOT NULL DEFAULT 'ok',
                    fetched_at TEXT,
                    author TEXT,
                    site_name TEXT,
                    is_done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    source_url TEXT,
                    source_context TEXT,
                    sender_open_id TEXT,
                    message_time_iso TEXT,
                    imported_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (imported_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS paper_attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    local_path TEXT,
                    source_url TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (paper_id, source_url),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (paper_id, user_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_user_states (
                    paper_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    is_done INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, user_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_likes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (paper_id, user_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    parent_id INTEGER,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    created_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS paper_tags (
                    paper_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    score REAL,
                    added_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (paper_id, tag_id, source_type),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    FOREIGN KEY (added_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    prompt_slug TEXT NOT NULL,
                    prompt_name TEXT NOT NULL,
                    prompt_version_id INTEGER,
                    prompt_version INTEGER NOT NULL DEFAULT 1,
                    model TEXT NOT NULL,
                    triggered_by_user_id INTEGER,
                    status TEXT NOT NULL,
                    shared INTEGER NOT NULL DEFAULT 1,
                    result_rel_path TEXT,
                    generated_at TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE (paper_id, prompt_slug, prompt_version, model, shared),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (triggered_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS chat_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    visibility TEXT NOT NULL,
                    owner_user_id INTEGER,
                    thread_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL,
                    user_id INTEGER,
                    role TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS paper_date_overrides (
                    paper_id INTEGER PRIMARY KEY,
                    display_date TEXT NOT NULL,
                    precision TEXT NOT NULL DEFAULT 'month',
                    sort_date TEXT NOT NULL,
                    updated_by_user_id INTEGER,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (updated_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    abstract TEXT NOT NULL DEFAULT '',
                    score REAL NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new', 'imported', 'dismissed')),
                    surfaced_at TEXT NOT NULL,
                    decided_by INTEGER,
                    decided_at TEXT,
                    UNIQUE(source, external_id),
                    FOREIGN KEY (decided_by) REFERENCES users(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_papers_sort_date ON papers(sort_date);
                CREATE INDEX IF NOT EXISTS idx_papers_display_title ON papers(display_title);
                CREATE INDEX IF NOT EXISTS idx_papers_folder ON papers(folder);
                CREATE INDEX IF NOT EXISTS idx_paper_recommendations_paper_id ON paper_recommendations(paper_id);
                CREATE INDEX IF NOT EXISTS idx_paper_attachments_paper_id ON paper_attachments(paper_id);
                CREATE INDEX IF NOT EXISTS idx_paper_user_states_user_id_done ON paper_user_states(user_id, is_done);
                CREATE INDEX IF NOT EXISTS idx_comments_paper_id ON comments(paper_id);
                CREATE INDEX IF NOT EXISTS idx_prompt_runs_paper_id ON prompt_runs(paper_id);
                CREATE INDEX IF NOT EXISTS idx_paper_sources_paper_id ON paper_sources(paper_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_sources_canonical
                    ON paper_sources(source_type, source_value)
                    WHERE source_type IN ('arxiv', 'openreview', 'pdf_url', 'canonical_url');
                CREATE INDEX IF NOT EXISTS idx_paper_tags_tag_id ON paper_tags(tag_id);
                CREATE INDEX IF NOT EXISTS idx_chat_threads_paper_id ON chat_threads(paper_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);
                CREATE INDEX IF NOT EXISTS idx_discoveries_surfaced_at ON discoveries(surfaced_at DESC);
                CREATE INDEX IF NOT EXISTS idx_discoveries_url ON discoveries(url);
                """
            )
            self._ensure_users_avatar_column(conn)
            self._ensure_papers_date_columns(conn)
            self._ensure_papers_resource_columns(conn)
            self._ensure_paper_sources_columns(conn)
            self._seed_roles(conn)

    def _seed_roles(self, conn: sqlite3.Connection) -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO roles(slug, name) VALUES (?, ?)",
            [("admin", "管理员"), ("member", "成员")],
        )

    def _ensure_users_avatar_column(self, conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "avatar_rel_path" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN avatar_rel_path TEXT")

    def _ensure_papers_date_columns(self, conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        if "date_precision" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN date_precision TEXT")
        if "date_source" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN date_source TEXT")

    def _ensure_papers_resource_columns(self, conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        definitions = {
            "resource_kind": "TEXT NOT NULL DEFAULT 'paper'",
            "canonical_url": "TEXT",
            "source_domain": "TEXT",
            "fetch_status": "TEXT NOT NULL DEFAULT 'ok'",
            "fetched_at": "TEXT",
            "author": "TEXT",
            "site_name": "TEXT",
        }
        for name, definition in definitions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE papers ADD COLUMN {name} {definition}")

    def _ensure_paper_sources_columns(self, conn: sqlite3.Connection) -> None:
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'paper_sources'"
        ).fetchone()
        schema_sql = str(schema_row["sql"] or "") if schema_row is not None else ""
        normalized_schema = re.sub(r"\s+", " ", schema_sql).casefold()
        has_legacy_global_unique = "unique (source_type, source_value)" in normalized_schema
        if has_legacy_global_unique:
            old_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(paper_sources)").fetchall()}
            conn.execute(
                """
                CREATE TABLE paper_sources_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    source_url TEXT,
                    source_context TEXT,
                    sender_open_id TEXT,
                    message_time_iso TEXT,
                    imported_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (imported_by_user_id) REFERENCES users(id) ON DELETE SET NULL
                )
                """
            )
            optional = {
                "source_context": "NULL",
                "sender_open_id": "NULL",
                "message_time_iso": "NULL",
            }
            select_fields = [
                "id",
                "paper_id",
                "source_type",
                "source_value",
                "source_url",
                *(name if name in old_columns else fallback for name, fallback in optional.items()),
                "imported_by_user_id",
                "created_at",
            ]
            conn.execute(
                """
                INSERT INTO paper_sources_new(
                    id, paper_id, source_type, source_value, source_url, source_context,
                    sender_open_id, message_time_iso, imported_by_user_id, created_at
                )
                SELECT %s FROM paper_sources
                """ % ", ".join(select_fields)
            )
            conn.execute("DROP TABLE paper_sources")
            conn.execute("ALTER TABLE paper_sources_new RENAME TO paper_sources")
        else:
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(paper_sources)").fetchall()}
            for name in ("source_context", "sender_open_id", "message_time_iso"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE paper_sources ADD COLUMN {name} TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_sources_paper_id ON paper_sources(paper_id)")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_sources_canonical
            ON paper_sources(source_type, source_value)
            WHERE source_type IN ('arxiv', 'openreview', 'pdf_url', 'canonical_url')
            """
        )

    def _migrate_likes_into_recommendations(self) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO paper_recommendations(paper_id, user_id, reason, created_at, updated_at)
                    SELECT pl.paper_id, pl.user_id, '', pl.created_at, pl.created_at
                    FROM paper_likes pl
                    LEFT JOIN paper_recommendations pr
                      ON pr.paper_id = pl.paper_id AND pr.user_id = pl.user_id
                    WHERE pr.id IS NULL
                    """
                )

    def _timestamp(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def bootstrap_default_user(self, username: str, password: str) -> TeamUser:
        with self._write_lock:
            with self._connect() as conn:
                count = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
                existing = conn.execute(
                    """
                    SELECT u.id, u.username, u.display_name, u.avatar_rel_path, u.is_active, u.created_at, r.slug AS role
                    FROM users u
                    LEFT JOIN user_roles ur ON ur.user_id = u.id
                    LEFT JOIN roles r ON r.id = ur.role_id
                    WHERE u.username = ?
                    """,
                    (username,),
                ).fetchone()
                if existing is None:
                    now = self._timestamp()
                    cursor = conn.execute(
                        """
                        INSERT INTO users(username, display_name, password_hash, is_active, created_at, updated_at)
                        VALUES (?, ?, ?, 1, ?, ?)
                        """,
                        (username, username, generate_password_hash(password), now, now),
                    )
                    role_id = int(conn.execute("SELECT id FROM roles WHERE slug = 'admin'").fetchone()[0])
                    created_user_id = cursor.lastrowid
                    if created_user_id is None:
                        raise RuntimeError("Failed to persist default admin user.")
                    conn.execute(
                        "INSERT OR REPLACE INTO user_roles(user_id, role_id) VALUES (?, ?)",
                        (int(created_user_id), role_id),
                    )
                user = conn.execute(
                    """
                    SELECT u.id, u.username, u.display_name, u.avatar_rel_path, u.is_active, u.created_at, COALESCE(r.slug, 'member') AS role
                    FROM users u
                    LEFT JOIN user_roles ur ON ur.user_id = u.id
                    LEFT JOIN roles r ON r.id = ur.role_id
                    WHERE u.username = ?
                    """,
                    (username,),
                ).fetchone()
        if user is None:
            raise RuntimeError("Failed to bootstrap default user.")
        return TeamUser(
            id=int(user["id"]),
            username=str(user["username"]),
            display_name=str(user["display_name"]),
            role=str(user["role"]),
            is_active=bool(user["is_active"]),
            created_at=str(user["created_at"]),
            avatar_rel_path=(str(user["avatar_rel_path"]) if user["avatar_rel_path"] else None),
        )

    def _ensure_configured_admins(self) -> None:
        """Apply only the deployment's explicit admin allow-list to existing users."""
        if not self.admin_usernames:
            return
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                admin_role_id = self._role_id_for_slug_locked(conn, "admin")
                rows = conn.execute(
                    """
                    SELECT u.id, u.username
                    FROM users u
                    WHERE lower(u.username) IN ({placeholders})
                    """.format(placeholders=", ".join("?" for _ in self.admin_usernames)),
                    tuple(sorted(self.admin_usernames)),
                ).fetchall()
                for row in rows:
                    conn.execute(
                        "INSERT OR REPLACE INTO user_roles(user_id, role_id) VALUES (?, ?)",
                        (int(row["id"]), admin_role_id),
                    )
                    conn.execute(
                        "UPDATE users SET updated_at = ? WHERE id = ?",
                        (now, int(row["id"])),
                    )

    def authenticate_user(self, username: str, password: str) -> TeamUser | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.display_name, u.password_hash, u.avatar_rel_path, u.is_active, u.created_at, COALESCE(r.slug, 'member') AS role
                FROM users u
                LEFT JOIN user_roles ur ON ur.user_id = u.id
                LEFT JOIN roles r ON r.id = ur.role_id
                WHERE u.username = ?
                """,
                (username,),
            ).fetchone()
        if row is None or not row["is_active"]:
            return None
        if not check_password_hash(str(row["password_hash"]), password):
            return None
        return TeamUser(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
        )

    def get_user(self, user_id: int | None) -> TeamUser | None:
        if user_id is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.display_name, u.avatar_rel_path, u.is_active, u.created_at, COALESCE(r.slug, 'member') AS role
                FROM users u
                LEFT JOIN user_roles ur ON ur.user_id = u.id
                LEFT JOIN roles r ON r.id = ur.role_id
                WHERE u.id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return TeamUser(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
        )

    def get_user_by_username(self, username: str) -> TeamUser | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.display_name, u.avatar_rel_path, u.is_active, u.created_at, COALESCE(r.slug, 'member') AS role
                FROM users u
                LEFT JOIN user_roles ur ON ur.user_id = u.id
                LEFT JOIN roles r ON r.id = ur.role_id
                WHERE u.username = ?
                """,
                (username,),
            ).fetchone()
        if row is None:
            return None
        return TeamUser(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
        )

    def list_users(self) -> list[TeamUser]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.id, u.username, u.display_name, u.avatar_rel_path, u.is_active, u.created_at, COALESCE(r.slug, 'member') AS role
                FROM users u
                LEFT JOIN user_roles ur ON ur.user_id = u.id
                LEFT JOIN roles r ON r.id = ur.role_id
                ORDER BY CASE WHEN r.slug = 'admin' THEN 0 ELSE 1 END, LOWER(u.display_name), LOWER(u.username)
                """
            ).fetchall()
        return [
            TeamUser(
                id=int(row["id"]),
                username=str(row["username"]),
                display_name=str(row["display_name"]),
                role=str(row["role"]),
                is_active=bool(row["is_active"]),
                created_at=str(row["created_at"]),
                avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
            )
            for row in rows
        ]

    def create_user(self, username: str, display_name: str, password: str, role_slug: str = "member") -> TeamUser:
        username = username.strip()
        display_name = display_name.strip() or username
        if not username:
            raise ValueError("用户名不能为空。")
        if not password:
            raise ValueError("密码不能为空。")
        if role_slug not in {"admin", "member"}:
            raise ValueError("不支持的角色。")
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone() is not None:
                    raise ValueError("用户名已存在。")
                user_count = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
                assigned_role = "admin" if user_count == 0 or username.casefold() in self.admin_usernames else role_slug
                cursor = conn.execute(
                    """
                    INSERT INTO users(username, display_name, password_hash, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (username, display_name, generate_password_hash(password), now, now),
                )
                role_id = int(conn.execute("SELECT id FROM roles WHERE slug = ?", (assigned_role,)).fetchone()[0])
                created_user_id = cursor.lastrowid
                if created_user_id is None:
                    raise RuntimeError("Failed to persist user.")
                conn.execute(
                    "INSERT OR REPLACE INTO user_roles(user_id, role_id) VALUES (?, ?)",
                    (int(created_user_id), role_id),
                )
        user = self.get_user_by_username(username)
        if user is None:
            raise RuntimeError("Failed to create user.")
        return user

    def register_user(self, username: str, display_name: str, password: str) -> TeamUser:
        return self.create_user(username, display_name, password, role_slug="member")

    def _role_id_for_slug_locked(self, conn: sqlite3.Connection, role_slug: str) -> int:
        row = conn.execute("SELECT id FROM roles WHERE slug = ?", (role_slug,)).fetchone()
        if row is None:
            raise ValueError("不支持的角色。")
        return int(row[0])

    def _active_admin_count_locked(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM users u
            JOIN user_roles ur ON ur.user_id = u.id
            JOIN roles r ON r.id = ur.role_id
            WHERE u.is_active = 1 AND r.slug = 'admin'
            """
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def update_user(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role_slug: str | None = None,
        is_active: bool | None = None,
        password: str | None = None,
        acting_user_id: int | None = None,
    ) -> TeamUser:
        if role_slug is not None and role_slug not in {"admin", "member"}:
            raise ValueError("不支持的角色。")
        cleaned_password = (password or "").strip()
        with self._write_lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT u.id, u.username, u.display_name, u.is_active, COALESCE(r.slug, 'member') AS role
                    FROM users u
                    LEFT JOIN user_roles ur ON ur.user_id = u.id
                    LEFT JOIN roles r ON r.id = ur.role_id
                    WHERE u.id = ?
                    """,
                    (user_id,),
                ).fetchone()
                if row is None:
                    raise FileNotFoundError(user_id)

                username = str(row["username"])
                next_display_name = (display_name or str(row["display_name"])).strip() or username
                next_role = role_slug or str(row["role"] or "member")
                next_is_active = bool(row["is_active"]) if is_active is None else bool(is_active)
                current_role = str(row["role"] or "member")
                current_is_active = bool(row["is_active"])

                if acting_user_id == user_id and not next_is_active:
                    raise ValueError("不能停用当前登录账号。")

                if current_role == "admin" and current_is_active and (next_role != "admin" or not next_is_active):
                    if self._active_admin_count_locked(conn) <= 1:
                        raise ValueError("至少需要保留一个启用中的管理员账号。")

                now = self._timestamp()
                conn.execute(
                    "UPDATE users SET display_name = ?, is_active = ?, updated_at = ? WHERE id = ?",
                    (next_display_name, 1 if next_is_active else 0, now, user_id),
                )
                if cleaned_password:
                    conn.execute(
                        "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                        (generate_password_hash(cleaned_password), now, user_id),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO user_roles(user_id, role_id) VALUES (?, ?)",
                    (user_id, self._role_id_for_slug_locked(conn, next_role)),
                )

        user = self.get_user(user_id)
        if user is None:
            raise RuntimeError("Failed to update user.")
        return user

    def update_user_avatar(self, user_id: int, avatar_rel_path: str | None) -> TeamUser:
        with self._write_lock:
            with self._connect() as conn:
                row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
                if row is None:
                    raise FileNotFoundError(user_id)
                conn.execute(
                    "UPDATE users SET avatar_rel_path = ?, updated_at = ? WHERE id = ?",
                    (avatar_rel_path, self._timestamp(), user_id),
                )
        user = self.get_user(user_id)
        if user is None:
            raise RuntimeError("Failed to update user avatar.")
        return user

    def is_admin(self, user_id: int | None) -> bool:
        user = self.get_user(user_id)
        return bool(user and user.role == "admin")

    def sync_papers(self, papers: Iterable[Any]) -> None:
        rows = list(papers)
        if not rows:
            return
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                for paper in rows:
                    conn.execute(
                        """
                        INSERT INTO papers(
                            rel_path, file_name, folder, extension, title, display_title, preview_text,
                            extracted_date, sort_date, date_precision, date_source, file_size, modified_at, preview_kind,
                            resource_kind, canonical_url, source_domain, fetch_status, fetched_at, author, site_name, is_done,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(rel_path) DO UPDATE SET
                            file_name = excluded.file_name,
                            folder = excluded.folder,
                            extension = excluded.extension,
                            title = excluded.title,
                            display_title = excluded.display_title,
                            preview_text = excluded.preview_text,
                            extracted_date = excluded.extracted_date,
                            sort_date = excluded.sort_date,
                            date_precision = excluded.date_precision,
                            date_source = excluded.date_source,
                            file_size = excluded.file_size,
                            modified_at = excluded.modified_at,
                            preview_kind = excluded.preview_kind,
                            resource_kind = excluded.resource_kind,
                            canonical_url = excluded.canonical_url,
                            source_domain = excluded.source_domain,
                            fetch_status = excluded.fetch_status,
                            fetched_at = excluded.fetched_at,
                            author = excluded.author,
                            site_name = excluded.site_name,
                            is_done = excluded.is_done,
                            updated_at = excluded.updated_at
                        """,
                        (
                            paper.rel_path,
                            paper.file_name,
                            paper.folder,
                            paper.extension,
                            paper.title,
                            paper.display_title,
                            str(paper.preview_text or "")[:20000],
                            paper.extracted_date,
                            paper.sort_date,
                            getattr(paper, "date_precision", None),
                            getattr(paper, "date_source", None),
                            int(paper.file_size),
                            paper.modified_at,
                            paper.preview_kind,
                            getattr(paper, "resource_kind", "paper"),
                            getattr(paper, "source_url", None),
                            getattr(paper, "source_domain", None),
                            getattr(paper, "fetch_status", "ok"),
                            getattr(paper, "fetched_at", None),
                            getattr(paper, "author", None),
                            getattr(paper, "site_name", None),
                            1 if paper.is_done else 0,
                            now,
                            now,
                        ),
                    )
                    paper_id = self._paper_id_for_rel_path_locked(conn, paper.rel_path)
                    self._persist_manual_date_from_record_locked(conn, paper_id, paper)
                    self._apply_date_override_to_paper_locked(conn, paper_id)
                    self._ensure_auto_tags_locked(conn, paper_id, paper)

    def all_manual_date_overrides(self) -> dict[str, dict[str, str | None]]:
        overrides: dict[str, dict[str, str | None]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.rel_path, o.display_date, o.precision, o.sort_date, o.updated_at
                FROM paper_date_overrides o
                JOIN papers p ON p.id = o.paper_id
                """
            ).fetchall()
            for row in rows:
                overrides[str(row["rel_path"])] = {
                    "display_date": str(row["display_date"]),
                    "precision": str(row["precision"] or "month"),
                    "source": "manual",
                    "sort_date": str(row["sort_date"]),
                    "updated_at": str(row["updated_at"]),
                }
        return overrides

    def manual_date_overrides_for_rel_paths(self, rel_paths: Iterable[str]) -> dict[str, dict[str, str | None]]:
        unique_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
        if not unique_rel_paths:
            return {}
        overrides: dict[str, dict[str, str | None]] = {}
        with self._connect() as conn:
            for start in range(0, len(unique_rel_paths), 900):
                chunk = unique_rel_paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT p.rel_path, o.display_date, o.precision, o.sort_date, o.updated_at
                    FROM paper_date_overrides o
                    JOIN papers p ON p.id = o.paper_id
                    WHERE p.rel_path IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    overrides[str(row["rel_path"])] = {
                        "display_date": str(row["display_date"]),
                        "precision": str(row["precision"] or "month"),
                        "source": "manual",
                        "sort_date": str(row["sort_date"]),
                        "updated_at": str(row["updated_at"]),
                    }
        return overrides

    def set_manual_date_override(
        self,
        rel_path: str,
        *,
        display_date: str,
        precision: str,
        sort_date: str,
        updated_by_user_id: int | None,
    ) -> None:
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                conn.execute(
                    """
                    INSERT INTO paper_date_overrides(
                        paper_id, display_date, precision, sort_date, updated_by_user_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        display_date = excluded.display_date,
                        precision = excluded.precision,
                        sort_date = excluded.sort_date,
                        updated_by_user_id = excluded.updated_by_user_id,
                        updated_at = excluded.updated_at
                    """,
                    (paper_id, display_date, precision, sort_date, updated_by_user_id, now),
                )
                conn.execute(
                    """
                    UPDATE papers
                    SET extracted_date = ?, sort_date = ?, date_precision = ?, date_source = 'manual', updated_at = ?
                    WHERE id = ?
                    """,
                    (display_date, sort_date, precision, now, paper_id),
                )

    def migrate_manual_date_overrides(self, payload: dict[str, Any], *, updated_by_user_id: int | None = None) -> int:
        if not payload:
            return 0
        now = self._timestamp()
        migrated = 0
        with self._write_lock:
            with self._connect() as conn:
                for rel_path, raw in payload.items():
                    if not isinstance(rel_path, str) or not isinstance(raw, dict):
                        continue
                    display_date = str(raw.get("display_date") or "").strip()
                    precision = str(raw.get("precision") or "month").strip() or "month"
                    sort_date = str(raw.get("sort_date") or "").strip()
                    if not display_date or precision not in {"year", "month", "day"} or not sort_date:
                        continue
                    paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                    if paper_id is None:
                        continue
                    updated_at = str(raw.get("updated_at") or "").strip() or now
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO paper_date_overrides(
                            paper_id, display_date, precision, sort_date, updated_by_user_id, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (paper_id, display_date, precision, sort_date, updated_by_user_id, updated_at),
                    )
                    if cursor.rowcount <= 0:
                        continue
                    conn.execute(
                        """
                        UPDATE papers
                        SET extracted_date = ?, sort_date = ?, date_precision = ?, date_source = 'manual', updated_at = ?
                        WHERE id = ?
                        """,
                        (display_date, sort_date, precision, now, paper_id),
                    )
                    migrated += 1
        return migrated

    def rename_paper(self, old_rel_path: str, new_paper: Any) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE papers
                    SET rel_path = ?, file_name = ?, folder = ?, extension = ?, title = ?, display_title = ?,
                        preview_text = ?, extracted_date = ?, sort_date = ?, date_precision = ?, date_source = ?, file_size = ?, modified_at = ?,
                        preview_kind = ?, resource_kind = ?, canonical_url = ?, source_domain = ?, fetch_status = ?,
                        fetched_at = ?, author = ?, site_name = ?, is_done = ?, updated_at = ?
                    WHERE rel_path = ?
                    """,
                    (
                        new_paper.rel_path,
                        new_paper.file_name,
                        new_paper.folder,
                        new_paper.extension,
                        new_paper.title,
                        new_paper.display_title,
                        str(new_paper.preview_text or "")[:20000],
                        new_paper.extracted_date,
                        new_paper.sort_date,
                        getattr(new_paper, "date_precision", None),
                        getattr(new_paper, "date_source", None),
                        int(new_paper.file_size),
                        new_paper.modified_at,
                        new_paper.preview_kind,
                        getattr(new_paper, "resource_kind", "paper"),
                        getattr(new_paper, "source_url", None),
                        getattr(new_paper, "source_domain", None),
                        getattr(new_paper, "fetch_status", "ok"),
                        getattr(new_paper, "fetched_at", None),
                        getattr(new_paper, "author", None),
                        getattr(new_paper, "site_name", None),
                        1 if new_paper.is_done else 0,
                        self._timestamp(),
                        old_rel_path,
                    ),
                )
                paper_id = self._paper_id_for_rel_path_locked(conn, new_paper.rel_path)
                self._persist_manual_date_from_record_locked(conn, paper_id, new_paper)
                self._apply_date_override_to_paper_locked(conn, paper_id)
                self._ensure_auto_tags_locked(conn, paper_id, new_paper)

    def delete_paper(self, rel_path: str) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM papers WHERE rel_path = ?", (rel_path,))

    def add_source(
        self,
        rel_path: str,
        *,
        source_type: str,
        source_value: str,
        source_url: str | None = None,
        source_context: str | None = None,
        sender_open_id: str | None = None,
        message_time_iso: str | None = None,
        imported_by_user_id: int | None = None,
    ) -> None:
        normalized_source_value = self._normalize_source_value(source_type, source_value)
        if not normalized_source_value:
            return
        canonical_types = {"arxiv", "openreview", "pdf_url", "canonical_url"}
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    return
                values = (
                    paper_id,
                    source_type,
                    normalized_source_value,
                    source_url,
                    source_context,
                    sender_open_id,
                    message_time_iso,
                    imported_by_user_id,
                    self._timestamp(),
                )
                if source_type in canonical_types:
                    conn.execute(
                        """
                        INSERT INTO paper_sources(
                            paper_id, source_type, source_value, source_url, source_context,
                            sender_open_id, message_time_iso, imported_by_user_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_type, source_value)
                        WHERE source_type IN ('arxiv', 'openreview', 'pdf_url', 'canonical_url')
                        DO UPDATE SET
                            paper_id = excluded.paper_id,
                            source_url = COALESCE(excluded.source_url, paper_sources.source_url),
                            imported_by_user_id = COALESCE(excluded.imported_by_user_id, paper_sources.imported_by_user_id)
                        """,
                        values,
                    )
                    return

                existing = conn.execute(
                    """
                    SELECT id FROM paper_sources
                    WHERE paper_id = ? AND source_type = ? AND source_value = ?
                      AND COALESCE(source_url, '') = COALESCE(?, '')
                      AND COALESCE(message_time_iso, '') = COALESCE(?, '')
                    ORDER BY id DESC LIMIT 1
                    """,
                    (paper_id, source_type, normalized_source_value, source_url, message_time_iso),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO paper_sources(
                            paper_id, source_type, source_value, source_url, source_context,
                            sender_open_id, message_time_iso, imported_by_user_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
                else:
                    conn.execute(
                        """
                        UPDATE paper_sources
                        SET source_context = COALESCE(?, source_context),
                            sender_open_id = COALESCE(?, sender_open_id),
                            imported_by_user_id = COALESCE(?, imported_by_user_id)
                        WHERE id = ?
                        """,
                        (source_context, sender_open_id, imported_by_user_id, int(existing["id"])),
                    )

    def find_paper_by_source(self, source_type: str, source_value: str) -> str | None:
        normalized_source_value = self._normalize_source_value(source_type, source_value)
        if not normalized_source_value:
            return None
        with self._connect() as conn:
            if source_type == "arxiv":
                rows = conn.execute(
                    """
                    SELECT p.rel_path, ps.source_value, p.updated_at, p.id
                    FROM paper_sources ps
                    JOIN papers p ON p.id = ps.paper_id
                    WHERE ps.source_type = ?
                    ORDER BY p.updated_at DESC, p.id DESC
                    """,
                    (source_type,),
                ).fetchall()
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if self._normalize_source_value(source_type, str(candidate["source_value"] or "")) == normalized_source_value
                    ),
                    None,
                )
            else:
                row = conn.execute(
                    """
                    SELECT p.rel_path
                    FROM paper_sources ps
                    JOIN papers p ON p.id = ps.paper_id
                    WHERE ps.source_type = ? AND ps.source_value = ?
                    """,
                    (source_type, normalized_source_value),
                ).fetchone()
        return str(row["rel_path"]) if row is not None else None

    def _normalize_source_value(self, source_type: str, source_value: str) -> str:
        value = source_value.strip()
        if not value:
            return ""
        if source_type != "arxiv":
            return value
        normalized = normalize_arxiv_id(value)
        if normalized is None:
            return value
        return base_arxiv_id(normalized)

    def sources_for_rel_path(self, rel_path: str) -> list[dict[str, str | None]]:
        with self._connect() as conn:
            paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
            if paper_id is None:
                return []
            rows = conn.execute(
                """
                SELECT source_type, source_value, source_url, source_context, sender_open_id, message_time_iso
                FROM paper_sources WHERE paper_id = ? ORDER BY created_at DESC, id DESC
                """,
                (paper_id,),
            ).fetchall()
        return [
            {
                "source_type": str(row["source_type"]),
                "source_value": str(row["source_value"]),
                "source_url": (str(row["source_url"]) if row["source_url"] else None),
                "source_context": (str(row["source_context"]) if row["source_context"] else None),
                "sender_open_id": (str(row["sender_open_id"]) if row["sender_open_id"] else None),
                "message_time_iso": (str(row["message_time_iso"]) if row["message_time_iso"] else None),
            }
            for row in rows
        ]

    def library_interest_rows(self, *, limit: int = 240) -> list[dict[str, Any]]:
        """Return the compact, human-curated signals used to describe lab interests."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.rel_path, p.display_title, p.resource_kind,
                       GROUP_CONCAT(DISTINCT t.name) AS tags
                FROM papers p
                LEFT JOIN paper_tags pt ON pt.paper_id = p.id
                LEFT JOIN tags t ON t.id = pt.tag_id
                GROUP BY p.id
                ORDER BY COALESCE(p.sort_date, p.updated_at) DESC, p.id DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "rel_path": str(row["rel_path"]),
                "title": str(row["display_title"]),
                "kind": str(row["resource_kind"]),
                "tags": [tag.strip() for tag in str(row["tags"] or "").split(",") if tag.strip()],
            }
            for row in rows
        ]

    def discovery_exists(self, *, source: str, external_id: str, url: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM discoveries
                WHERE (source = ? AND external_id = ?) OR url = ?
                LIMIT 1
                """,
                (source, external_id, url),
            ).fetchone()
        return row is not None

    def discovery_is_in_library(self, *, url: str, arxiv_id: str | None = None) -> bool:
        """Check canonical web sources and the arXiv base id before scoring a candidate."""
        with self._connect() as conn:
            existing = conn.execute("SELECT 1 FROM papers WHERE canonical_url = ? LIMIT 1", (url,)).fetchone()
            if existing is not None:
                return True
            existing = conn.execute(
                """
                SELECT 1 FROM paper_sources
                WHERE source_type IN ('canonical_url', 'pdf_url') AND source_value = ?
                LIMIT 1
                """,
                (url,),
            ).fetchone()
            if existing is not None:
                return True
            if arxiv_id:
                normalized = self._normalize_source_value("arxiv", arxiv_id)
                rows = conn.execute("SELECT source_value FROM paper_sources WHERE source_type = 'arxiv'").fetchall()
                return any(self._normalize_source_value("arxiv", str(row["source_value"])) == normalized for row in rows)
        return False

    def add_discoveries(self, rows: Iterable[dict[str, Any]]) -> int:
        """Persist freshly scored candidates; an old surface is never silently re-ranked."""
        entries = list(rows)
        if not entries:
            return 0
        inserted = 0
        with self._write_lock:
            with self._connect() as conn:
                for entry in entries:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO discoveries(
                            source, external_id, title, url, abstract, score, reason, status, surfaced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?)
                        """,
                        (
                            str(entry["source"]),
                            str(entry["external_id"]),
                            str(entry["title"]),
                            str(entry["url"]),
                            str(entry.get("abstract") or ""),
                            float(entry.get("score") or 0),
                            str(entry.get("reason") or ""),
                            str(entry.get("surfaced_at") or self._timestamp()),
                        ),
                    )
                    inserted += max(0, cursor.rowcount)
        return inserted

    def list_discoveries(self, *, limit: int = 100) -> list[DiscoveryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source, external_id, title, url, abstract, score, reason, status,
                       surfaced_at, decided_by, decided_at
                FROM discoveries
                ORDER BY date(surfaced_at) DESC,
                         CASE status WHEN 'new' THEN 0 WHEN 'imported' THEN 1 ELSE 2 END,
                         score DESC, id DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            DiscoveryRecord(
                id=int(row["id"]), source=str(row["source"]), external_id=str(row["external_id"]),
                title=str(row["title"]), url=str(row["url"]), abstract=str(row["abstract"] or ""),
                score=float(row["score"]), reason=str(row["reason"] or ""), status=str(row["status"]),
                surfaced_at=str(row["surfaced_at"]), decided_by=(int(row["decided_by"]) if row["decided_by"] is not None else None),
                decided_at=(str(row["decided_at"]) if row["decided_at"] else None),
            )
            for row in rows
        ]

    def get_discovery(self, discovery_id: int) -> DiscoveryRecord | None:
        return next((item for item in self.list_discoveries(limit=1000) if item.id == discovery_id), None)

    def decide_discovery(self, discovery_id: int, *, status: str, decided_by: int) -> DiscoveryRecord:
        if status not in {"imported", "dismissed"}:
            raise ValueError("Discovery status must be imported or dismissed.")
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE discoveries
                    SET status = ?, decided_by = ?, decided_at = ?
                    WHERE id = ? AND status = 'new'
                    """,
                    (status, decided_by, now, discovery_id),
                )
                if cursor.rowcount != 1:
                    row = conn.execute("SELECT status FROM discoveries WHERE id = ?", (discovery_id,)).fetchone()
                    if row is None:
                        raise KeyError(discovery_id)
                    raise ValueError(f"Discovery has already been {row['status']}.")
        decided = self.get_discovery(discovery_id)
        if decided is None:
            raise KeyError(discovery_id)
        return decided

    def replace_attachments(self, rel_path: str, attachments: Iterable[Any]) -> list[PaperAttachment]:
        bounded = list(attachments)[:3]
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                source_urls: list[str] = []
                for attachment in bounded:
                    source_url = str(getattr(attachment, "source_url", "") or "").strip()
                    kind = str(getattr(attachment, "kind", "") or "").strip()
                    title = str(getattr(attachment, "title", "") or "").strip()
                    local_path = str(getattr(attachment, "local_path", "") or "").strip() or None
                    if not source_url or not kind or not title:
                        continue
                    if local_path:
                        target = (self.root / local_path).resolve()
                        if self.root not in target.parents or not target.is_file():
                            local_path = None
                    conn.execute(
                        """
                        INSERT INTO paper_attachments(paper_id, kind, title, local_path, source_url, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(paper_id, source_url) DO UPDATE SET
                            kind = excluded.kind,
                            title = excluded.title,
                            local_path = excluded.local_path
                        """,
                        (paper_id, kind, title[:300], local_path, source_url, now),
                    )
                    source_urls.append(source_url)
                if source_urls:
                    placeholders = ",".join("?" for _ in source_urls)
                    conn.execute(
                        f"DELETE FROM paper_attachments WHERE paper_id = ? AND source_url NOT IN ({placeholders})",
                        (paper_id, *source_urls),
                    )
                else:
                    conn.execute("DELETE FROM paper_attachments WHERE paper_id = ?", (paper_id,))
        return self.attachments_for_rel_path(rel_path)

    def attachments_for_rel_path(self, rel_path: str) -> list[PaperAttachment]:
        with self._connect() as conn:
            paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
            if paper_id is None:
                return []
            rows = conn.execute(
                """
                SELECT id, kind, title, local_path, source_url, created_at
                FROM paper_attachments
                WHERE paper_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT 3
                """,
                (paper_id,),
            ).fetchall()
        return [
            PaperAttachment(
                id=int(row["id"]),
                kind=str(row["kind"]),
                title=str(row["title"]),
                local_path=(str(row["local_path"]) if row["local_path"] else None),
                source_url=str(row["source_url"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def attachment_paths_for_rel_path(self, rel_path: str) -> list[Path]:
        paths: list[Path] = []
        for attachment in self.attachments_for_rel_path(rel_path):
            if not attachment.local_path:
                continue
            path = (self.root / attachment.local_path).resolve()
            if self.root in path.parents and path.is_file():
                paths.append(path)
        return paths

    def add_recommendation(self, rel_path: str, user_id: int, reason: str) -> None:
        reason = reason.strip()
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                now = self._timestamp()
                conn.execute(
                    """
                    INSERT INTO paper_recommendations(paper_id, user_id, reason, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id, user_id) DO UPDATE SET
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                    (paper_id, user_id, reason, now, now),
                )

    def toggle_recommendation(self, rel_path: str, user_id: int) -> bool:
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                row = conn.execute(
                    "SELECT id FROM paper_recommendations WHERE paper_id = ? AND user_id = ?",
                    (paper_id, user_id),
                ).fetchone()
                if row is None:
                    now = self._timestamp()
                    conn.execute(
                        "INSERT INTO paper_recommendations(paper_id, user_id, reason, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
                        (paper_id, user_id, now, now),
                    )
                    return True
                conn.execute("DELETE FROM paper_recommendations WHERE id = ?", (int(row["id"]),))
                return False

    def set_done_state(self, rel_path: str, user_id: int, *, is_done: bool) -> bool:
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                now = self._timestamp()
                conn.execute(
                    """
                    INSERT INTO paper_user_states(paper_id, user_id, is_done, done_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id, user_id) DO UPDATE SET
                        is_done = excluded.is_done,
                        done_at = excluded.done_at,
                        updated_at = excluded.updated_at
                    """,
                    (paper_id, user_id, 1 if is_done else 0, now if is_done else None, now),
                )
        return is_done

    def toggle_done_state(self, rel_path: str, user_id: int) -> bool:
        current = self.is_done_for_user(rel_path, user_id)
        return self.set_done_state(rel_path, user_id, is_done=(not current))

    def is_done_for_user(self, rel_path: str, user_id: int) -> bool:
        with self._connect() as conn:
            paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
            if paper_id is None:
                return False
            row = conn.execute(
                "SELECT is_done FROM paper_user_states WHERE paper_id = ? AND user_id = ?",
                (paper_id, user_id),
            ).fetchone()
        return bool(row["is_done"]) if row is not None else False

    def done_rel_paths_for_user(self, user_id: int | None) -> set[str]:
        if user_id is None:
            return set()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.rel_path
                FROM paper_user_states pus
                JOIN papers p ON p.id = pus.paper_id
                WHERE pus.user_id = ? AND pus.is_done = 1
                """,
                (user_id,),
            ).fetchall()
        return {str(row["rel_path"]) for row in rows}

    def mark_done_for_all_users(self, rel_path: str) -> None:
        users = self.list_users()
        for user in users:
            if not user.is_active:
                continue
            self.set_done_state(rel_path, user.id, is_done=True)


    def add_comment(self, rel_path: str, user_id: int, body: str, parent_id: int | None = None) -> None:
        body = body.strip()
        if not body:
            raise ValueError("评论不能为空。")
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                now = self._timestamp()
                conn.execute(
                    """
                    INSERT INTO comments(paper_id, user_id, parent_id, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (paper_id, user_id, parent_id, body, now, now),
                )

    def add_tag(self, rel_path: str, name: str, user_id: int | None, *, source_type: str = "manual", is_locked: bool = False) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("标签不能为空。")
        slug = slugify_text(normalized_name)
        if not slug:
            raise ValueError("标签格式无效。")
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                now = self._timestamp()
                row = conn.execute("SELECT id FROM tags WHERE slug = ?", (slug,)).fetchone()
                if row is None:
                    cursor = conn.execute(
                        """
                        INSERT INTO tags(name, slug, created_by_user_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (normalized_name, slug, user_id, now, now),
                    )
                    created_tag_id = cursor.lastrowid
                    if created_tag_id is None:
                        raise RuntimeError("Failed to persist tag.")
                    tag_id = int(created_tag_id)
                else:
                    tag_id = int(row["id"])
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_tags(paper_id, tag_id, source_type, score, added_by_user_id, created_at, is_locked)
                    VALUES (?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (paper_id, tag_id, source_type, user_id, now, 1 if is_locked else 0),
                )

    def replace_generated_tags(
        self,
        rel_path: str,
        names: list[str],
        *,
        source_type: str = "ai",
        user_id: int | None = None,
        is_locked: bool = True,
    ) -> None:
        cleaned_names = [name.strip() for name in names if name and name.strip()]
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                now = self._timestamp()
                desired_tag_ids: list[int] = []
                for name in cleaned_names:
                    slug = slugify_text(name)
                    if not slug:
                        continue
                    row = conn.execute("SELECT id FROM tags WHERE slug = ?", (slug,)).fetchone()
                    if row is None:
                        cursor = conn.execute(
                            """
                            INSERT INTO tags(name, slug, created_by_user_id, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (name, slug, user_id, now, now),
                        )
                        created_tag_id = cursor.lastrowid
                        if created_tag_id is None:
                            raise RuntimeError("Failed to persist generated tag.")
                        tag_id = int(created_tag_id)
                    else:
                        tag_id = int(row["id"])
                    desired_tag_ids.append(tag_id)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO paper_tags(paper_id, tag_id, source_type, score, added_by_user_id, created_at, is_locked)
                        VALUES (?, ?, ?, NULL, ?, ?, ?)
                        """,
                        (paper_id, tag_id, source_type, user_id, now, 1 if is_locked else 0),
                    )

                if desired_tag_ids:
                    placeholders = ", ".join("?" for _ in desired_tag_ids)
                    conn.execute(
                        f"DELETE FROM paper_tags WHERE paper_id = ? AND source_type = ? AND tag_id NOT IN ({placeholders})",
                        (paper_id, source_type, *desired_tag_ids),
                    )
                else:
                    conn.execute(
                        "DELETE FROM paper_tags WHERE paper_id = ? AND source_type = ?",
                        (paper_id, source_type),
                    )

    def remove_tag(self, rel_path: str, tag_id: int, *, is_admin: bool, acting_user_id: int | None) -> None:
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                row = conn.execute(
                    """
                    SELECT added_by_user_id, is_locked, source_type
                    FROM paper_tags
                    WHERE paper_id = ? AND tag_id = ?
                    """,
                    (paper_id, tag_id),
                ).fetchone()
                if row is None:
                    return
                if bool(row["is_locked"]) and not is_admin:
                    raise PermissionError("系统标签只能由管理员移除。")
                if not is_admin and row["added_by_user_id"] not in {None, acting_user_id}:
                    raise PermissionError("只能删除自己添加的标签。")
                conn.execute(
                    "DELETE FROM paper_tags WHERE paper_id = ? AND tag_id = ?",
                    (paper_id, tag_id),
                )

    def search_rel_paths(self, query: str) -> set[str]:
        query_text = query.strip().lower()
        if not query_text:
            return set()
        wildcard = f"%{query_text}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.rel_path
                FROM papers p
                LEFT JOIN paper_tags pt ON pt.paper_id = p.id
                LEFT JOIN tags t ON t.id = pt.tag_id
                LEFT JOIN paper_recommendations pr ON pr.paper_id = p.id
                LEFT JOIN users u ON u.id = pr.user_id
                LEFT JOIN paper_sources ps ON ps.paper_id = p.id
                WHERE lower(COALESCE(p.display_title, '')) LIKE ?
                   OR lower(COALESCE(p.file_name, '')) LIKE ?
                   OR lower(COALESCE(p.preview_text, '')) LIKE ?
                   OR lower(COALESCE(t.name, '')) LIKE ?
                   OR lower(COALESCE(pr.reason, '')) LIKE ?
                   OR lower(COALESCE(u.display_name, '')) LIKE ?
                   OR lower(COALESCE(u.username, '')) LIKE ?
                   OR lower(COALESCE(ps.source_value, '')) LIKE ?
                """,
                (wildcard, wildcard, wildcard, wildcard, wildcard, wildcard, wildcard, wildcard),
            ).fetchall()
        return {str(row["rel_path"]) for row in rows}

    def _chat_thread_key(self, paper_id: int, visibility: str, owner_user_id: int | None) -> str:
        if visibility == "shared":
            return f"shared:{paper_id}"
        if visibility == "private" and owner_user_id is not None:
            return f"private:{paper_id}:{owner_user_id}"
        raise ValueError("Unsupported chat visibility.")

    def _ensure_chat_thread_locked(
        self,
        conn: sqlite3.Connection,
        paper_id: int,
        visibility: str,
        owner_user_id: int | None,
    ) -> int:
        thread_key = self._chat_thread_key(paper_id, visibility, owner_user_id)
        row = conn.execute("SELECT id FROM chat_threads WHERE thread_key = ?", (thread_key,)).fetchone()
        if row is not None:
            return int(row["id"])
        now = self._timestamp()
        cursor = conn.execute(
            """
            INSERT INTO chat_threads(paper_id, visibility, owner_user_id, thread_key, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (paper_id, visibility, owner_user_id, thread_key, now, now),
        )
        created_thread_id = cursor.lastrowid
        if created_thread_id is None:
            raise RuntimeError("Failed to persist chat thread.")
        return int(created_thread_id)

    def _chat_messages_for_thread_locked(
        self,
        conn: sqlite3.Connection,
        thread_id: int,
        *,
        limit: int = 16,
    ) -> list[TeamChatMessage]:
        rows = conn.execute(
            """
            SELECT *
            FROM (
                SELECT cm.id, cm.role, cm.body, cm.created_at, cm.status, cm.model,
                       COALESCE(u.display_name, u.username) AS display_name,
                       u.avatar_rel_path AS avatar_rel_path
                FROM chat_messages cm
                LEFT JOIN users u ON u.id = cm.user_id
                WHERE cm.thread_id = ?
                ORDER BY cm.id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            (thread_id, limit),
        ).fetchall()
        return [
            TeamChatMessage(
                id=int(row["id"]),
                role=str(row["role"]),
                display_name=(
                    str(row["display_name"])
                    if row["display_name"]
                    else ("Paper Bot" if str(row["role"]) == "assistant" else "成员")
                ),
                avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
                body=str(row["body"]),
                created_at=str(row["created_at"]),
                status=str(row["status"]),
                model=(str(row["model"]) if row["model"] else None),
            )
            for row in rows
        ]

    def add_chat_message(
        self,
        rel_path: str,
        *,
        visibility: str,
        body: str,
        user_id: int | None,
        role: str,
        status: str = "completed",
        model: str | None = None,
    ) -> int:
        message_body = body.strip()
        if not message_body:
            raise ValueError("消息不能为空。")
        if visibility not in {"shared", "private"}:
            raise ValueError("不支持的聊天模式。")
        if role not in {"user", "assistant", "system"}:
            raise ValueError("不支持的消息角色。")
        if visibility == "private" and user_id is None:
            raise ValueError("私聊消息必须绑定当前用户。")

        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    raise FileNotFoundError(rel_path)
                owner_user_id = user_id if visibility == "private" else None
                thread_id = self._ensure_chat_thread_locked(conn, paper_id, visibility, owner_user_id)
                now = self._timestamp()
                cursor = conn.execute(
                    """
                    INSERT INTO chat_messages(thread_id, user_id, role, body, status, model, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, user_id if role == "user" else None, role, message_body, status, model, now, now),
                )
                conn.execute("UPDATE chat_threads SET updated_at = ? WHERE id = ?", (now, thread_id))
                message_id = cursor.lastrowid
                if message_id is None:
                    raise RuntimeError("Failed to persist chat message.")
                return int(message_id)

    def update_chat_message(
        self,
        message_id: int,
        *,
        body: str | None = None,
        status: str | None = None,
        model: str | None = None,
    ) -> None:
        changes: list[str] = []
        values: list[Any] = []
        if body is not None:
            changes.append("body = ?")
            values.append(body.strip() or "消息内容为空。")
        if status is not None:
            changes.append("status = ?")
            values.append(status)
        if model is not None:
            changes.append("model = ?")
            values.append(model)
        if not changes:
            return
        changes.append("updated_at = ?")
        values.append(self._timestamp())
        values.append(message_id)
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE chat_messages SET {', '.join(changes)} WHERE id = ?",
                    values,
                )

    def mark_pending_chat_messages_failed(self) -> None:
        now = self._timestamp()
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE chat_messages
                    SET status = 'failed',
                        body = CASE
                            WHEN body LIKE 'Paper Bot 正在思考%' THEN '服务重启前的聊天任务没有完成，请重新发送一次。'
                            ELSE body
                        END,
                        updated_at = ?
                    WHERE status = 'pending'
                    """,
                    (now,),
                )

    def chat_history(self, rel_path: str, *, visibility: str, current_user_id: int | None, limit: int = 12) -> list[dict[str, str]]:
        if visibility == "private" and current_user_id is None:
            return []
        with self._connect() as conn:
            paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
            if paper_id is None:
                return []
            owner_user_id = current_user_id if visibility == "private" else None
            thread_id = self._ensure_chat_thread_locked(conn, paper_id, visibility, owner_user_id)
            messages = self._chat_messages_for_thread_locked(conn, thread_id, limit=limit)
        return [
            {"role": message.role, "display_name": message.display_name, "body": message.body}
            for message in messages
        ]

    def chat_context(self, rel_path: str, current_user_id: int | None) -> dict[str, Any]:
        empty_thread = {"messages": [], "count": 0, "visibility": "shared"}
        with self._connect() as conn:
            paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
            if paper_id is None:
                return {
                    "shared": empty_thread,
                    "private": {"messages": [], "count": 0, "visibility": "private"},
                }

            shared_thread_id = self._ensure_chat_thread_locked(conn, paper_id, "shared", None)
            shared_messages = self._chat_messages_for_thread_locked(conn, shared_thread_id, limit=16)
            shared_count = int(
                conn.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = ?", (shared_thread_id,)).fetchone()[0]
            )
            private_messages: list[TeamChatMessage] = []
            private_count = 0
            if current_user_id is not None:
                private_thread_id = self._ensure_chat_thread_locked(conn, paper_id, "private", current_user_id)
                private_messages = self._chat_messages_for_thread_locked(conn, private_thread_id, limit=16)
                private_count = int(
                    conn.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = ?", (private_thread_id,)).fetchone()[0]
                )
        return {
            "shared": {"messages": shared_messages, "count": shared_count, "visibility": "shared"},
            "private": {"messages": private_messages, "count": private_count, "visibility": "private"},
        }

    def paper_context(self, rel_path: str, current_user_id: int | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            paper_row = conn.execute(
                "SELECT id FROM papers WHERE rel_path = ?",
                (rel_path,),
            ).fetchone()
            if paper_row is None:
                return {
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
            paper_id = int(paper_row["id"])
            recommendations = [
                TeamRecommendation(
                    id=int(row["id"]),
                    user_id=int(row["user_id"]),
                    display_name=str(row["display_name"]),
                    reason=str(row["reason"]),
                    created_at=str(row["created_at"]),
                )
                for row in conn.execute(
                    """
                    SELECT pr.id, pr.user_id, pr.reason, pr.created_at, COALESCE(u.display_name, u.username) AS display_name
                    FROM paper_recommendations pr
                    JOIN users u ON u.id = pr.user_id
                    WHERE pr.paper_id = ?
                    ORDER BY pr.created_at DESC
                    """,
                    (paper_id,),
                ).fetchall()
            ]
            tags = [
                TeamTag(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    slug=str(row["slug"]),
                    source_type=str(row["source_type"]),
                    added_by_display_name=(str(row["display_name"]) if row["display_name"] else None),
                    created_at=str(row["created_at"]),
                    is_locked=bool(row["is_locked"]),
                )
                for row in conn.execute(
                    """
                    SELECT t.id, t.name, t.slug, pt.source_type, pt.created_at, pt.is_locked,
                           COALESCE(u.display_name, u.username) AS display_name
                    FROM paper_tags pt
                    JOIN tags t ON t.id = pt.tag_id
                    LEFT JOIN users u ON u.id = pt.added_by_user_id
                    WHERE pt.paper_id = ?
                    ORDER BY CASE pt.source_type WHEN 'official' THEN 0 WHEN 'ai' THEN 1 WHEN 'auto' THEN 2 ELSE 3 END, LOWER(t.name)
                    """,
                    (paper_id,),
                ).fetchall()
            ]
            comments = self._load_comments_locked(conn, paper_id)
            recommended_by_current_user = False
            current_user_recommendation_reason = ""
            if current_user_id is not None:
                recommendation_row = conn.execute(
                    "SELECT reason FROM paper_recommendations WHERE paper_id = ? AND user_id = ?",
                    (paper_id, current_user_id),
                ).fetchone()
                recommended_by_current_user = recommendation_row is not None
                if recommendation_row is not None:
                    current_user_recommendation_reason = str(recommendation_row["reason"] or "")
            prompt_runs = [
                TeamPromptRun(
                    prompt_slug=str(row["prompt_slug"]),
                    prompt_name=str(row["prompt_name"]),
                    prompt_version=int(row["prompt_version"]),
                    model=str(row["model"]),
                    generated_by_display_name=(str(row["display_name"]) if row["display_name"] else None),
                    generated_at=(str(row["generated_at"]) if row["generated_at"] else None),
                    result_rel_path=(str(row["result_rel_path"]) if row["result_rel_path"] else None),
                    status=str(row["status"]),
                )
                for row in conn.execute(
                    """
                    SELECT pr.prompt_slug, pr.prompt_name, pr.prompt_version, pr.model, pr.generated_at,
                           pr.result_rel_path, pr.status, COALESCE(u.display_name, u.username) AS display_name
                    FROM prompt_runs pr
                    LEFT JOIN users u ON u.id = pr.triggered_by_user_id
                    WHERE pr.paper_id = ? AND pr.shared = 1 AND pr.status = 'completed'
                    ORDER BY pr.generated_at DESC, pr.updated_at DESC
                    """,
                    (paper_id,),
                ).fetchall()
            ]
            sources = [
                {
                    "source_type": str(row["source_type"]),
                    "source_value": str(row["source_value"]),
                    "source_url": (str(row["source_url"]) if row["source_url"] else None),
                    "source_context": (str(row["source_context"]) if row["source_context"] else None),
                    "sender_open_id": (str(row["sender_open_id"]) if row["sender_open_id"] else None),
                    "message_time_iso": (str(row["message_time_iso"]) if row["message_time_iso"] else None),
                }
                for row in conn.execute(
                    """
                    SELECT source_type, source_value, source_url, source_context, sender_open_id, message_time_iso
                    FROM paper_sources WHERE paper_id = ? ORDER BY created_at DESC, id DESC
                    """,
                    (paper_id,),
                ).fetchall()
            ]
            attachments = [
                PaperAttachment(
                    id=int(row["id"]),
                    kind=str(row["kind"]),
                    title=str(row["title"]),
                    local_path=(str(row["local_path"]) if row["local_path"] else None),
                    source_url=str(row["source_url"]),
                    created_at=str(row["created_at"]),
                )
                for row in conn.execute(
                    """
                    SELECT id, kind, title, local_path, source_url, created_at
                    FROM paper_attachments WHERE paper_id = ? ORDER BY created_at ASC, id ASC LIMIT 3
                    """,
                    (paper_id,),
                ).fetchall()
            ]
        return {
            "recommendations": recommendations,
            "tags": tags,
            "comments": comments,
            "like_count": len(recommendations),
            "liked_by_current_user": recommended_by_current_user,
            "recommended_by_current_user": recommended_by_current_user,
            "recommendation_count": len(recommendations),
            "comment_count": len(list(flatten_comments(comments))),
            "prompt_runs": prompt_runs,
            "sources": sources,
            "attachments": attachments,
            "current_user_recommendation_reason": current_user_recommendation_reason,
        }

    def recent_recommendation_feed(self, *, limit: int = 5, window_days: int = 14) -> list[dict[str, Any]]:
        cutoff = (datetime.utcnow() - timedelta(days=max(1, int(window_days)))).isoformat(timespec="seconds")
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH recommendation_counts AS (
                    SELECT paper_id, COUNT(*) AS recommendation_count, MAX(updated_at) AS latest_recommendation_at
                    FROM paper_recommendations
                    GROUP BY paper_id
                ),
                comment_counts AS (
                    SELECT paper_id, COUNT(*) AS comment_count, MAX(created_at) AS latest_comment_at
                    FROM comments
                    GROUP BY paper_id
                ),
                recent_activity AS (
                    SELECT paper_id, MAX(activity_at) AS latest_activity
                    FROM (
                        SELECT paper_id, updated_at AS activity_at FROM paper_recommendations WHERE updated_at >= ?
                        UNION ALL
                        SELECT paper_id, created_at AS activity_at FROM comments WHERE created_at >= ?
                    ) recent_rows
                    GROUP BY paper_id
                )
                SELECT p.rel_path, p.display_title, p.file_name, p.extracted_date,
                       COALESCE(rc.recommendation_count, 0) AS recommendation_count,
                       COALESCE(cc.comment_count, 0) AS comment_count,
                       recent_activity.latest_activity
                FROM recent_activity
                JOIN papers p ON p.id = recent_activity.paper_id
                LEFT JOIN recommendation_counts rc ON rc.paper_id = p.id
                LEFT JOIN comment_counts cc ON cc.paper_id = p.id
                WHERE p.is_done = 0
                  AND COALESCE(rc.recommendation_count, 0) > 0
                ORDER BY COALESCE(rc.recommendation_count, 0) DESC,
                         COALESCE(cc.comment_count, 0) DESC,
                         recent_activity.latest_activity DESC,
                         LOWER(p.display_title) ASC
                LIMIT ?
                """,
                (cutoff, cutoff, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "rel_path": str(row["rel_path"]),
                "display_title": str(row["display_title"]),
                "file_name": str(row["file_name"]),
                "extracted_date": (str(row["extracted_date"]) if row["extracted_date"] else None),
                "recommendation_count": int(row["recommendation_count"]),
                "like_count": int(row["recommendation_count"]),
                "comment_count": int(row["comment_count"]),
                "latest_activity": str(row["latest_activity"]),
            }
            for row in rows
        ]

    def record_prompt_run(
        self,
        rel_path: str,
        *,
        prompt_slug: str,
        prompt_name: str,
        prompt_version_id: int | None,
        prompt_version: int,
        model: str,
        result_rel_path: str | None,
        status: str,
        triggered_by_user_id: int | None,
        generated_at: str | None = None,
        shared: bool = True,
    ) -> None:
        with self._write_lock:
            with self._connect() as conn:
                paper_id = self._paper_id_for_rel_path_locked(conn, rel_path)
                if paper_id is None:
                    return
                now = self._timestamp()
                conn.execute(
                    """
                    INSERT INTO prompt_runs(
                        paper_id, prompt_slug, prompt_name, prompt_version_id, prompt_version, model,
                        triggered_by_user_id, status, shared, result_rel_path, generated_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id, prompt_slug, prompt_version, model, shared) DO UPDATE SET
                        prompt_name = excluded.prompt_name,
                        prompt_version_id = excluded.prompt_version_id,
                        triggered_by_user_id = COALESCE(excluded.triggered_by_user_id, prompt_runs.triggered_by_user_id),
                        status = excluded.status,
                        result_rel_path = COALESCE(excluded.result_rel_path, prompt_runs.result_rel_path),
                        generated_at = COALESCE(excluded.generated_at, prompt_runs.generated_at),
                        updated_at = excluded.updated_at
                    """,
                    (
                        paper_id,
                        prompt_slug,
                        prompt_name,
                        prompt_version_id,
                        prompt_version,
                        model,
                        triggered_by_user_id,
                        status,
                        1 if shared else 0,
                        result_rel_path,
                        generated_at,
                        now,
                    ),
                )

    def backfill_prompt_run(
        self,
        rel_path: str,
        *,
        prompt_slug: str,
        prompt_name: str,
        prompt_version_id: int | None,
        prompt_version: int,
        model: str,
        result_rel_path: str,
        generated_at: str,
    ) -> None:
        self.record_prompt_run(
            rel_path,
            prompt_slug=prompt_slug,
            prompt_name=prompt_name,
            prompt_version_id=prompt_version_id,
            prompt_version=prompt_version,
            model=model,
            result_rel_path=result_rel_path,
            status="completed",
            triggered_by_user_id=None,
            generated_at=generated_at,
            shared=True,
        )

    def _load_comments_locked(self, conn: sqlite3.Connection, paper_id: int) -> list[TeamComment]:
        rows = conn.execute(
            """
            SELECT c.id, c.user_id, c.parent_id, c.body, c.created_at,
                   COALESCE(u.display_name, u.username) AS display_name,
                   u.avatar_rel_path AS avatar_rel_path
            FROM comments c
            JOIN users u ON u.id = c.user_id
            WHERE c.paper_id = ?
            ORDER BY c.created_at ASC, c.id ASC
            """,
            (paper_id,),
        ).fetchall()
        comments_by_id: dict[int, TeamComment] = {}
        roots: list[TeamComment] = []
        for row in rows:
            comment = TeamComment(
                id=int(row["id"]),
                user_id=int(row["user_id"]),
                display_name=str(row["display_name"]),
                avatar_rel_path=(str(row["avatar_rel_path"]) if row["avatar_rel_path"] else None),
                body=str(row["body"]),
                created_at=str(row["created_at"]),
                parent_id=(int(row["parent_id"]) if row["parent_id"] is not None else None),
                replies=[],
            )
            comments_by_id[comment.id] = comment
            if comment.parent_id and comment.parent_id in comments_by_id:
                comments_by_id[comment.parent_id].replies.append(comment)
            else:
                roots.append(comment)
        return roots

    def _paper_id_for_rel_path_locked(self, conn: sqlite3.Connection, rel_path: str) -> int | None:
        row = conn.execute("SELECT id FROM papers WHERE rel_path = ?", (rel_path,)).fetchone()
        return int(row["id"]) if row is not None else None

    def _persist_manual_date_from_record_locked(self, conn: sqlite3.Connection, paper_id: int | None, paper: Any) -> None:
        if paper_id is None or getattr(paper, "date_source", None) != "manual":
            return
        display_date = str(getattr(paper, "extracted_date", "") or "").strip()
        sort_date = str(getattr(paper, "sort_date", "") or "").strip()
        precision = str(getattr(paper, "date_precision", "") or "month").strip() or "month"
        if not display_date or not sort_date or precision not in {"year", "month", "day"}:
            return
        now = self._timestamp()
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_date_overrides(
                paper_id, display_date, precision, sort_date, updated_by_user_id, updated_at
            ) VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (paper_id, display_date, precision, sort_date, now),
        )

    def _apply_date_override_to_paper_locked(self, conn: sqlite3.Connection, paper_id: int | None) -> None:
        if paper_id is None:
            return
        row = conn.execute(
            """
            SELECT display_date, precision, sort_date
            FROM paper_date_overrides
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """
            UPDATE papers
            SET extracted_date = ?, sort_date = ?, date_precision = ?, date_source = 'manual', updated_at = ?
            WHERE id = ?
            """,
            (str(row["display_date"]), str(row["sort_date"]), str(row["precision"] or "month"), self._timestamp(), paper_id),
        )

    def _ensure_auto_tags_locked(self, conn: sqlite3.Connection, paper_id: int | None, paper: Any) -> None:
        if paper_id is None:
            return
        suggestions = generate_auto_tags(
            title=str(getattr(paper, "display_title", "") or getattr(paper, "title", "")),
            preview_text=str(getattr(paper, "preview_text", "") or ""),
            folder=str(getattr(paper, "folder", "") or ""),
            extension=str(getattr(paper, "extension", "") or ""),
        )
        now = self._timestamp()
        desired_tag_ids: list[int] = []
        for name in suggestions:
            slug = slugify_text(name)
            if not slug:
                continue
            row = conn.execute("SELECT id FROM tags WHERE slug = ?", (slug,)).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO tags(name, slug, created_by_user_id, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)",
                    (name, slug, now, now),
                )
                created_tag_id = cursor.lastrowid
                if created_tag_id is None:
                    raise RuntimeError("Failed to persist auto tag.")
                tag_id = int(created_tag_id)
            else:
                tag_id = int(row["id"])
            desired_tag_ids.append(tag_id)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_tags(paper_id, tag_id, source_type, score, added_by_user_id, created_at, is_locked)
                VALUES (?, ?, 'auto', 1.0, NULL, ?, 1)
                """,
                (paper_id, tag_id, now),
            )
        if desired_tag_ids:
            placeholders = ", ".join("?" for _ in desired_tag_ids)
            conn.execute(
                f"DELETE FROM paper_tags WHERE paper_id = ? AND source_type = 'auto' AND tag_id NOT IN ({placeholders})",
                (paper_id, *desired_tag_ids),
            )
        else:
            conn.execute("DELETE FROM paper_tags WHERE paper_id = ? AND source_type = 'auto'", (paper_id,))


def flatten_comments(comments: Iterable[TeamComment]) -> Iterable[TeamComment]:
    for comment in comments:
        yield comment
        yield from flatten_comments(comment.replies)


def slugify_text(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-")[:80]


def _extend_unique(items: list[str], *candidates: str) -> None:
    seen = {slugify_text(item) for item in items}
    for candidate in candidates:
        cleaned = candidate.strip()
        normalized = slugify_text(cleaned)
        if not normalized or normalized in seen:
            continue
        items.append(cleaned)
        seen.add(normalized)


def _match_specialized_auto_tags(title: str, text: str) -> list[str]:
    title_matches: list[str] = []
    body_matches: list[str] = []
    low_priority_matches: list[str] = []
    for tag_name, patterns in _SPECIALIZED_AUTO_TAG_PATTERNS:
        matched_in_title = any(re.search(pattern, title, flags=re.IGNORECASE) for pattern in patterns)
        matched_in_text = matched_in_title or any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
        if not matched_in_text:
            continue
        if tag_name in _LOW_PRIORITY_AUTO_TAGS:
            low_priority_matches.append(tag_name)
        elif matched_in_title:
            title_matches.append(tag_name)
        else:
            body_matches.append(tag_name)
    all_matches = [*title_matches, *body_matches, *low_priority_matches]
    if "on-policy distillation" in all_matches:
        title_matches = [tag for tag in title_matches if tag != "distillation"]
        body_matches = [tag for tag in body_matches if tag != "distillation"]
        low_priority_matches = [tag for tag in low_priority_matches if tag != "distillation"]
    return [*title_matches, *body_matches, *low_priority_matches]


def _fallback_title_tags(title: str, preview_text: str, folder: str, blocked: set[str], blocked_phrases: set[str]) -> list[str]:
    text = " ".join(part for part in [title, preview_text[:6000], folder] if part)
    title_tokens = [token.lower() for token in _TITLE_TOKEN_RE.findall(title)]
    words = [token.lower() for token in _TOKEN_RE.findall(text)]
    seen_counts: dict[str, int] = {}
    for word in words:
        normalized = slugify_text(word)
        if (
            not normalized
            or normalized in _AUTO_TAG_STOPWORDS
            or normalized in blocked
            or any(normalized in phrase for phrase in blocked_phrases)
            or len(normalized) < 4
        ):
            continue
        seen_counts[normalized] = seen_counts.get(normalized, 0) + 1

    ranked = sorted(
        seen_counts.items(),
        key=lambda item: (
            0 if item[0] in title_tokens else 1,
            -item[1],
            -len(item[0]),
            item[0],
        ),
    )
    return [word.replace("-", " ") for word, _ in ranked[:3]]


def generate_auto_tags(*, title: str, preview_text: str, folder: str, extension: str) -> list[str]:
    del extension
    text = " ".join(part for part in [title, preview_text[:8000], folder] if part)
    candidates: list[str] = []

    specialized_candidates = _match_specialized_auto_tags(title, text)
    _extend_unique(candidates, *specialized_candidates)
    blocked_fallback_tokens: set[str] = set()
    blocked_specialized_phrases: set[str] = set()
    for item in specialized_candidates:
        normalized = slugify_text(item)
        if not normalized:
            continue
        blocked_specialized_phrases.add(normalized)
        blocked_fallback_tokens.add(normalized)
        blocked_fallback_tokens.update(part for part in normalized.split("-") if len(part) >= 4)
    _extend_unique(
        candidates,
        *_fallback_title_tags(title, preview_text, folder, blocked_fallback_tokens, blocked_specialized_phrases),
    )

    filtered: list[str] = []
    for item in candidates:
        normalized = slugify_text(item)
        if not normalized or normalized in _AUTO_TAG_STOPWORDS:
            continue
        filtered.append(item)
        if len(filtered) >= 6:
            break
    return filtered
