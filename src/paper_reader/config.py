"""Runtime configuration for Paper Reader.

Configuration is loaded from the project `.env` file and then overridden by the
process environment.  No credentials or deployment paths are embedded in code.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
DEFAULT_FEISHU_API_BASE_URL = "https://open.feishu.cn"
DEFAULT_LLM_BASE_URL = "https://litellm.dex-gem.ai/v1"
DEFAULT_LLM_MODEL = "glm-5.3"
DEFAULT_LLM_CHEAP_MODEL = "deepseek-flash"
DEFAULT_LLM_ALLOWED_MODELS = frozenset(
    {"claude-sonnet-5-5", "claude-sonnet-5", "kimi-k3", "glm-5.3", "deepseek-flash"}
)


def load_env_file_values(path: Path) -> dict[str, str]:
    """Read the small dotenv subset used by the application."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def normalize_public_base_url(raw_value: str) -> str:
    value = raw_value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("PAPER_READER_BASE_URL must be a full http/https URL without a query or fragment.")
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _path_setting(value: str, *, cwd: Path) -> Path:
    path = Path(value).expanduser()
    return (cwd / path).resolve() if not path.is_absolute() else path.resolve()


def _port_setting(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("PAPER_READER_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PAPER_READER_PORT must be between 1 and 65535.")
    return port


def _positive_float_setting(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def _positive_int_setting(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _csv_setting(value: str) -> frozenset[str]:
    return frozenset(item.strip().casefold() for item in value.split(",") if item.strip())


def _bool_setting(value: str, *, name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false.")
    return normalized == "true"


@dataclass(frozen=True, slots=True)
class PaperReaderConfig:
    data_dir: Path
    library_dir: Path
    source_archive_dir: Path
    host: str
    port: int
    secret_key: str
    secret_key_is_ephemeral: bool
    login_username: str | None
    login_password: str | None
    admin_usernames: frozenset[str]
    public_base_url: str
    feishu_app_id: str
    feishu_app_secret: str
    feishu_api_base_url: str
    feishu_allowed_chat_id: str
    auth_mode: str
    auth_mode_is_default: bool
    session_ttl_hours: int
    session_cookie_secure: bool
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_cheap_model: str
    llm_allowed_models: frozenset[str]
    llm_qps: float
    llm_timeout: float
    llm_context_tokens: int
    max_upload_bytes: int
    fetch_max_bytes: int
    fetch_page_max_bytes: int
    fetch_max_redirects: int
    screen_imports: bool

    @property
    def feishu_enabled(self) -> bool:
        return bool(self.public_base_url or self.feishu_app_id or self.feishu_app_secret)

    @property
    def feishu_oauth_configured(self) -> bool:
        return bool(self.public_base_url and self.feishu_app_id and self.feishu_app_secret)


def load_config(
    *,
    project_dir: Path | None = None,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PaperReaderConfig:
    """Load Paper Reader settings without touching application storage."""
    project_dir = (project_dir or Path(__file__).resolve().parents[2]).resolve()
    cwd = (cwd or Path.cwd()).resolve()
    environment = os.environ if environ is None else environ
    env_file = Path(environment.get("PAPER_READER_ENV_FILE", str(project_dir / ".env"))).expanduser()
    env_values = load_env_file_values(env_file)

    def setting(key: str, default: str = "") -> str:
        return str(environment.get(key) or env_values.get(key) or default).strip()

    data_dir = _path_setting(setting("PAPER_READER_DATA_DIR", "data"), cwd=cwd)
    library_dir = _path_setting(setting("PAPER_READER_LIBRARY_DIR", str(data_dir / "library")), cwd=cwd)
    source_archive_dir = _path_setting(
        setting("PAPER_READER_SOURCES_ARCHIVE_DIR", str(data_dir / "sources" / "huggingface_daily")), cwd=cwd
    )
    login_username = setting("PAPER_READER_LOGIN_USERNAME") or None
    login_password = setting("PAPER_READER_LOGIN_PASSWORD") or None
    if bool(login_username) != bool(login_password):
        raise ValueError("Set both PAPER_READER_LOGIN_USERNAME and PAPER_READER_LOGIN_PASSWORD, or neither.")

    secret_key = setting("PAPER_READER_SECRET_KEY")
    secret_key_is_ephemeral = not bool(secret_key)
    if secret_key_is_ephemeral:
        secret_key = secrets.token_urlsafe(32)

    public_base_url = normalize_public_base_url(setting("PAPER_READER_BASE_URL"))
    feishu_app_id = setting("PAPER_READER_FEISHU_APP_ID")
    feishu_app_secret = setting("PAPER_READER_FEISHU_APP_SECRET")
    feishu_oauth_configured = bool(public_base_url and feishu_app_id and feishu_app_secret)
    raw_auth_mode = setting("PAPER_READER_AUTH_MODE").casefold()
    if raw_auth_mode and raw_auth_mode not in {"feishu", "local", "mixed"}:
        raise ValueError("PAPER_READER_AUTH_MODE must be one of: feishu, local, mixed.")
    auth_mode = raw_auth_mode or ("feishu" if feishu_oauth_configured else "local")
    secure_cookie_default = "true" if urlparse(public_base_url).scheme == "https" else "false"
    raw_secure_cookie = setting("PAPER_READER_SESSION_COOKIE_SECURE", secure_cookie_default).casefold()
    if raw_secure_cookie not in {"true", "false"}:
        raise ValueError("PAPER_READER_SESSION_COOKIE_SECURE must be true or false.")

    return PaperReaderConfig(
        data_dir=data_dir,
        library_dir=library_dir,
        source_archive_dir=source_archive_dir,
        host=setting("PAPER_READER_HOST", "0.0.0.0"),
        port=_port_setting(setting("PAPER_READER_PORT", "8000")),
        secret_key=secret_key,
        secret_key_is_ephemeral=secret_key_is_ephemeral,
        login_username=login_username,
        login_password=login_password,
        admin_usernames=_csv_setting(setting("PAPER_READER_ADMIN_USERS")),
        public_base_url=public_base_url,
        feishu_app_id=feishu_app_id,
        feishu_app_secret=feishu_app_secret,
        feishu_api_base_url=normalize_public_base_url(
            setting("PAPER_READER_FEISHU_API_BASE_URL", DEFAULT_FEISHU_API_BASE_URL)
        ),
        feishu_allowed_chat_id=setting("PAPER_READER_FEISHU_ALLOWED_CHAT_ID"),
        auth_mode=auth_mode,
        auth_mode_is_default=not bool(raw_auth_mode),
        session_ttl_hours=_positive_int_setting(
            setting("PAPER_READER_SESSION_TTL_HOURS", "12"), name="PAPER_READER_SESSION_TTL_HOURS"
        ),
        session_cookie_secure=raw_secure_cookie == "true",
        llm_base_url=normalize_public_base_url(setting("PAPER_READER_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)),
        llm_api_key=setting("PAPER_READER_LLM_API_KEY"),
        llm_model=setting("PAPER_READER_LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_cheap_model=setting("PAPER_READER_LLM_CHEAP_MODEL", DEFAULT_LLM_CHEAP_MODEL),
        llm_allowed_models=_csv_setting(
            setting("PAPER_READER_LLM_ALLOWED_MODELS", ",".join(sorted(DEFAULT_LLM_ALLOWED_MODELS)))
        ),
        llm_qps=_positive_float_setting(setting("PAPER_READER_LLM_QPS", "10"), name="PAPER_READER_LLM_QPS"),
        llm_timeout=_positive_float_setting(setting("PAPER_READER_LLM_TIMEOUT", "120"), name="PAPER_READER_LLM_TIMEOUT"),
        llm_context_tokens=_positive_int_setting(
            setting("PAPER_READER_LLM_CONTEXT_TOKENS", "12000"), name="PAPER_READER_LLM_CONTEXT_TOKENS"
        ),
        max_upload_bytes=_positive_int_setting(
            setting("PAPER_READER_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)),
            name="PAPER_READER_MAX_UPLOAD_BYTES",
        ),
        fetch_max_bytes=_positive_int_setting(
            setting("PAPER_READER_FETCH_MAX_BYTES", str(100 * 1024 * 1024)),
            name="PAPER_READER_FETCH_MAX_BYTES",
        ),
        fetch_page_max_bytes=_positive_int_setting(
            setting("PAPER_READER_PAGE_FETCH_MAX_BYTES", str(25 * 1024 * 1024)),
            name="PAPER_READER_PAGE_FETCH_MAX_BYTES",
        ),
        fetch_max_redirects=_positive_int_setting(
            setting("PAPER_READER_FETCH_MAX_REDIRECTS", "10"),
            name="PAPER_READER_FETCH_MAX_REDIRECTS",
        ),
        screen_imports=_bool_setting(
            setting("PAPER_READER_SCREEN_IMPORTS", "true"),
            name="PAPER_READER_SCREEN_IMPORTS",
        ),
    )
