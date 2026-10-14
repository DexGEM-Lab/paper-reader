"""Bounds checks for dates inferred from untrusted document and web metadata."""

from __future__ import annotations

from datetime import date, timedelta
import re

EARLIEST_INFERRED_DATE = date(1990, 1, 1)
_MAX_FUTURE_DAYS = 30
_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def sane_date(value: object, *, today: date | None = None) -> date | None:
    """Parse an ISO date prefix only when it is plausible for imported metadata."""
    match = _DATE_PREFIX.match(str(value or "").strip())
    if match is None:
        return None
    try:
        parsed = date.fromisoformat(match.group(1))
    except ValueError:
        return None
    maximum = (today or date.today()) + timedelta(days=_MAX_FUTURE_DAYS)
    return parsed if EARLIEST_INFERRED_DATE <= parsed <= maximum else None


def sane_sort_date(value: object, *, today: date | None = None) -> str | None:
    parsed = sane_date(value, today=today)
    return parsed.isoformat() if parsed is not None else None
