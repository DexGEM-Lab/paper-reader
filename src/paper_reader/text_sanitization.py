"""Small, dependency-free normalization helpers for persisted reader metadata."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _TextOnlyHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def sanitize_title(value: object, *, limit: int = 500) -> str:
    """Return a compact plain-text title safe to persist and display.

    Metadata producers inconsistently return HTML markup and encoded markup.  A
    title is a label, never rich content, so decode entities and retain only its
    textual content before it reaches the database.
    """
    raw = html.unescape(str(value or "").replace("\x00", " "))
    parser = _TextOnlyHTMLParser()
    try:
        parser.feed(raw)
        parser.close()
        cleaned = " ".join(parser.parts)
    except (AssertionError, ValueError):
        cleaned = raw
    # Encoded tags become literal text after entity decoding; remove that second
    # representation too without mistaking mathematical comparisons for tags.
    cleaned = re.sub(r"</?[A-Za-z!][^>\n]{0,500}>", " ", html.unescape(cleaned))
    return " ".join(cleaned.split())[:limit]
