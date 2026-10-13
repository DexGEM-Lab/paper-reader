from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")
_RULE_RE = re.compile(r"^([-*_])(?:\s*\1){2,}\s*$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
_CODE_LANG_RE = re.compile(r"^[A-Za-z0-9_+#.-]+$")
_MATH_ENV_BEGIN_RE = re.compile(r"^\\begin\{([A-Za-z*]+)\}$")
_MATH_ENV_SINGLE_LINE_RE = re.compile(r"^\\begin\{([A-Za-z*]+)\}.*\\end\{\1\}$")
_TASK_LIST_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")
_ANSWER_WRAPPER_LINES = {"<answer>", "</answer>"}
_MATH_BLOCK_DELIMITERS = {"$$": "$$", r"\[": r"\]"}
_PLACEHOLDER_TEMPLATE = "%%MDPH{index}%%"


def render_markdown(text: str) -> str:
    parts: list[str] = []
    paragraph: list[str] = []
    quote_lines: list[str] = []
    list_mode: str | None = None
    list_items: list[str] = []
    code_lines: list[str] = []
    code_fence: str | None = None
    code_language: str | None = None
    math_lines: list[str] = []
    math_block_end: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        joined = " ".join(line.strip() for line in paragraph if line.strip())
        if joined:
            parts.append(f"<p>{render_inline(joined)}</p>")
        paragraph = []

    def flush_quote() -> None:
        nonlocal quote_lines
        if not quote_lines:
            return
        content = " ".join(line.strip() for line in quote_lines if line.strip())
        if content:
            parts.append(f"<blockquote><p>{render_inline(content)}</p></blockquote>")
        quote_lines = []

    def flush_list() -> None:
        nonlocal list_mode, list_items
        if list_mode and list_items:
            tag = "ol" if list_mode == "ol" else "ul"
            items_html = "".join(f"<li>{_render_list_item(item)}</li>" for item in list_items)
            parts.append(f"<{tag}>{items_html}</{tag}>")
        list_mode = None
        list_items = []

    def flush_code_block() -> None:
        nonlocal code_lines, code_fence, code_language
        code_html = html.escape("\n".join(code_lines))
        class_name = ""
        language_badge = ""
        if code_language:
            escaped_language = html.escape(code_language)
            class_name = f' class="language-{escaped_language}"'
            language_badge = (
                '<div class="code-block-head">'
                f'<span class="code-block-language">{escaped_language}</span>'
                "</div>"
            )
        parts.append(
            '<div class="code-block">'
            f"{language_badge}"
            f"<pre><code{class_name}>{code_html}</code></pre>"
            "</div>"
        )
        code_lines = []
        code_fence = None
        code_language = None

    def flush_math_block() -> None:
        nonlocal math_lines, math_block_end
        if not math_lines:
            math_block_end = None
            return
        math_html = html.escape("\n".join(math_lines))
        parts.append(f'<div class="math-block">{math_html}</div>')
        math_lines = []
        math_block_end = None

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped in _ANSWER_WRAPPER_LINES:
            index += 1
            continue

        if math_block_end is not None:
            math_lines.append(line)
            if stripped == math_block_end:
                flush_math_block()
            index += 1
            continue

        if code_fence is not None:
            if _is_closing_fence(stripped, code_fence):
                flush_code_block()
            else:
                code_lines.append(line)
            index += 1
            continue

        code_match = _CODE_FENCE_RE.match(stripped)
        if code_match:
            flush_paragraph()
            flush_quote()
            flush_list()
            code_fence = code_match.group(1)
            code_language = _code_language_from_info(code_match.group(2))
            code_lines = []
            index += 1
            continue

        if _looks_like_table_header(lines, index):
            flush_paragraph()
            flush_quote()
            flush_list()
            table_html, consumed = _render_table(lines, index)
            parts.append(table_html)
            index += consumed
            continue

        if stripped in _MATH_BLOCK_DELIMITERS:
            flush_paragraph()
            flush_quote()
            flush_list()
            math_block_end = _MATH_BLOCK_DELIMITERS[stripped]
            math_lines = [line]
            index += 1
            continue

        if _MATH_ENV_SINGLE_LINE_RE.match(stripped):
            flush_paragraph()
            flush_quote()
            flush_list()
            parts.append(f'<div class="math-block">{html.escape(stripped)}</div>')
            index += 1
            continue

        env_match = _MATH_ENV_BEGIN_RE.match(stripped)
        if env_match:
            flush_paragraph()
            flush_quote()
            flush_list()
            math_block_end = rf"\end{{{env_match.group(1)}}}"
            math_lines = [line]
            index += 1
            continue

        if (stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4) or (
            stripped.startswith(r"\[") and stripped.endswith(r"\]") and len(stripped) > 4
        ):
            flush_paragraph()
            flush_quote()
            flush_list()
            parts.append(f'<div class="math-block">{html.escape(stripped)}</div>')
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            flush_quote()
            flush_list()
            index += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            flush_quote()
            flush_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{render_inline(heading_match.group(2).strip())}</h{level}>")
            index += 1
            continue

        if _RULE_RE.match(stripped):
            flush_paragraph()
            flush_quote()
            flush_list()
            parts.append("<hr>")
            index += 1
            continue

        quote_match = _BLOCKQUOTE_RE.match(stripped)
        if quote_match:
            flush_paragraph()
            flush_list()
            quote_lines.append(quote_match.group(1))
            index += 1
            continue
        flush_quote()

        bullet_match = _BULLET_RE.match(stripped)
        ordered_match = _ORDERED_RE.match(stripped)
        if bullet_match or ordered_match:
            flush_paragraph()
            mode = "ol" if ordered_match else "ul"
            item_text = ordered_match.group(1).strip() if ordered_match is not None else bullet_match.group(1).strip()
            if list_mode != mode:
                flush_list()
                list_mode = mode
            list_items.append(item_text)
            index += 1
            continue

        if list_mode:
            flush_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_quote()
    flush_list()
    if code_fence is not None:
        flush_code_block()
    if math_block_end is not None:
        flush_math_block()

    return "\n".join(parts)


def render_inline(text: str) -> str:
    placeholders: dict[str, str] = {}

    def store(fragment: str) -> str:
        key = _PLACEHOLDER_TEMPLATE.format(index=len(placeholders))
        placeholders[key] = fragment
        return key

    protected = _protect_inline_segments(text, store)
    escaped = html.escape(protected)
    escaped = _BOLD_RE.sub(lambda match: f"<strong>{match.group(1)}</strong>", escaped)
    escaped = _ITALIC_RE.sub(lambda match: f"<em>{match.group(1)}</em>", escaped)
    escaped = _STRIKE_RE.sub(lambda match: f"<del>{match.group(1)}</del>", escaped)
    for key, fragment in placeholders.items():
        escaped = escaped.replace(key, fragment)
    return escaped


def _protect_inline_segments(text: str, store: callable) -> str:
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "`":
            span_end = _find_inline_code_end(text, index)
            if span_end is not None:
                tick_count = _count_repeated_char(text, index, "`")
                inner = text[index + tick_count : span_end]
                parts.append(store(_render_inline_code(inner)))
                index = span_end + tick_count
                continue
        image_match = _IMAGE_RE.match(text, index)
        if image_match:
            parts.append(store(_render_image(image_match.group(1), image_match.group(2))))
            index = image_match.end()
            continue
        if text.startswith(r"\(", index):
            math_end = _find_escaped_delimiter_end(text, index + 2, r"\)")
            if math_end is not None:
                parts.append(store(_render_inline_math(text[index : math_end + 2])))
                index = math_end + 2
                continue
        if char == "$" and not _is_escaped(text, index) and not text.startswith("$$", index):
            math_end = _find_inline_dollar_end(text, index + 1)
            if math_end is not None:
                parts.append(store(_render_inline_math(text[index : math_end + 1])))
                index = math_end + 1
                continue
        link_match = _LINK_RE.match(text, index)
        if link_match:
            parts.append(store(_render_link(link_match.group(1), link_match.group(2))))
            index = link_match.end()
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def _render_inline_code(content: str) -> str:
    return f"<code>{html.escape(content.strip())}</code>"


def _render_inline_math(content: str) -> str:
    return html.escape(content)


def _render_image(alt: str, src: str) -> str:
    safe_alt = html.escape(alt.strip(), quote=True)
    safe_src = html.escape(src.strip(), quote=True)
    return f'<img src="{safe_src}" alt="{safe_alt}" loading="lazy">'


def _render_link(label: str, href: str) -> str:
    safe_label = html.escape(label)
    candidate = href.strip()
    try:
        scheme = urlsplit(candidate).scheme.casefold()
    except ValueError:
        scheme = ""
    if scheme not in {"http", "https", "mailto"}:
        return safe_label
    safe_href = html.escape(candidate, quote=True)
    return f'<a href="{safe_href}" target="_blank" rel="noreferrer">{safe_label}</a>'


def _render_list_item(item: str) -> str:
    match = _TASK_LIST_RE.match(item)
    if match is None:
        return render_inline(item)
    checked = match.group(1).lower() == "x"
    body = render_inline(match.group(2))
    return (
        '<label class="task-list-item">'
        f'<input type="checkbox" disabled {"checked" if checked else ""}>'
        f"<span>{body}</span>"
        "</label>"
    )


def _find_inline_code_end(text: str, start: int) -> int | None:
    tick_count = _count_repeated_char(text, start, "`")
    marker = "`" * tick_count
    position = text.find(marker, start + tick_count)
    return position if position != -1 else None


def _find_escaped_delimiter_end(text: str, start: int, delimiter: str) -> int | None:
    position = start
    while True:
        position = text.find(delimiter, position)
        if position == -1:
            return None
        if not _is_escaped(text, position):
            return position
        position += len(delimiter)


def _find_inline_dollar_end(text: str, start: int) -> int | None:
    position = start
    while position < len(text):
        if text[position] == "\n":
            return None
        if text[position] == "$" and not _is_escaped(text, position) and not text.startswith("$$", position):
            return position
        position += 1
    return None


def _count_repeated_char(text: str, start: int, char: str) -> int:
    count = 0
    while start + count < len(text) and text[start + count] == char:
        count += 1
    return count


def _is_closing_fence(stripped: str, opening_fence: str) -> bool:
    if not opening_fence:
        return False
    marker = opening_fence[0]
    required = len(opening_fence)
    if not stripped or stripped[0] != marker:
        return False
    actual = _count_repeated_char(stripped, 0, marker)
    return actual >= required and stripped[actual:].strip() == ""


def _code_language_from_info(info: str) -> str | None:
    candidate = (info or "").strip().split(" ", 1)[0].strip().lower()
    if not candidate or not _CODE_LANG_RE.fullmatch(candidate):
        return None
    return candidate


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _looks_like_table_header(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    if not header or not separator or "|" not in header or "|" not in separator:
        return False
    if _HEADING_RE.match(header) or _BULLET_RE.match(header) or _ORDERED_RE.match(header) or _RULE_RE.match(header):
        return False
    cells = _split_table_row(header)
    separator_cells = _split_table_row(separator)
    if not cells or len(cells) != len(separator_cells):
        return False
    return all(_is_table_separator_cell(cell) for cell in separator_cells)


def _render_table(lines: list[str], start_index: int) -> tuple[str, int]:
    header_cells = _split_table_row(lines[start_index].strip())
    alignments = _table_alignments(_split_table_row(lines[start_index + 1].strip()))
    body_rows: list[list[str]] = []
    index = start_index + 2
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or "|" not in stripped:
            break
        row_cells = _split_table_row(stripped)
        if not row_cells:
            break
        if len(row_cells) < len(header_cells):
            row_cells.extend([""] * (len(header_cells) - len(row_cells)))
        elif len(row_cells) > len(header_cells):
            row_cells = row_cells[: len(header_cells)]
        body_rows.append(row_cells)
        index += 1

    header_html = "".join(_render_table_cell("th", cell, alignments[idx]) for idx, cell in enumerate(header_cells))
    body_html = "".join(
        "<tr>" + "".join(_render_table_cell("td", cell, alignments[idx]) for idx, cell in enumerate(row)) + "</tr>"
        for row in body_rows
    )
    table_html = (
        '<div class="table-wrapper">'
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )
    return table_html, max(2, index - start_index)


def _render_table_cell(tag: str, value: str, alignment: str | None) -> str:
    style = f' style="text-align:{alignment}"' if alignment else ""
    return f"<{tag}{style}>{render_inline(value.strip())}</{tag}>"


def _split_table_row(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator_cell(cell: str) -> bool:
    trimmed = cell.strip()
    if not trimmed:
        return False
    core = trimmed.strip(":")
    return len(core) >= 1 and set(core) == {"-"}


def _table_alignments(cells: list[str]) -> list[str | None]:
    alignments: list[str | None] = []
    for cell in cells:
        trimmed = cell.strip()
        left = trimmed.startswith(":")
        right = trimmed.endswith(":")
        if left and right:
            alignments.append("center")
        elif right:
            alignments.append("right")
        elif left:
            alignments.append("left")
        else:
            alignments.append(None)
    return alignments
