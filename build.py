#!/usr/bin/env python3
"""Build notes.json from wiki/*.md for the graph viewer."""

import json
import re
import subprocess
import sys
from pathlib import Path

WIKI_DIR = Path(__file__).parent / "wiki"
DOCS_DIR = Path(__file__).parent / "docs"


def ensure_deps():
    missing = []
    for pkg in ("yaml", "markdown"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("pyyaml" if pkg == "yaml" else pkg)
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def parse_frontmatter(text):
    """Return (metadata_dict, body_text). Frontmatter is optional."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            import yaml
            fm_text = text[3:end].strip()
            body = text[end + 4:].lstrip("\n")
            try:
                meta = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, body
    return {}, text


def extract_wikilinks(text):
    """Return list of note IDs referenced via [[NoteTitle]] or ![[NoteTitle]] syntax.

    Skips wikilinks inside inline-code spans (`...`) and fenced code blocks so
    template placeholders like `[[${name}]]` or prose like `[[linked]]` in
    documentation text are not treated as real links.
    """
    # Strip inline code spans before scanning (replace with same-length whitespace
    # so that character positions stay consistent; we only need the [[...]] matches).
    cleaned = re.sub(r"`[^`\n]+`", lambda m: " " * len(m.group()), text)
    # Also strip fenced code blocks (``` ... ```)
    cleaned = re.sub(r"```.*?```", lambda m: " " * len(m.group()), cleaned, flags=re.DOTALL)
    raw = re.findall(r"!?\[\[([^\]|#]+?)(?:[\\|#][^\]]*)?\]\]", cleaned)
    ids = []
    for r in raw:
        # Strip trailing backslash — Obsidian uses \| inside table cells to escape the pipe
        note_id = r.strip().rstrip("\\").strip().replace(" ", "-")
        if note_id not in ids:
            ids.append(note_id)
    return ids


# ── Obsidian-specific syntax preprocessor ───────────────────────────────────
DATAVIEW_NOTICE = (
    '<div class="obsidian-note">'
    '📊 Live Dataview query — view in Obsidian'
    '</div>'
)
MERMAID_NOTICE = (
    '<div class="obsidian-note">'
    '📈 Diagram — view in Obsidian'
    '</div>'
)

CALLOUT_TYPES = {
    "note":     ("📝", "Note"),
    "info":     ("ℹ️", "Info"),
    "tip":      ("💡", "Tip"),
    "hint":     ("💡", "Hint"),
    "warning":  ("⚠️", "Warning"),
    "caution":  ("⚠️", "Caution"),
    "danger":   ("🛑", "Danger"),
    "important":("❗", "Important"),
    "summary":  ("📌", "Summary"),
    "abstract": ("📌", "Abstract"),
    "example":  ("🔬", "Example"),
    "question": ("❓", "Question"),
    "quote":    ("❝",  "Quote"),
}


def _strip_fenced_blocks(text, lang):
    """Replace ```{lang} ... ``` (case-insensitive) with a one-line marker we
    can later swap for an HTML notice (after the markdown lib has run)."""
    pattern = re.compile(
        r"^```[ \t]*" + lang + r"\b[^\n]*\n.*?^```[ \t]*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub("", text)


def _convert_callouts(text):
    """Convert Obsidian callouts (> [!type] Optional Title) to styled HTML divs.

    A callout is a blockquote whose first line begins with `[!type]`. Following
    blockquote lines (starting with `>`) form the body. We emit raw HTML so the
    markdown library leaves it alone."""
    lines = text.split("\n")
    out = []
    i = 0
    callout_re = re.compile(r"^>\s*\[!(?P<type>[A-Za-z]+)\][+\-]?\s*(?P<title>.*)$")
    while i < len(lines):
        m = callout_re.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ctype = m.group("type").lower()
        ctitle = m.group("title").strip()
        body_lines = []
        i += 1
        while i < len(lines) and lines[i].lstrip().startswith(">"):
            body_lines.append(re.sub(r"^>\s?", "", lines[i]))
            i += 1
        emoji, default_label = CALLOUT_TYPES.get(ctype, ("💬", ctype.capitalize()))
        title_html = (ctitle or default_label)
        body_html = "<br>".join(l for l in body_lines if l.strip()) if body_lines else ""
        out.append(
            f'<div class="obsidian-note"><strong>{emoji} {title_html}</strong>'
            + (f"<br>{body_html}" if body_html else "")
            + "</div>"
        )
    return "\n".join(out)


def _convert_embeds(text):
    """Convert ![[NoteName]] embeds to clickable links pointing at the note."""
    def repl(m):
        ref = m.group(1).strip()
        note_id = ref.replace(" ", "-")
        return f'<a href="#" class="wikilink" data-noteid="{note_id}">📎 {ref}</a>'
    return re.sub(r"!\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]", repl, text)


# ── Math protection ─────────────────────────────────────────────────────────
# We swap LaTeX math segments for plain alphanumeric placeholders before the
# markdown renderer runs, then put the raw LaTeX back afterwards. This keeps
# the markdown library from interpreting `_`, `*`, `[`, etc. inside math, and
# it gives KaTeX the un-escaped LaTeX it needs at render time.

_MATH_PLACEHOLDER_FMT = "MATHXBLOCKX{:06d}XEND"


def _split_code_aware(text):
    """Yield (is_code, segment) tuples, splitting on fenced code blocks so we
    don't extract math from inside ``` blocks."""
    pattern = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            yield False, text[last:m.start()]
        yield True, m.group(0)
        last = m.end()
    if last < len(text):
        yield False, text[last:]


def protect_math(text):
    """Replace $$...$$, $...$, \\[...\\], \\(...\\) outside fenced code blocks
    with placeholders. Returns (modified_text, placeholders_dict)."""
    placeholders = {}
    counter = [0]

    def make_ph(raw_with_delims):
        counter[0] += 1
        key = _MATH_PLACEHOLDER_FMT.format(counter[0])
        placeholders[key] = raw_with_delims
        return key

    out_parts = []
    for is_code, seg in _split_code_aware(text):
        if is_code:
            out_parts.append(seg)
            continue

        # Order matters: longest delimiters first so $$ doesn't get eaten by $.
        # 1. $$ ... $$  (display, may span lines)
        seg = re.sub(
            r"\$\$(.+?)\$\$",
            lambda m: make_ph("$$" + m.group(1) + "$$"),
            seg,
            flags=re.DOTALL,
        )
        # 2. \[ ... \]  (display, may span lines)
        seg = re.sub(
            r"\\\[(.+?)\\\]",
            lambda m: make_ph("\\[" + m.group(1) + "\\]"),
            seg,
            flags=re.DOTALL,
        )
        # 3. \( ... \)  (inline)
        seg = re.sub(
            r"\\\((.+?)\\\)",
            lambda m: make_ph("\\(" + m.group(1) + "\\)"),
            seg,
            flags=re.DOTALL,
        )
        # 4. $ ... $   (inline). Reject empty bodies, leading/trailing whitespace,
        #    and $ that's been escaped with a backslash.
        seg = re.sub(
            r"(?<!\\)\$(?!\s)([^\n$]+?)(?<!\s)\$",
            lambda m: make_ph("$" + m.group(1) + "$"),
            seg,
        )
        out_parts.append(seg)

    return "".join(out_parts), placeholders


def restore_math(html, placeholders):
    for key, raw in placeholders.items():
        html = html.replace(key, raw)
    return html


def preprocess_obsidian(text):
    """Apply all Obsidian-specific transforms before markdown rendering."""
    # Strip Dataview blocks (replace with a placeholder line we'll swap later).
    # Use plain alphanumerics so the markdown renderer does not HTML-escape it.
    placeholder_dv = "OBSIDIANNOTEXDATAVIEWXBLOCK"
    placeholder_mm = "OBSIDIANNOTEXMERMAIDXBLOCK"

    def replace_block(m, marker):
        return marker

    text = re.sub(
        r"^```[ \t]*(?:dataview|dataviewjs)\b[^\n]*\n.*?^```[ \t]*$",
        placeholder_dv,
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"^```[ \t]*mermaid\b[^\n]*\n.*?^```[ \t]*$",
        placeholder_mm,
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    # Obsidian callouts → HTML divs
    text = _convert_callouts(text)

    # ![[note]] embeds → clickable links (do this before extract_wikilinks
    # so the link survives markdown rendering as raw HTML)
    text = _convert_embeds(text)

    return text, placeholder_dv, placeholder_mm


def file_id(path: Path) -> str:
    return path.stem.replace(" ", "-")


def build():
    ensure_deps()
    import markdown as md_lib

    DOCS_DIR.mkdir(exist_ok=True)

    md_files = sorted(WIKI_DIR.glob("*.md"))
    notes = []

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)

        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        aliases = meta.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",")]

        links = extract_wikilinks(body)

        # Protect math BEFORE other transforms so markdown / Obsidian rules
        # never see the LaTeX. KaTeX needs the raw LaTeX at render time.
        body_protected, math_placeholders = protect_math(body)

        processed, ph_dv, ph_mm = preprocess_obsidian(body_protected)
        html = md_lib.markdown(processed, extensions=["tables", "fenced_code"])
        # Swap placeholder text for the final HTML notice (post-render so the
        # markdown library never wraps the marker in a <p>).
        html = html.replace(f"<p>{ph_dv}</p>", DATAVIEW_NOTICE)
        html = html.replace(f"<p>{ph_mm}</p>", MERMAID_NOTICE)
        html = html.replace(ph_dv, DATAVIEW_NOTICE)
        html = html.replace(ph_mm, MERMAID_NOTICE)

        # Restore the raw LaTeX so KaTeX can render it on the client.
        html = restore_math(html, math_placeholders)

        citation_count = meta.get("citation_count")
        cited_by_details = meta.get("cited_by_details")

        # Strip HTML tags → plain text for full-text search
        search_text = re.sub(r'<[^>]+>', ' ', html)
        # Unescape common HTML entities
        search_text = search_text.replace('&amp;', '&').replace('&lt;', '<') \
                                 .replace('&gt;', '>').replace('&quot;', '"') \
                                 .replace('&#39;', "'").replace('&nbsp;', ' ')
        search_text = re.sub(r'\s+', ' ', search_text).strip().lower()

        notes.append({
            "id": file_id(path),
            "title": meta.get("title") or path.stem,
            "authors": meta.get("authors", ""),
            "tags": [str(t) for t in tags] if tags else [],
            "year": str(meta.get("year", "")),
            "tldr": meta.get("tldr", ""),
            "aliases": [str(a) for a in aliases] if aliases else [],
            "links": links,
            "citation_count": int(citation_count) if citation_count else None,
            "arxiv": str(meta.get("arxiv", "")) if meta.get("arxiv") else "",
            "technical_report": str(meta.get("technical_report", "")) if meta.get("technical_report") else "",
            "source_type": str(meta.get("source_type", "")) if meta.get("source_type") else "",
            "cited_by_details": cited_by_details if isinstance(cited_by_details, list) else None,
            "html": html,
            "search_text": search_text,
        })

    # Second pass: compute backlinks (inverse of links[])
    id_set = {n["id"] for n in notes}
    backlink_map = {n["id"]: [] for n in notes}
    for n in notes:
        for link_id in n.get("links", []):
            if link_id in backlink_map and link_id != n["id"]:
                backlink_map[link_id].append(n["id"])
    for n in notes:
        n["backlinks"] = backlink_map.get(n["id"], [])

    out_path = DOCS_DIR / "notes.json"
    out_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Processed {len(notes)} notes → {out_path}")

    # Warn about broken wikilinks so authors notice them at build time.
    # Normalize everything to lowercase-with-hyphens for comparison.
    note_id_norm  = {re.sub(r"[\s_]", "-", n["id"].lower()) for n in notes}
    note_ttl_norm = {n.get("title", "").lower() for n in notes}
    # Notes whose broken links are intentional external references (e.g. Citation Map's
    # "Top 10 citing papers" rows point to papers not in the wiki by design).
    INTENTIONAL_EXTERNAL_SOURCES = {"Citation-Map"}
    broken_links = []
    for n in notes:
        if n["id"] in INTENTIONAL_EXTERNAL_SOURCES:
            continue
        for link_id in n.get("links", []):
            clean = re.sub(r"[\s_]", "-", link_id.lower())
            if clean not in note_id_norm and link_id.lower() not in note_ttl_norm:
                broken_links.append(f"{n['title']} → [[{link_id}]]")
    if broken_links:
        print(f"  ⚠️  {len(broken_links)} broken wikilinks detected")


if __name__ == "__main__":
    build()
