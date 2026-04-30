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
    """Return list of note IDs referenced via [[NoteTitle]] syntax."""
    raw = re.findall(r"\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]", text)
    ids = []
    for r in raw:
        note_id = r.strip().replace(" ", "-")
        if note_id not in ids:
            ids.append(note_id)
    return ids


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
        html = md_lib.markdown(body, extensions=["tables", "fenced_code"])

        notes.append({
            "id": file_id(path),
            "title": meta.get("title") or path.stem,
            "tags": [str(t) for t in tags] if tags else [],
            "year": str(meta.get("year", "")),
            "tldr": meta.get("tldr", ""),
            "aliases": [str(a) for a in aliases] if aliases else [],
            "links": links,
            "html": html,
        })

    out_path = DOCS_DIR / "notes.json"
    out_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Processed {len(notes)} notes → {out_path}")


if __name__ == "__main__":
    build()
