"""
agent.py — Agentic wiki ingestion system
Uses Claude (via Vertex AI) with tool use to find, evaluate,
and add content to the LLM wiki from multiple sources.

Usage:
    python3 agent.py daily              # check arXiv + GitHub last 24 h
    python3 agent.py topic <query>      # add content about a topic
    python3 agent.py citations          # track high-impact citing papers
"""

import anthropic
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# Wiki location: override with WIKI_VAULT env var; defaults to this file's directory.
VAULT = Path(os.environ.get("WIKI_VAULT", Path(__file__).resolve().parent))
RAW   = VAULT / "raw"
WIKI  = VAULT / "wiki"
LOG   = VAULT / "agent_log.json"


def _strip_md_fence(text: str) -> str:
    """Strip leading/trailing markdown code fences that the API
    sometimes wraps responses in.

    Handles ```markdown ... ```, ```md ... ```, and bare ``` ... ```.
    """
    text = text.strip()
    # Remove opening fence (```markdown, ```md, ```)
    text = re.sub(r'^```(?:markdown|md|)[\r\n]+', '', text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r'[\r\n]+```\s*$', '', text)
    return text.strip() + '\n'

WIKI_THEMES = [
    "transformer", "attention", "state space model", "SSM",
    "mamba", "LSTM", "speculative decoding", "KV cache",
    "mixture of experts", "MoE", "flash attention",
    "hardware acceleration", "LLM inference", "scaling",
    "RLHF", "reinforcement learning", "tokenization",
    "neural network", "deep learning", "language model",
]

SOURCES_TO_WATCH = {
    "arxiv_categories": ["cs.LG", "cs.CL", "cs.AI"],
    "github_repos": [
        "karpathy/nanoGPT", "karpathy/micrograd",
        "karpathy/minbpe", "karpathy/llm.c",
        "huggingface/transformers", "state-spaces/mamba",
    ],
    "blogs": [
        "https://karpathy.github.io/",
        "https://lilianweng.github.io/",
        "https://huggingface.co/blog/",
        "https://distill.pub/",
    ],
    "youtube_channels": [
        "UCbfYPyITQ-7l4upoX8nvctg",  # Andrej Karpathy
    ],
}

# ── Client (Vertex AI) ─────────────────────────────────────────────────────
_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "your-gcp-project-id")
_REGION  = os.environ.get("VERTEX_REGION_CLAUDE_4_6_SONNET", "europe-west1")
_MODEL   = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

client = anthropic.AnthropicVertex(project_id=_PROJECT, region=_REGION)


# ── Tool implementations ───────────────────────────────────────────────────

def search_arxiv(query: str, days_back: int = 1, max_results: int = 10) -> list:
    """Search arXiv for recent papers sorted by submission date.

    Retries up to 3 times on HTTP 429 (rate limit), waiting 5 s between attempts.
    A 1-second inter-call delay is enforced by the caller via _arxiv_sleep().
    """
    base = "http://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = base + params

    for attempt in range(3):
        try:
            time.sleep(1)  # polite delay before every arXiv call
            req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                tree = ET.parse(r)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            cutoff = datetime.now() - timedelta(days=days_back)
            papers = []
            for entry in tree.findall("atom:entry", ns):
                published = entry.find("atom:published", ns).text
                pub_date  = datetime.fromisoformat(published.replace("Z", ""))
                if pub_date < cutoff and days_back <= 7:
                    continue
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                papers.append({
                    "id":        arxiv_id,
                    "title":     entry.find("atom:title",   ns).text.strip(),
                    "abstract":  entry.find("atom:summary", ns).text.strip(),
                    "published": published,
                    "url":       f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}",
                })
            return papers
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                print(f"  arXiv 429 — waiting 5 s (attempt {attempt + 1}/3)")
                time.sleep(5)
                continue
            return [{"error": f"HTTP {e.code}: {e.reason}"}]
        except Exception as e:
            return [{"error": str(e)}]
    return [{"error": "arXiv rate-limited after 3 attempts"}]


def get_semantic_scholar_citations(arxiv_id: str, min_citations: int = 100) -> list:
    """Return highly-cited papers that cite a given arXiv paper."""
    try:
        url = (
            f"https://api.semanticscholar.org/graph/v1/"
            f"paper/ArXiv:{arxiv_id}/citations?"
            f"fields=title,citationCount,year,externalIds&limit=50"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        high_impact = []
        for item in data.get("data", []):
            paper = item.get("citing_paper", {})
            if paper.get("citationCount", 0) >= min_citations:
                ext_ids = paper.get("externalIds", {})
                high_impact.append({
                    "title":    paper.get("title", ""),
                    "citations": paper.get("citationCount", 0),
                    "year":     paper.get("year", ""),
                    "arxiv_id": ext_ids.get("ArXiv", ""),
                })
        return sorted(high_impact, key=lambda x: x["citations"], reverse=True)[:10]
    except Exception as e:
        return [{"error": str(e)}]


def fetch_github_readme(repo: str) -> dict:
    """Fetch the raw README of a GitHub repository."""
    try:
        url = f"https://api.github.com/repos/{repo}/readme"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3.raw",
            "User-Agent": "LLMWikiAgent/1.0",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            content = r.read().decode("utf-8")
        return {"repo": repo, "content": content[:8000], "url": f"https://github.com/{repo}"}
    except Exception as e:
        return {"error": str(e), "repo": repo}


def fetch_blog_post(url: str) -> dict:
    """Fetch and strip HTML from a blog URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "content": text[:6000]}
    except Exception as e:
        return {"error": str(e), "url": url}


def get_youtube_transcript(video_id: str) -> dict:
    """Get auto-generated transcript for a YouTube video via yt-dlp."""
    try:
        import subprocess, tempfile, glob
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "yt-dlp", "--skip-download",
                    "--write-auto-sub", "--sub-format", "vtt",
                    "--sub-lang", "en",
                    "-o", f"{tmp}/%(title)s",
                    f"https://youtube.com/watch?v={video_id}",
                ],
                capture_output=True, text=True, timeout=60,
            )
            vtt_files = glob.glob(f"{tmp}/*.vtt")
            if vtt_files:
                text = open(vtt_files[0]).read()
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\d{2}:\d{2}.*\n", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                return {"video_id": video_id, "transcript": text[:8000]}
    except Exception as e:
        return {"error": str(e), "video_id": video_id}
    return {"error": "transcript unavailable", "video_id": video_id}


def download_pdf(arxiv_id: str, filename: str) -> dict:
    """Download a PDF from arXiv into raw/."""
    out_path = RAW / filename
    if out_path.exists():
        return {"status": "already_exists", "path": str(out_path)}
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out_path.write_bytes(r.read())
        return {"status": "downloaded", "path": str(out_path), "filename": filename}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def verify_arxiv_paper(arxiv_id: str, claimed_title: str) -> tuple:
    """Fetch the real arXiv title via the API and compare with claimed_title.

    Uses the arXiv Atom API (no key required) for reliable structured output.
    Returns (is_valid: bool, real_title: str | None).
    Fails closed — returns (False, None) on any network error so hallucinated
    IDs are never silently accepted.

    Word-overlap similarity >= 0.6 required (common stopwords excluded).
    """
    try:
        params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
        url = f"http://export.arxiv.org/api/query?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            tree = ET.parse(r)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = tree.findall("atom:entry", ns)
        if not entries:
            print(f"  ⚠️  VERIFY {arxiv_id}: ID not found on arXiv — skipping")
            return (False, None)
        real_title = re.sub(r"\s+", " ", entries[0].find("atom:title", ns).text).strip()

        stopwords = {"the", "a", "an", "of", "for", "and", "in", "on", "with", "to", "via"}
        def norm(s):
            return set(re.sub(r"[^\w\s]", "", s.lower()).split()) - stopwords

        claimed_words = norm(claimed_title)
        real_words    = norm(real_title)
        overlap       = len(claimed_words & real_words)
        min_len       = min(len(claimed_words), len(real_words))
        similarity    = overlap / min_len if min_len > 0 else 0.0
        is_valid      = similarity >= 0.6

        if is_valid:
            print(f"  ✓  VERIFY {arxiv_id}: '{real_title}' (sim={similarity:.2f})")
        else:
            print(f"  ⚠️  VERIFY {arxiv_id}: claimed '{claimed_title}' vs "
                  f"real '{real_title}' (sim={similarity:.2f}) — MISMATCH, skipping")
        return (is_valid, real_title)
    except Exception as e:
        print(f"  ⚠️  Could not verify {arxiv_id}: {e} — failing closed")
        return (False, None)


def check_already_in_wiki(title: str) -> bool:
    """True if a note with a similar title already exists in wiki/."""
    needle = title.lower().replace(" ", "")
    for f in WIKI.glob("*.md"):
        if needle in f.stem.lower().replace(" ", ""):
            return True
    return False


def _sanitize_cited_by_details(content: str) -> str:
    """Strip unverified arXiv IDs from cited_by_details entries before writing a note.
    Removes individual bad arxiv: lines; does not reject the whole entry or note.
    Calls verify_arxiv_paper() for each entry that carries an arxiv field.
    """
    if "cited_by_details:" not in content:
        return content

    def strip_if_bad(m):
        chunk = m.group(0)
        arxiv_m = re.search(r'([ \t]*arxiv:\s*["\']?(\d{4}\.\d{4,5}[a-z0-9]*)["\']?[ \t]*\n)', chunk)
        if not arxiv_m:
            return chunk
        arxiv_id = arxiv_m.group(2)
        title_m = re.search(r'-\s+title:\s*["\']?(.+?)["\']?[ \t]*\n', chunk)
        if not title_m:
            return chunk
        claimed_title = title_m.group(1).strip()
        is_valid, _ = verify_arxiv_paper(arxiv_id, claimed_title)
        if not is_valid:
            return chunk.replace(arxiv_m.group(1), "")
        return chunk

    return re.sub(r'(?m)(^  - title:[^\n]*\n(?:^    [^\n]*\n)*)', strip_if_bad, content)


def write_content_note(filename: str, content: str) -> dict:
    """Write a markdown note to wiki/ for non-PDF content (GitHub, blog, YouTube)."""
    content = _strip_md_fence(content)
    out_path = WIKI / filename
    if out_path.exists():
        return {"status": "already_exists", "path": str(out_path)}
    out_path.write_text(content, encoding="utf-8")
    return {"status": "written", "path": str(out_path)}


def check_broken_links() -> dict:
    """Scan wiki/ for broken [[wikilinks]] using exact-match only.
    Reports counts; does not auto-fix (auto-fix is a human decision).
    """
    note_index: set[str] = set()
    for f in WIKI.glob("*.md"):
        note_index.add(re.sub(r"[\s_]", "-", f.stem.lower()))
        txt = f.read_text(encoding="utf-8")
        m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", txt, re.MULTILINE | re.IGNORECASE)
        if m:
            note_index.add(re.sub(r"[\s_]", "-", m.group(1).strip().lower()))

    broken: list[str] = []
    for f in sorted(WIKI.glob("*.md")):
        txt = f.read_text(encoding="utf-8")
        # Skip inline code spans and fenced blocks so template placeholders
        # like `[[${name}]]` are not counted as broken wikilinks.
        cleaned = re.sub(r"`[^`\n]+`", lambda m: " " * len(m.group()), txt)
        cleaned = re.sub(r"```.*?```", lambda m: " " * len(m.group()), cleaned, flags=re.DOTALL)
        for m in re.finditer(r"\[\[([^\]]+)\]\]", cleaned):
            target = m.group(1).split("|")[0].split("#")[0].strip()
            key = re.sub(r"[\s_]", "-", target.lower())
            if key not in note_index:
                broken.append(f"{f.name} → [[{target}]]")

    if broken:
        print(f"  ⚠️  {len(broken)} broken wikilinks detected (run link audit to fix)")
    return {"broken": len(broken)}


def log_agent_run(results: dict):
    """Append a run record to agent_log.json."""
    log = []
    if LOG.exists():
        try:
            log = json.loads(LOG.read_text())
        except Exception:
            pass
    log.append({"timestamp": datetime.now().isoformat(), **results})
    LOG.write_text(json.dumps(log, indent=2))


# ── Tool schemas for Claude ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search arXiv for recent papers by keyword or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "days_back":   {"type": "integer", "default": 1,  "description": "How many days back to include"},
                "max_results": {"type": "integer", "default": 10, "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_citations",
        "description": "Get high-impact papers (≥ min_citations) that cite an existing wiki paper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id":      {"type": "string",  "description": "arXiv ID of the source paper"},
                "min_citations": {"type": "integer", "default": 100, "description": "Minimum citation count"},
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "fetch_github_readme",
        "description": "Fetch README text from a GitHub repository (owner/repo).",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "GitHub repo in owner/repo format"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "fetch_blog_post",
        "description": "Fetch and extract text from a blog post URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the blog post"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_youtube_transcript",
        "description": "Get auto-generated transcript of a YouTube video.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "YouTube video ID (11-char string)"},
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "download_pdf",
        "description": "Download an arXiv paper PDF into raw/ for later processing by build_wiki.py.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID, e.g. 2312.00752"},
                "filename":  {"type": "string", "description": "Output filename, e.g. Mamba.pdf"},
            },
            "required": ["arxiv_id", "filename"],
        },
    },
    {
        "name": "check_wiki",
        "description": "Check if content with a given title already exists in the wiki.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title to check"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "write_note",
        "description": (
            "Write a markdown note directly to wiki/ for non-PDF content "
            "(GitHub repos, blog posts, YouTube transcripts). "
            "Include full YAML frontmatter: title, tags, year, tldr, wikilinks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output filename, e.g. nanoGPT.md"},
                "content":  {"type": "string", "description": "Full markdown content including YAML frontmatter"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "done",
        "description": "Signal the agent is finished. Provide a summary plus lists of added and skipped items.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "added":   {"type": "array", "items": {"type": "string"}},
                "skipped": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "added", "skipped"],
        },
    },
]


# ── Tool dispatcher ────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    if name == "search_arxiv":
        result = search_arxiv(**inputs)
    elif name == "get_citations":
        result = get_semantic_scholar_citations(
            inputs["arxiv_id"], inputs.get("min_citations", 100)
        )
    elif name == "fetch_github_readme":
        result = fetch_github_readme(inputs["repo"])
    elif name == "fetch_blog_post":
        result = fetch_blog_post(inputs["url"])
    elif name == "get_youtube_transcript":
        result = get_youtube_transcript(inputs["video_id"])
    elif name == "download_pdf":
        arxiv_id     = inputs["arxiv_id"]
        filename     = inputs["filename"]
        claimed_title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
        claimed_title = re.sub(r"[_\-]", " ", claimed_title).strip()
        is_valid, real_title = verify_arxiv_paper(arxiv_id, claimed_title)
        if not is_valid:
            result = {
                "status":        "skipped_unverified",
                "arxiv_id":      arxiv_id,
                "claimed_title": claimed_title,
                "real_title":    real_title,
                "message":       (
                    "Title mismatch between claimed and real arXiv entry "
                    "(or ID not found); paper not downloaded"
                ),
            }
        else:
            result = download_pdf(arxiv_id, filename)
    elif name == "check_wiki":
        result = {"exists": check_already_in_wiki(inputs["title"])}
    elif name == "write_note":
        clean_content = _sanitize_cited_by_details(inputs["content"])
        result = write_content_note(inputs["filename"], clean_content)
    elif name == "done":
        result = inputs
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ── System prompts ─────────────────────────────────────────────────────────

def _base_context(existing: list[str]) -> str:
    return (
        f"Wiki themes: {', '.join(WIKI_THEMES)}.\n"
        f"Already in wiki (skip duplicates): {', '.join(existing[:30])}.\n\n"
        "For each candidate:\n"
        "1. call check_wiki first to avoid duplicates\n"
        "2. score relevance 1–10; only proceed if ≥ 7\n"
        "3. for arXiv papers → download_pdf\n"
        "4. for GitHub/blog/YouTube → write_note with full YAML frontmatter "
        "(title, authors, year, tags, tldr) and [[wikilinks]] to existing notes\n"
        "5. call done when finished with summary, added list, skipped list"
    )


def _make_system(mode: str, extra: str, existing: list[str]) -> list[dict]:
    text = (
        f"You are a research ingestion agent for an AI/ML knowledge wiki. "
        f"Mode: {mode}.\n\n"
        + extra
        + "\n\n"
        + _base_context(existing)
    )
    # Cache the (large, stable) system prompt
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# ── Citation refresh (Semantic Scholar) ─────────────────────────────────────

def fetch_citations_bulk() -> dict:
    """Fetch current citation counts from the Semantic Scholar batch API for every
    wiki note that has an arXiv ID in its frontmatter. Free, no key required.

    Returns {filename: {arxiv_id, citation_count, title}}.
    """
    # Collect notes that have an arXiv ID in their YAML frontmatter ONLY.
    # (Searching the whole file would wrongly match body lines like
    # "arXiv: 2312.00752 | Authors: …" in meta notes such as Citation Map.md.)
    papers = {}
    for filename in sorted(os.listdir(WIKI)):
        if not filename.endswith(".md"):
            continue
        content = (WIKI / filename).read_text(encoding="utf-8")
        if not content.startswith("---"):
            continue
        fm_end = content.find("\n---", 3)
        if fm_end == -1:
            continue
        frontmatter = content[3:fm_end]
        m = re.search(
            r'^arxiv(?:_id)?:\s*["\']?(\d{4}\.\d{4,5})["\']?',
            frontmatter, re.MULTILINE | re.IGNORECASE,
        )
        if m:
            cc = re.search(r"^citation_count:\s*(\d+)", frontmatter, re.MULTILINE)
            papers[filename] = (m.group(1), int(cc.group(1)) if cc else 0)

    # Dedup notes that share an arXiv ID (e.g. a paper note + a same-titled stub):
    # keep only the canonical one (highest current citation_count) so we never
    # write the same count into two notes and create duplicate explorer entries.
    by_arxiv = {}
    for filename, (aid, cur) in papers.items():
        if aid not in by_arxiv or cur > by_arxiv[aid][1]:
            by_arxiv[aid] = (filename, cur)
    papers = {fname: aid for aid, (fname, _) in by_arxiv.items()}

    print(f"Found {len(papers)} unique papers with arXiv IDs")

    results = {}
    errors = []
    items = list(papers.items())
    batch_size = 10

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        paper_ids = [f"ArXiv:{aid}" for _, aid in batch]
        try:
            payload = json.dumps({"ids": paper_ids}).encode("utf-8")
            url = ("https://api.semanticscholar.org/graph/v1/paper/batch"
                   "?fields=citationCount,title,year")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "LLMWikiAgent/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                response = json.loads(r.read())

            for (filename, aid), entry in zip(batch, response):
                if not entry:
                    continue  # null = paper not found on Semantic Scholar
                count = entry.get("citationCount")
                if count is None:
                    continue
                title = entry.get("title", filename)
                results[filename] = {
                    "arxiv_id": aid,
                    "citation_count": int(count),
                    "title": title,
                }
                print(f"  {title[:40]:40} → {int(count):,}")

            if i + batch_size < len(items):
                time.sleep(2)  # be polite to the unauthenticated rate limit
        except Exception as e:
            errors.append(f"batch@{i}: {e}")
            print(f"  ⚠️  batch error: {e}")
            time.sleep(5)

    if errors:
        print(f"\nErrors ({len(errors)}): {errors}")
    return results


def update_citations_in_wiki(citation_data: dict) -> dict:
    """Write fresh citation_count values into wiki note frontmatter.
    Returns {updated: [...], unchanged: int, total: int}."""
    updated = []
    unchanged = 0

    for filename, data in citation_data.items():
        path = WIKI / filename
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        new_count = data["citation_count"]

        m = re.search(r"^citation_count:\s*(\d+)\s*$", content, re.MULTILINE)
        if m:
            current = int(m.group(1))
            if current == new_count:
                unchanged += 1
                continue
            content = re.sub(r"^citation_count:\s*\d+\s*$",
                             f"citation_count: {new_count}",
                             content, count=1, flags=re.MULTILINE)
        elif re.search(r"^tags:", content, re.MULTILINE):
            current = 0
            content = re.sub(r"^(tags:)", f"citation_count: {new_count}\n\\1",
                             content, count=1, flags=re.MULTILINE)
        else:
            continue  # no place to put it; skip rather than corrupt the file

        path.write_text(content, encoding="utf-8")
        diff = new_count - current
        updated.append({
            "file": filename, "title": data["title"],
            "old": current, "new": new_count,
            "diff": f"{'+' if diff >= 0 else ''}{diff:,}",
        })

    return {"updated": updated, "unchanged": unchanged, "total": len(citation_data)}


# ── Main agent loop ────────────────────────────────────────────────────────

def run_agent(mode: str = "daily", topic: str = None, min_citations: int = 100) -> dict:
    existing = [f.stem for f in WIKI.glob("*.md")]

    # Deterministic (non-LLM) mode: refresh citation counts from Semantic Scholar.
    if mode == "update-citations":
        print("Fetching latest citation counts from Semantic Scholar…")
        citation_data = fetch_citations_bulk()
        if not citation_data:
            return {"mode": mode, "topic": None, "added": [], "skipped": [],
                    "message": "No papers with arXiv IDs found (or API unreachable)",
                    "timestamp": datetime.now().isoformat()}
        changes = update_citations_in_wiki(citation_data)
        print(f"\nCitation update — {len(changes['updated'])} changed, "
              f"{changes['unchanged']} unchanged")
        for c in changes["updated"]:
            print(f"  {c['title'][:35]:35} {c['old']:>8,} → {c['new']:>8,} ({c['diff']})")
        added = [f"{c['file']} ({c['diff']})" for c in changes["updated"]]
        return {"mode": mode, "topic": None, "added": added, "skipped": [],
                "timestamp": datetime.now().isoformat()}

    if mode == "topic" and topic:
        extra = f'The user wants to add content about: "{topic}". Search for the most relevant and high-quality papers, repos, or blog posts on this topic.'
        messages = [{"role": "user", "content": f"Add content about: {topic}"}]

    elif mode == "citations":
        wiki_arxiv_ids = {
            "2312.00752": "Mamba",
            "2405.21060": "Transformers are SSMs (Mamba-2)",
            "2405.04517": "xLSTM",
            "2205.14135": "FlashAttention",
            "2307.08691": "FlashAttention-2",
            "2111.00396": "S4",
            "2305.13048": "RWKV",
            "2307.08621": "RetNet",
        }
        extra = (
            f"Check each wiki paper for highly-cited new papers that cite it "
            f"(min {min_citations} citations). For each high-impact citing paper NOT already in the wiki, download it."
        )
        messages = [{"role": "user", "content": f"Check citations (threshold={min_citations}): {json.dumps(wiki_arxiv_ids)}"}]

    else:  # daily
        extra = (
            f"Search arXiv for papers from the last 24 hours in: {', '.join(WIKI_THEMES[:8])}. "
            f"Also check these GitHub repos for notable updates: {', '.join(SOURCES_TO_WATCH['github_repos'][:4])}."
        )
        messages = [{"role": "user", "content": "Run daily monitoring check."}]

    system = _make_system(mode, extra, existing)

    added   = []
    skipped = []

    for iteration in range(20):
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        finished     = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"  → {block.name}({json.dumps(block.input)[:100]})")
            result = execute_tool(block.name, block.input)

            if block.name == "done":
                data    = json.loads(result)
                added   = data.get("added", [])
                skipped = data.get("skipped", [])
                finished = True
            elif block.name == "download_pdf":
                r = json.loads(result)
                if r.get("status") == "downloaded":
                    added.append(r["filename"])
            elif block.name == "write_note":
                r = json.loads(result)
                if r.get("status") == "written":
                    added.append(r["path"])

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if finished:
            break

        messages.append({"role": "user", "content": tool_results})

    return {
        "mode":    mode,
        "topic":   topic,
        "added":   added,
        "skipped": skipped,
        "timestamp": datetime.now().isoformat(),
    }


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode  = sys.argv[1] if len(sys.argv) > 1 else "daily"

    # citations mode accepts an optional numeric threshold:
    #   python3 agent.py citations 200
    min_citations = 100
    topic = None
    if mode == "citations" and len(sys.argv) > 2:
        try:
            min_citations = int(sys.argv[2])
        except ValueError:
            pass
    elif mode != "citations":
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    print(f"\nRunning agent — mode: '{mode}'" + (f", topic: '{topic}'" if topic else "")
          + (f", min_citations: {min_citations}" if mode == "citations" else ""))
    print("-" * 60)

    results = run_agent(mode=mode, topic=topic, min_citations=min_citations)
    log_agent_run(results)

    print(f"\nAdded  : {len(results['added'])} item(s)")
    for item in results["added"]:
        print(f"  + {item}")
    print(f"Skipped: {len(results['skipped'])} item(s)")
    for item in results["skipped"]:
        print(f"  - {item}")

    if results["added"]:
        print("\nRebuilding wiki and pushing…")
        import subprocess
        # update-citations only edits frontmatter — skip the PDF→note build so the
        # commit stays limited to citation changes.
        if mode != "update-citations":
            subprocess.run(["python3", "build_wiki.py"], cwd=VAULT)
        subprocess.run(["python3", "build.py"], cwd=VAULT)
        subprocess.run(["git", "add", "."], cwd=VAULT)
        commit_msg = (
            f"Auto-update citations — {len(results['added'])} papers"
            if mode == "update-citations"
            else f"Agent ({mode}): added {len(results['added'])} item(s)"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=VAULT)
        subprocess.run(["git", "push", "origin", "main"], cwd=VAULT)
        print("Done — wiki updated and pushed.")
        check_broken_links()  # report link health after every successful agent run
