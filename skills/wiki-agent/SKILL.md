---
name: wiki-agent
description: Operational guide for the agent.py modes that maintain the HHRI-AI Research Wiki (topic, citations, daily, update-citations, conference, audit-citations). Use whenever the user mentions running agent.py, updating the wiki, adding papers, citation updates, audit-citations, daily sweep, conference sweep, topic search, weekly maintenance, or the wiki pipeline on this server. Also trigger on "my wiki", "run the agent", "update citations", "check broken links", "add this paper to the wiki".
---

# HHRI-AI Research Wiki — Agent Operations Skill

## Shared Context

**Repo**: `/work/HHRI-AI/Saqlain/my-wiki/` (this server)
**Remote**: `github.com:MuhammadSaqlainAslam/my-llm-wiki.git`
**Vertex AI**: configured via `ANTHROPIC_VERTEX_PROJECT_ID` / `VERTEX_REGION_CLAUDE_4_6_SONNET`
in `.env` (region defaults to `europe-west1` in code if unset — see `.env.example`)
**Notes**: 284 as of 2026-07-22 (`python3 -c "import json; print(len(json.load(open('docs/notes.json'))))"`)

`agent.py` requires the `anthropic` package (`pip install anthropic pymupdf markdown pyyaml`
per README) — not guaranteed to be installed in every shell; check with
`python3 -c "import anthropic"` before assuming a mode will run.

### Two build scripts — don't conflate them
- `build_wiki.py` — LLM-powered PDF → wiki note converter. Reads unconverted PDFs from
  `raw/`, asks Claude to write a structured note, saves to `wiki/`. Used when a paper was
  added via the `download_pdf` tool (raw PDF only, no note yet).
- `build.py` — deterministic `wiki/*.md` → `docs/notes.json` compiler for the graph viewer.
  Always run this after ANY manual edit to a `.md` file — it's what makes edits visible on
  the live demo.

Manual edit workflow:
```bash
python3 build.py
git add .
git commit -m "your message"
git push origin main
```

### Verification standard — before adding or trusting ANY paper metadata
Never write author names or arXiv IDs from memory — LLM-recalled author lists are
frequently wrong (fabricated or garbled names), confirmed repeatedly across past sessions.
Always pull from the API response:
```python
import urllib.request, re
url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
req = urllib.request.Request(url, headers={'User-Agent': 'WikiVerify/1.0'})
with urllib.request.urlopen(req, timeout=15) as r:
    xml = r.read().decode('utf-8', errors='ignore')
m = re.search(r'<entry>.*?<title>(.+?)</title>', xml, re.DOTALL)
a = re.findall(r'<name>(.+?)</name>', xml)
print(m.group(1).strip() if m else 'NOT FOUND', a)
```
Also check `wiki/` for the actual link target before writing a `[[WikiLink]]` — filenames
and titles frequently diverge from the short display name a paper is casually known by
(e.g. `[[LoRA]]` doesn't resolve; the real file is
`LoRA Low-Rank Adaptation of Large Language Models.md`, needs an alias link).

---

## Mode: Topic Search

```bash
python3 agent.py topic "your topic here"
```
LLM-driven. Searches arXiv/GitHub/blogs for the given topic, scores relevance 1–10 (only
proceeds ≥7 per the system prompt), can call `download_pdf` (PDF only, needs `build_wiki.py`
after) or `write_note` (writes directly to `wiki/`). Auto-commits and pushes at the end of
`__main__` *only if* `results["added"]` is non-empty.

---

## Mode: Citation Tracking

```bash
python3 agent.py citations [min_citations]
```
LLM-driven. Checks a **hardcoded list of 8 wiki papers** (Mamba, Mamba-2/"Transformers are
SSMs", xLSTM, FlashAttention, FlashAttention-2, S4, RWKV, RetNet) for citing papers above
`min_citations` (default 100). This mode does NOT cover the rest of the wiki — for other
papers' citation graphs, use Topic Search instead. Same auto-commit-if-added behavior.

---

## Mode: Daily Monitor

```bash
python3 agent.py daily
```
LLM-driven. Searches arXiv for papers from the **last 24 hours** in `WIKI_THEMES`, plus a
handful of watched GitHub repos. Same auto-commit-if-added behavior.

**Known failure mode (real incident, 2026-06-16)**: the daily sweep quarantined 4 papers
with unverifiable-at-the-time arXiv IDs (`2606.18206/18208/18246/18056`) because web search
hadn't indexed them yet — papers submitted hours earlier aren't hallucinations just because
search can't find them. An editorial call later permanently removed them anyway (not
influential enough yet), but 3 of the 4 were confirmed real via direct arXiv API lookup
first. This is why `verify_arxiv_paper()` (word-overlap check against the live arXiv API,
fails closed on network error) now gates every `download_pdf` call — check arXiv directly,
never rely on web-search absence as a hallucination signal.

---

## Mode: Update Citation Counts

```bash
python3 agent.py update-citations
```
**Deterministic, no LLM.** Fetches fresh `citation_count` values from Semantic Scholar
(`fetch_citations_bulk`) and writes them into each note's frontmatter
(`update_citations_in_wiki`). Auto-commits and pushes **if any counts changed** — same
`results["added"]` gate as the LLM modes, not unconditional, but there is no diff-review
step before the push either way.

**Known real incident**: `Mamba.md`'s `cited_by_details` block once carried two corrupted
arXiv IDs from a Semantic Scholar ID mismatch — VMamba's ID pointed to an unrelated
"Multi-Agent Diagnostics" paper, Zamba's pointed to a "Contextual Position Encoding" paper.
A follow-up manual audit of all `cited_by_details` entries across 6 notes found 29 bad IDs
out of 60 checked. Fix shipped as `_sanitize_cited_by_details()`, called from the `write_note`
path — verifies every `cited_by_details` entry's arXiv ID against its claimed title, strips
the `arxiv:` field (keeping the entry) on mismatch.

**After running**, spot-check anything implausible:
```python
import json
notes = json.load(open('docs/notes.json'))
for n in notes:
    if n.get('citation_count', 0) > 100 and n.get('arxiv', '').startswith(('25', '26')):
        print(f"{n['title'][:45]:45} {n['citation_count']:>8,}")
```

---

## Mode: Conference Papers Sweep

```bash
python3 agent.py conference
```
LLM-driven, **report-only — never touches `wiki/`, never commits**. Searches for papers
accepted at ACL, EMNLP, NAACL, COLING, TACL, NeurIPS, ICML, ICLR, AAAI, or UAI relevant to
`WIKI_THEMES`. Each candidate goes through `propose_candidate()`, which calls the same
`verify_arxiv_paper()` gate before accepting it, and accumulates in memory only. At the end,
`write_conference_candidates_file()` overwrites `conference_candidates.txt` (gitignored,
server-only — never appends, so stale picks from a prior run never linger). `results["added"]`
stays empty for this mode, so the `__main__` auto-commit block never fires — confirmed by
reading the code, not just documentation.

To add a reviewed pick: `python3 agent.py topic "paper title"`.

**Known quota constraint**: this is the most token-intensive mode. Vertex AI 429
(`RESOURCE_EXHAUSTED`) has been hit on both `europe-west1` and `us-east5` in the same
session — same project-level quota pool, switching `VERTEX_REGION_CLAUDE_4_6_SONNET` does
not help. If quota is exhausted, either request a GCP quota increase or ask directly in
conversation for candidate papers by venue/theme — same research, zero Vertex quota cost.

---

## Mode: Audit Citation Data Integrity

```bash
python3 agent.py audit-citations
```
**Deterministic, no LLM, read-only** — never writes any file. Walks every `cited_by_details`
block in every note, fetches the real arXiv title for each entry's `arxiv_id`, and flags
mismatches/not-found entries. Entries in the `CONFIRMED_CITING_PAPERS` allowlist (manually
verified informal-name-vs-formal-title false positives, e.g. a wiki using a paper's short
nickname instead of its full title) are skipped. Reports flagged entries but modifies
nothing — always safe to run. Run it after `update-citations` if you suspect ID corruption,
or periodically as a health check; no fixed baseline count is asserted here — run it to see
current numbers rather than trusting a stale figure.

---

## Quick Reference

| Mode | LLM? | Touches wiki/? | Auto-commits? |
|---|---|---|---|
| `topic "<x>"` | ✅ | ✅ (new notes) | if `added` non-empty |
| `citations [n]` | ✅ | ✅ (new notes, 8 hardcoded source papers only) | if `added` non-empty |
| `daily` | ✅ | ✅ (new notes) | if `added` non-empty |
| `update-citations` | ❌ | ✅ (frontmatter only) | if any count changed |
| `conference` | ✅ | ❌ never | never (writes local .txt only) |
| `audit-citations` | ❌ | ❌ never | never (report only) |

Safest to run anytime: `conference`, `audit-citations`. Every LLM-driven mode is
quota-dependent; a 429 means the shared Vertex AI project quota is exhausted, not a
region-specific issue.
