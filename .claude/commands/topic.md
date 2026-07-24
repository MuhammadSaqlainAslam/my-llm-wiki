---
description: Search arXiv/GitHub/blogs for papers on a topic and add relevant ones to the wiki (LLM-driven, auto-commits if anything is added)
argument-hint: <topic>
allowed-tools: Bash(python3 agent.py topic:*), Bash(python3 build_wiki.py:*), Bash(git status:*), Bash(git log:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py topic "$ARGUMENTS"
```

This is LLM-driven. It searches arXiv/GitHub/blogs for the given topic, scoring relevance
1–10 (only proceeds at ≥7). It may call `download_pdf` (raw PDF only — run
`python3 build_wiki.py` afterward to convert it into a note) or `write_note` (writes
directly to `wiki/`). It auto-commits and pushes at the end **only if** something was added.

After it finishes:
- Verify any new paper's arXiv ID and author list directly via the arXiv API
  (`http://export.arxiv.org/api/query?id_list=<id>`) — never trust LLM-recalled authors,
  they are frequently wrong (fabricated or garbled names).
- Check any new `[[WikiLink]]` targets actually resolve to a real file in `wiki/` —
  filenames and titles often diverge from a paper's casual short name.
