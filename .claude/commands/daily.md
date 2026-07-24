---
description: Run the daily arXiv/GitHub sweep for new papers matching wiki themes (LLM-driven, auto-commits if anything is added)
allowed-tools: Bash(python3 agent.py daily:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py daily
```

Searches arXiv for papers from the last 24 hours, plus a handful of watched GitHub repos.
Auto-commits and pushes only if something was added.

⚠️ Known failure mode: a paper submitted to arXiv hours earlier can look "unverifiable"
simply because web search hasn't indexed it yet — that is NOT the same as a hallucination.
Always confirm directly via the arXiv API
(`http://export.arxiv.org/api/query?id_list=<id>`) before treating a paper as fake.
`verify_arxiv_paper()` already gates every `download_pdf` call for this reason (added
after a real 2026-06-16 incident where 3 of 4 quarantined papers turned out to be real).
