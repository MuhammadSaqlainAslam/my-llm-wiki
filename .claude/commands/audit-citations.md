---
description: Verify every cited_by_details arXiv ID against the live arXiv API — deterministic, read-only, always safe
allowed-tools: Bash(python3 agent.py audit-citations:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py audit-citations
```

Deterministic, no LLM call, read-only — never writes any file. Flags mismatched or
not-found `cited_by_details` entries, skipping known false positives already in the
`CONFIRMED_CITING_PAPERS` allowlist. Safe to run anytime.
