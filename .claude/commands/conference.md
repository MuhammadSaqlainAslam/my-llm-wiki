---
description: Report-only sweep for papers accepted at major NLP/ML conferences — never touches wiki/ or commits
allowed-tools: Bash(python3 agent.py conference:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py conference
```

LLM-driven but strictly report-only: writes `conference_candidates.txt` (gitignored,
overwritten each run) and never writes to `wiki/` or commits. Covers ACL, EMNLP, NAACL,
COLING, TACL, NeurIPS, ICML, ICLR, AAAI, UAI.

To add a reviewed pick, use `/topic "paper title"`.

⚠️ Most token-intensive mode — if you hit a Vertex AI 429 (`RESOURCE_EXHAUSTED`), switching
regions will not help; it has been confirmed to be the same project-level quota pool across
regions.
