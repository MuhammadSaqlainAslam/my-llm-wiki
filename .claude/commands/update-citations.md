---
description: Refresh citation_count for all wiki papers from Semantic Scholar (deterministic, no LLM — auto-commits if any counts changed)
allowed-tools: Bash(python3 agent.py update-citations:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py update-citations
```

Deterministic, no LLM call. Auto-commits and pushes only if citation_count values actually
changed — but there is no diff-review step before that push either way.

After running, spot-check for implausible values (a real past incident: a Semantic Scholar
ID mismatch corrupted `cited_by_details` arXiv IDs for VMamba/Zamba in `Mamba.md`):
```python
import json
notes = json.load(open('docs/notes.json'))
for n in notes:
    if n.get('citation_count', 0) > 100 and n.get('arxiv', '').startswith(('25', '26')):
        print(f"{n['title'][:45]:45} {n['citation_count']:>8,}")
```
