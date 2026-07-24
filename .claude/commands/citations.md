---
description: Check 8 foundational wiki papers for new high-impact citing papers (LLM-driven, auto-commits if anything is added)
argument-hint: "[min_citations] (default 100)"
allowed-tools: Bash(python3 agent.py citations:*)
---

If the current directory isn't the wiki repo root (no `agent.py` here), find and `cd` into it first.

Run:
```bash
python3 agent.py citations $ARGUMENTS
```

This checks ONLY these 8 hardcoded papers for citing papers above the threshold: Mamba,
Mamba-2 ("Transformers are SSMs"), xLSTM, FlashAttention, FlashAttention-2, S4, RWKV,
RetNet. It does not cover the rest of the wiki — use `/topic` for other papers' citation
graphs.

Auto-commits and pushes only if any new citing papers were added.
