---
created: "2026-05-12"
title: "Token Budget"
tags: [kv-cache, eviction, memory, deployment]
tldr: "A hard cap on the number of KV-cache entries the serving system will retain. The eviction-family knob you actually tune in production — set the budget, the policy decides which entries get dropped to fit it."
---

# Token Budget

Production LLM-serving systems run on fixed memory budgets — "32 GB of HBM for KV cache, no more." Translated to entries, that's a **token budget**: the maximum number of $(K_i, V_i)$ pairs the cache can hold at any moment. When the budget is exceeded, the configured [[Eviction policy]] (sliding window, H2O, sinks-aware, etc.) decides which entries to drop. Budgets are typically per-request or per-tenant; multi-tenant servers enforce per-tenant budgets so one user's million-token document can't starve everyone else.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Cache eviction]] · [[Eviction policy]]*
