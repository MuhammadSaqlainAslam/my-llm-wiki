---
created: "2026-05-12"
title: "Cache Eviction"
tags: [kv-cache, inference, eviction, memory]
tldr: "Remove KV-cache entries during generation to keep memory bounded — the simplest family of [[KV Cache Optimization|KV-cache optimization]]. Trade-off: lower peak memory at the cost of recoverability (evicted tokens can't be re-attended)."
---

# Cache Eviction

When the KV cache hits a memory budget, eviction policies decide which entries to drop. The naïvest is sliding-window — keep the last $W$ tokens — but that destroys long-range recall. Smarter policies (H2O, StreamingLLM, attention-sink retention) use the attention scores themselves to identify which tokens carry information vs. which are noise. The crucial empirical finding is that the **first few tokens** of every sequence (the [[Attention sinks]]) accumulate disproportionate attention mass and must be kept; evict them and quality collapses. Cache eviction is cheap to deploy (no retraining) but always lossy — for accuracy-critical workloads, prefer [[Cache compression]] or [[Novel attention mechanisms]] instead.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Eviction policy]] · [[Sliding window attention]] · [[H2O eviction]] · [[Attention sinks]]*
