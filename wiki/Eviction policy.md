---
created: "2026-05-12"
title: "Eviction Policy"
tags: [kv-cache, eviction, lru, attention-score]
tldr: "The rule that decides *which* KV-cache entries to drop when evicting. Common choices: LRU (drop oldest), attention-score-based ([[H2O eviction]]), attention-sink-aware (StreamingLLM)."
---

# Eviction Policy

Once you've decided to evict KV-cache entries to fit a budget, you need a policy: which entries get dropped? LRU (least-recently-used) is the operating-systems default — drop oldest first — but for LLMs it discards the [[Attention sinks]] that anchor every head, and quality collapses. Better policies use attention scores: track which past tokens received the most attention mass over recent decoding steps and drop the ones that received the least. [[H2O eviction]] is the canonical example. The most robust policies combine rules: always retain the first 4 tokens (sinks) and the most recent $W$ tokens (locality), evict from the middle by attention score.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Cache eviction]] · [[H2O eviction]] · [[Attention sinks]] · [[Sliding window attention]]*
