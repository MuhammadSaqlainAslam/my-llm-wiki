---
created: "2026-05-12"
title: "H2O Eviction (Heavy Hitter Oracle)"
aliases: ["H2O", "Heavy Hitter Oracle"]
tags: [kv-cache, eviction, attention-score]
tldr: "An [[Eviction policy]] that drops KV-cache entries based on accumulated attention scores: the \"heavy hitters\" (highest cumulative attention received) stay, the rest are evicted. Budget-controlled, mild quality loss."
---

# H2O Eviction (Heavy Hitter Oracle)

H2O ranks every cache entry by the sum of attention weights it has received across all prior decoding steps — the entries that historically mattered most are kept ("heavy hitters"), the rest are evicted as the budget tightens. Compared to sliding-window, H2O retains semantically important early tokens that would otherwise scroll off. Compared to keeping everything, H2O bounds memory at a configurable budget. Quality loss is mild but non-zero — for accuracy-critical workloads use [[Cache compression]] instead.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Eviction policy]] · [[Cache eviction]] · [[Attention sinks]]*
