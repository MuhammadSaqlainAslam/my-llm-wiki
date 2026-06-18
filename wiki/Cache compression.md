---
created: "2026-05-12"
title: "Cache Compression"
tags: [kv-cache, inference, quantization, low-rank]
tldr: "Keep all KV-cache entries but shrink each one — typically via INT8/INT4 quantization, [[NVFP4]] storage, or low-rank projection. Saves memory without losing recoverability; the second family of [[KV Cache Optimization|KV-cache optimizations]]."
---

# Cache Compression

Where eviction throws information away, compression keeps every entry but stores it in fewer bytes. INT8 quantization of K and V is the standard production setting — ~2× smaller cache, near-zero quality loss. INT4 cuts size another 2× with mild perplexity hit and is common in long-context serving. [[NVFP4]] gives most of the INT4 win with better numerical behavior on outliers. Low-rank K/V projection takes a different angle: project K and V down to dimension $d' < d$ before storing, restore at attention time. Compression composes well with other strategies: GQA + INT4 + [[Paged attention]] is a typical production stack.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[KV Cache]] · [[NVFP4]] · [[Cache eviction]]*
