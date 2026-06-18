---
created: "2026-05-12"
title: "Novel Attention Mechanisms"
tags: [kv-cache, attention, gqa, mqa, csa]
tldr: "The fourth family of [[KV Cache Optimization|KV-cache optimization]]: change the attention formulation itself so the cache is structurally smaller. Includes [[GQA]], [[Multi-Query Attention]], [[Compressed Sparse Attention]], [[Heavily Compressed Attention]]."
---

# Novel Attention Mechanisms

Cache eviction and compression are post-hoc tricks — the model was trained with full multi-head attention, you're just lossy at serve time. Novel attention mechanisms change the architecture so the cache is structurally smaller from the beginning. [[Multi-Query Attention]] shares K/V across all heads — 32× cache reduction. [[GQA]] is the practical compromise — group-shared K/V, ~4× reduction. [[Compressed Sparse Attention]] and [[Heavily Compressed Attention]] go further: compress every $m$ tokens into one entry, then sparse-select or do dense attention over the compressed sequence. Most invasive (require retraining or fine-tuning) but give the largest memory wins.

## Where it appears

- [[KV Cache Optimization]]
- [[DeepSeek_V4]]

---

*Related: [[GQA]] · [[Multi-Query Attention]] · [[Compressed Sparse Attention]] · [[Heavily Compressed Attention]]*
