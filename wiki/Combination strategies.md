---
created: "2026-05-12"
title: "Combination Strategies"
tags: [kv-cache, production, stacking]
tldr: "The fifth family of [[KV Cache Optimization|KV-cache optimization]]: stack two or more techniques (e.g., GQA + INT4 + paged + sliding window with sinks). What real production stacks actually use."
---

# Combination Strategies

No single KV-cache trick gives you more than ~4× reduction without an unacceptable quality hit. Real production stacks therefore combine techniques: [[GQA]] (architectural, ~4×) + INT4 [[Cache compression]] (~4×) + [[Paged attention]] (no compression but eliminates fragmentation) + sliding window with [[Attention sinks]] (modest eviction, ~2×). Composed, these yield 10–20× cache reduction at <2 perplexity points lost. The composition is non-trivial: each technique has interaction effects with the others, and no theoretical framework predicts how they will combine — production teams largely discover the recipe empirically.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[GQA]] · [[Cache eviction]] · [[Cache compression]] · [[Paged attention]]*
