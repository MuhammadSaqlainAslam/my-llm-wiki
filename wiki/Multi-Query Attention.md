---
created: "2026-05-12"
title: "Multi-Query Attention (MQA)"
tags: [attention, kv-cache, gqa, efficiency]
tldr: "All query heads share a single K/V projection — $H$× reduction in KV-cache size. The extreme of [[GQA]]; standard in deployment of PaLM and several other production models."
---

# Multi-Query Attention (MQA)

Multi-head attention has $H$ query heads, $H$ key heads, $H$ value heads. **Multi-Query Attention** (MQA) collapses this: keep $H$ query heads but use only **1** key head and **1** value head, shared across all queries. The KV cache shrinks by a factor of $H$ — for 32 heads, 32× — at the cost of expressivity (every head attends with the same K/V). Empirically the quality loss is small for many tasks; MQA was used in PaLM and Falcon. [[GQA]] is the practical compromise: $g$ groups of heads share K/V instead of all heads sharing. MQA = GQA with $g = 1$; full MHA = GQA with $g = H$.

## Where it appears

- [[KV Cache Optimization]]
- [[GQA]]

---

*Related: [[GQA]] · [[KV Cache]] · [[Transformer]]*
