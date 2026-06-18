---
created: "2026-06-11"
title: "CLA"
aliases: ["CLA", "Cross-Layer Attention"]
year: 2024
tags: [kv-cache, architecture, cross-layer, stub]
tldr: "Cross-Layer Attention: adjacent Transformer layers share one KV cache instead of each maintaining their own, halving KV memory with minimal quality loss."
---

## TL;DR
CLA groups pairs of consecutive layers to share a single KV cache. Layer N computes keys/values, layer N+1 reuses them via a cross-attention read instead of projecting its own. KV memory drops by ~50%; quality loss is <1% on standard benchmarks.

## See Also
[[KV Cache]] · [[GQA]] · [[Multi-Query Attention]] · [[KVQuant]]
