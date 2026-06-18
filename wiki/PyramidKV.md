---
created: "2026-06-11"
title: "PyramidKV"
aliases: ["PyramidKV"]
year: 2024
tags: [kv-cache, compression, inference, stub]
tldr: "Layer-wise KV budget allocation shaped like a pyramid: lower layers get more KV slots (they handle local patterns), upper layers get fewer (they handle global abstractions)."
---

## TL;DR
Transformers exhibit a natural pattern: lower layers attend locally, upper layers attend globally. PyramidKV exploits this by allocating more KV budget to early layers (where dense local attention matters) and progressively fewer slots to later layers — matching the budget to where it's actually used.

## See Also
[[KV Cache]] · [[SnapKV]] · [[H2O eviction]] · [[GQA]]
