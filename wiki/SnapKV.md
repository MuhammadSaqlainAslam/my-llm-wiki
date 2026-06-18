---
created: "2026-06-11"
title: "SnapKV"
aliases: ["SnapKV"]
year: 2024
tags: [kv-cache, compression, inference, stub]
tldr: "Snapshot KV compression: identifies critical KV positions via attention pattern voting across heads, retaining a fixed-size snapshot per layer."
---

## TL;DR
SnapKV pools attention weights across all heads to vote on which KV positions are globally important, then keeps only those — creating a compact snapshot. Unlike recency-biased eviction, SnapKV retains semantically critical early tokens.

## See Also
[[KV Cache]] · [[H2O eviction]] · [[PyramidKV]] · [[GQA]]
