---
created: "2026-06-11"
title: "Hymba"
aliases: ["Hymba"]
year: 2024
tags: [ssm, hybrid, attention, architecture, stub]
tldr: "Hybrid Mamba-attention that puts SSM and attention heads in parallel within each layer, letting them specialize: SSM for recall, attention for precision."
---

## TL;DR
Hymba runs Mamba and attention heads side-by-side within every layer rather than alternating them. The two mechanisms complement each other — SSM accumulates long-range patterns, attention focuses on precise token interactions.

## See Also
[[Mamba]] · [[Jamba]] · [[State Space Models]] · [[GQA]]
