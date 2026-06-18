---
created: "2026-06-11"
title: "Zamba"
aliases: ["Zamba"]
year: 2024
tags: [ssm, hybrid, architecture, stub]
tldr: "Compact 7B SSM-Transformer hybrid; interleaves Mamba blocks with shared attention layers to match Transformer quality at lower inference cost."
---

## TL;DR
Zamba is a 7B parameter model that alternates Mamba SSM blocks with a single shared self-attention layer reused across the stack — cutting the KV cache overhead while preserving the global mixing that attention provides.

## See Also
[[Mamba]] · [[Jamba]] · [[RWKV]] · [[State Space Models]]
