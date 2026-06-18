---
created: "2026-06-11"
title: "GoldFinch"
aliases: ["GoldFinch"]
year: 2024
tags: [ssm, hybrid, kv-cache, recurrence, architecture, stub]
tldr: "Hybrid linear recurrence + KV cache: a RWKV-style recurrent layer feeds a compressed state into a small attention layer with bounded KV, combining sub-quadratic training with O(1) inference."
---

## TL;DR
GoldFinch stacks a RWKV-7-style recurrent block with a small local attention block that has a capped KV cache. The recurrent layer compresses history cheaply; attention resolves short-range interactions precisely. Inference cost is O(1) per step.

## See Also
[[RWKV]] · [[Mamba]] · [[KV Cache]] · [[State Space Models]] · [[Jamba]]
