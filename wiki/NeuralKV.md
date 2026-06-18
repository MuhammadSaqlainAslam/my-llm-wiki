---
created: "2026-06-11"
title: "NeuralKV"
aliases: ["NeuralKV"]
year: 2025
tags: [kv-cache, hardware, architecture, stub]
tldr: "Hardware-integrated KV cache management: moves KV eviction and compression decisions into the memory controller rather than the GPU compute stack."
---

## TL;DR
NeuralKV offloads KV cache management (eviction, compression, retrieval) to a hardware unit co-located with DRAM, removing it from the GPU's critical path and enabling finer-grained cache control without host-side overhead.

## See Also
[[KV Cache]] · [[H2O eviction]] · [[SnapKV]] · [[Hardware Acceleration for Neural Networks]]
